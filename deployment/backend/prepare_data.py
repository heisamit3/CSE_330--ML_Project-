"""
Data preparation script for MAP Misconception Detection API.
Run this script to generate required data artifacts before deployment.

Usage:
    python prepare_data.py --train-path ../DS/train.csv --output-dir ./data
"""

import argparse
import json
import sys
from pathlib import Path


def prepare_label_encoder(train_df, output_path):
    """Create and save label encoder from training data."""
    from sklearn.preprocessing import LabelEncoder
    import joblib

    # Create target column
    train_df['target'] = train_df['Category'] + ':' + train_df['Misconception'].fillna('NA')

    # Fit label encoder
    le = LabelEncoder()
    le.fit(train_df['target'])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, output_path)
    print(f"Saved label encoder with {len(le.classes_)} classes to {output_path}")
    return le


def prepare_correct_answers(train_df, output_path):
    """Extract correct answers map from training data."""
    # Find rows where category starts with "True"
    idx = train_df['Category'].apply(lambda c: str(c).startswith('True'))
    correct = train_df.loc[idx].copy()

    # Count occurrences to find most common correct answer per question
    correct['count'] = correct.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
    correct = correct.sort_values('count', ascending=False).drop_duplicates(['QuestionId'])

    # Create mapping
    correct_answers = dict(zip(
        correct['QuestionId'].astype(str),
        correct['MC_Answer']
    ))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(correct_answers, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved {len(correct_answers)} correct answers to {output_path}")
    return correct_answers


def prepare_misconception_kb(train_df, output_path):
    """Create a basic misconception knowledge base."""
    # Get unique misconceptions
    misconceptions = train_df['Misconception'].dropna().unique()

    kb = {}
    for m in misconceptions:
        if m and m != 'NA':
            kb[m] = {
                "plain_definition": f"The student shows a pattern consistent with '{m}' type reasoning.",
                "typical_wrong_thought": [
                    f"Students with this misconception often misunderstand a key concept.",
                ],
                "correct_rule": [
                    "Review the fundamental concept involved.",
                    "Apply the correct mathematical principle step by step.",
                ],
                "fix_steps": [
                    "Identify what the question is really asking.",
                    "Write down the relevant mathematical rule or formula.",
                    "Apply the rule carefully to this specific problem.",
                    "Check your answer with a simple example.",
                ],
                "quick_check": "Does your method work correctly for a simpler version of this problem?",
                "tiny_example": "Try applying your reasoning to a basic case first."
            }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved KB with {len(kb)} misconception entries to {output_path}")
    return kb


def main():
    parser = argparse.ArgumentParser(description='Prepare data artifacts for MAP API')
    parser.add_argument('--train-path', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory')
    parser.add_argument('--kb-path', type=str, default=None, help='Path to existing misconception_kb.json (optional)')
    args = parser.parse_args()

    train_path = Path(args.train_path)
    output_dir = Path(args.output_dir)

    if not train_path.exists():
        print(f"Error: Train file not found at {train_path}")
        sys.exit(1)

    # Load training data
    import pandas as pd
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df)} rows")

    # Copy train.csv to output
    output_train = output_dir / 'train.csv'
    output_train.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_train, index=False)
    print(f"Copied train.csv to {output_train}")

    # Create label encoder
    prepare_label_encoder(train_df, output_dir / 'label_encoder.pkl')

    # Create correct answers map
    prepare_correct_answers(train_df, output_dir / 'correct_answers.json')

    # Create or copy KB
    kb_output = output_dir / 'misconception_kb.json'
    if args.kb_path and Path(args.kb_path).exists():
        # Copy existing KB
        import shutil
        shutil.copy(args.kb_path, kb_output)
        print(f"Copied existing KB from {args.kb_path}")
    else:
        # Generate basic KB
        prepare_misconception_kb(train_df, kb_output)

    print("\nData preparation complete!")
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    for f in output_dir.glob('*'):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
