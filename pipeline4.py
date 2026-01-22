# ============================================================
# LLM-BASED INFERENCE FOR MAP COMPETITION
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.special import softmax
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time

print("="*70)
print(" LLM INFERENCE - STUDENT MATH MISCONCEPTIONS")
print("="*70)

# ============================================================
# STEP 1: SETUP
# ============================================================
print("\n[1/6] Setting up environment...")

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {device}")

# Load data
print("\n[2/6] Loading data...")

# Upload file if needed
# from google.colab import files
# uploaded = files.upload()

train = pd.read_csv('train.csv')
train['StudentExplanation'] = train['StudentExplanation'].fillna('')
train['Misconception'] = train['Misconception'].fillna('NA')
train['label'] = train['Category'] + ':' + train['Misconception']
train = train.dropna(subset=['label', 'StudentExplanation'])

# Get all unique classes
class_counts = train['label'].value_counts()
rare_classes = class_counts[class_counts < 2].index
train = train[~train['label'].isin(rare_classes)]
all_classes = sorted(train['label'].unique())

print(f"✓ Total samples: {len(train)}")
print(f"✓ Unique classes: {len(all_classes)}")

# Create validation set for testing
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    train[['QuestionText', 'MC_Answer', 'StudentExplanation']], 
    train['label'],
    test_size=0.1,  # Small sample for testing
    random_state=42,
    stratify=train['label']
)

print(f"✓ Validation samples: {len(X_val)}")

# ============================================================
# STEP 2: LOAD LLM MODEL
# ============================================================
print("\n[3/6] Loading LLM model...")
print("  (This may take 2-3 minutes...)")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Change to your preferred model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Use float16 for faster inference
    device_map="auto",
    low_cpu_mem_usage=True
)
model.eval()

print(f"✓ Model loaded: {MODEL_NAME}")

# ============================================================
# STEP 3: CREATE PROMPT TEMPLATE
# ============================================================
def create_prompt(question_text, mc_answer, student_explanation, classes):
    """Create a few-shot prompt for classification"""
    
    prompt = f"""You are an expert in analyzing student math misconceptions. 
    
Given a math question, the student's answer choice, and their explanation, classify the response into ONE of these categories:

Categories:
{', '.join(classes[:20])}... (and {len(classes)-20} more)

Question: {question_text}
Student's Answer: {mc_answer}
Student's Explanation: {student_explanation}

Based on the explanation, what is the classification? Respond with ONLY the category name, nothing else.

Classification:"""
    
    return prompt

# ============================================================
# STEP 4: INFERENCE WITH PROBABILITY EXTRACTION
# ============================================================
print("\n[4/6] Running inference on validation set...")
print("  (Testing on small sample first...)")

def get_logits_for_classes(prompt, classes, model, tokenizer):
    """Get logits for each possible class"""
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]  # Last token logits
    
    # Get logits for each class token
    class_logits = []
    for class_name in classes:
        # Tokenize each class name
        class_tokens = tokenizer(class_name, add_special_tokens=False)['input_ids']
        # Use first token of class name
        if len(class_tokens) > 0:
            class_logits.append(logits[0, class_tokens[0]].item())
        else:
            class_logits.append(-float('inf'))
    
    return np.array(class_logits)

# Test on small sample (first 10 examples)
sample_size = min(10, len(X_val))
results = []

for idx in tqdm(range(sample_size), desc="Processing"):
    row = X_val.iloc[idx]
    
    # Create prompt
    prompt = create_prompt(
        row['QuestionText'],
        row['MC_Answer'],
        row['StudentExplanation'],
        all_classes
    )
    
    # Get logits for all classes
    class_logits = get_logits_for_classes(prompt, all_classes, model, tokenizer)
    
    # Convert to probabilities
    probs = softmax(class_logits)
    
    # Get top-3 predictions
    top3_idx = np.argsort(-probs)[:3]
    top3_classes = [all_classes[i] for i in top3_idx]
    top3_probs = [probs[i] for i in top3_idx]
    
    # Store results
    result = {
        'true_label': y_val.iloc[idx],
        'pred_1': top3_classes[0],
        'prob_1': top3_probs[0],
        'pred_2': top3_classes[1],
        'prob_2': top3_probs[1],
        'pred_3': top3_classes[2],
        'prob_3': top3_probs[2],
    }
    
    # Add all class probabilities for ensembling
    for i, class_name in enumerate(all_classes):
        result[f'prob_class_{i}'] = probs[i]
    
    results.append(result)

results_df = pd.DataFrame(results)

# ============================================================
# STEP 5: EVALUATE RESULTS
# ============================================================
print("\n[5/6] Evaluating results...")

# Calculate MAP@3
def map_at_3(true_labels, predictions):
    """Calculate Mean Average Precision at 3"""
    scores = []
    for true, preds in zip(true_labels, predictions):
        score = 0
        for i, pred in enumerate(preds[:3]):
            if pred == true:
                score = 1 / (i + 1)
                break
        scores.append(score)
    return np.mean(scores)

predictions = results_df[['pred_1', 'pred_2', 'pred_3']].values.tolist()
map3_score = map_at_3(results_df['true_label'], predictions)

# Top-1 accuracy
top1_acc = (results_df['true_label'] == results_df['pred_1']).mean()

print(f"\n✓ MAP@3 Score: {map3_score:.4f}")
print(f"✓ Top-1 Accuracy: {top1_acc:.4f}")

# Show some examples
print("\nSample Predictions:")
for i in range(min(3, len(results_df))):
    print(f"\n{i+1}. True: {results_df.iloc[i]['true_label']}")
    print(f"   Pred: {results_df.iloc[i]['pred_1']} (prob: {results_df.iloc[i]['prob_1']:.3f})")

# ============================================================
# STEP 6: SAVE RESULTS FOR ENSEMBLING
# ============================================================
print("\n[6/6] Saving results...")

# Save predictions with probabilities
results_df.to_csv('llm_predictions_with_probs.csv', index=False)
print("✓ Saved: llm_predictions_with_probs.csv")

print("\n" + "="*70)
print("✅ LLM INFERENCE COMPLETE!")
print("="*70)

print(f"\nSUMMARY:")
print(f"  • Model: {MODEL_NAME}")
print(f"  • Samples processed: {len(results_df)}")
print(f"  • MAP@3 Score: {map3_score:.4f}")
print(f"  • Top-1 Accuracy: {top1_acc:.4f}")
print(f"\nNext: Use these probabilities for ensembling with XGBoost/LightGBM!")