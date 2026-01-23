"""
============================================================
DEBERTA + QWEN ENSEMBLE PIPELINE
============================================================
Uses trained DeBERTa model to generate candidate predictions,
then Qwen reranks the top-K candidates for better accuracy.

Strategy:
1. DeBERTa generates top-10 candidate classes with probabilities
2. Qwen reranks these candidates using reasoning
3. Final prediction combines both models' judgments
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import pickle
from scipy.special import softmax
from tqdm import tqdm
import re

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Paths - KAGGLE READY
    DATA_PATH = "/kaggle/input/dataset2/train.csv"
    
    # DeBERTa model paths (from your training output)
    DEBERTA_MODEL_PATH = "/kaggle/input/notebook364f208049/deberta_output/final_model"
    LABEL_ENCODER_PATH = "/kaggle/input/notebook364f208049/deberta_output/label_encoder.pkl"
    
    OUTPUT_DIR = "/kaggle/working/ensemble_output"
    
    # Qwen model
    QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    # QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Smaller option
    
    # Ensemble settings
    DEBERTA_TOP_K = 10       # Number of candidates from DeBERTa
    FINAL_TOP_K = 3          # Final top-K predictions
    
    # Inference
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    
    # Debug
    DEBUG = False
    DEBUG_SAMPLES = 50

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print(" DEBERTA + QWEN ENSEMBLE PIPELINE")
print("=" * 70)
print(f"Device: {device}")
print(f"Debug mode: {config.DEBUG}")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1️⃣ LOAD DEBERTA MODEL
# ============================================================

print("\n[1/5] Loading DeBERTa Model...")
print("-" * 40)

# Load label encoder
with open(config.LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

classes = label_encoder.classes_
num_classes = len(classes)
print(f"Number of classes: {num_classes}")

# Load DeBERTa
deberta_tokenizer = AutoTokenizer.from_pretrained(config.DEBERTA_MODEL_PATH)
deberta_model = AutoModelForSequenceClassification.from_pretrained(
    config.DEBERTA_MODEL_PATH
).to(device)
deberta_model.eval()

print(f"✓ DeBERTa loaded from: {config.DEBERTA_MODEL_PATH}")

# ============================================================
# 2️⃣ LOAD QWEN MODEL
# ============================================================

print("\n[2/5] Loading Qwen Model...")
print("-" * 40)

qwen_tokenizer = AutoTokenizer.from_pretrained(
    config.QWEN_MODEL,
    trust_remote_code=True
)

qwen_model = AutoModelForCausalLM.from_pretrained(
    config.QWEN_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
qwen_model.eval()

print(f"✓ Qwen loaded: {config.QWEN_MODEL}")

# ============================================================
# 3️⃣ LOAD DATA
# ============================================================

print("\n[3/5] Loading Data...")
print("-" * 40)

df = pd.read_csv(config.DATA_PATH)

# Clean data
df["StudentExplanation"] = df["StudentExplanation"].fillna("")
df["QuestionText"] = df["QuestionText"].fillna("")
df["MC_Answer"] = df["MC_Answer"].fillna("")
df["Misconception"] = df["Misconception"].fillna("NA")
df["Category"] = df["Category"].fillna("Unknown")

# Create label - MUST match DeBERTa training format exactly
df["label"] = df["Category"] + " | " + df["Misconception"]

# Debug: Check label format
print(f"Total samples in CSV: {len(df)}")
print(f"Sample labels from CSV: {df['label'].head(3).tolist()}")
print(f"Sample classes from encoder: {list(classes[:3])}")

# Filter to classes in label encoder
df_filtered = df[df["label"].isin(classes)]
print(f"Samples after filtering: {len(df_filtered)}")

# If filtering removed everything, skip filtering (use all data)
if len(df_filtered) == 0:
    print("⚠️ WARNING: No matching labels found! Using all data without filtering.")
    print("   This may cause issues if labels don't match.")
    df_filtered = df
    # Map labels to closest match in classes
    def find_closest_class(label):
        if label in classes:
            return label
        # Try partial match
        for c in classes:
            if label.split(":")[0] == c.split(":")[0]:  # Match category
                return c
        return classes[0]  # Fallback
    
    df_filtered["label"] = df_filtered["label"].apply(find_closest_class)

# Create validation set
from sklearn.model_selection import train_test_split
_, X_val = train_test_split(df_filtered, test_size=0.15, random_state=42, stratify=df_filtered["label"])

if config.DEBUG:
    X_val = X_val.head(config.DEBUG_SAMPLES)
    print(f"⚡ DEBUG MODE: Using {len(X_val)} samples")

print(f"Validation samples: {len(X_val)}")

# ============================================================
# 4️⃣ INFERENCE FUNCTIONS
# ============================================================

def build_deberta_text(row):
    """Build input text for DeBERTa"""
    return (
        f"Question: {row['QuestionText'][:300]} | "
        f"Answer: {row['MC_Answer']} | "
        f"Explanation: {row['StudentExplanation'][:200]}"
    )


def get_deberta_candidates(row, top_k=10):
    """Get top-K candidate predictions from DeBERTa with probabilities"""
    text = build_deberta_text(row)
    
    inputs = deberta_tokenizer(
        text,
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = deberta_model(**inputs)
        logits = outputs.logits[0].cpu().numpy()
    
    probs = softmax(logits)
    top_indices = np.argsort(-probs)[:top_k]
    
    candidates = []
    for idx in top_indices:
        candidates.append({
            "label": classes[idx],
            "probability": float(probs[idx]),
            "rank": len(candidates) + 1
        })
    
    return candidates


def create_qwen_rerank_prompt(row, candidates):
    """Create prompt for Qwen to rerank DeBERTa candidates"""
    
    # Format candidates with probabilities
    candidate_list = "\n".join([
        f"  {i+1}. {c['label']} (confidence: {c['probability']:.1%})"
        for i, c in enumerate(candidates)
    ])
    
    prompt = f"""You are an expert at analyzing student math responses and identifying misconceptions.

A classification model has provided these candidate predictions, ranked by confidence:

CANDIDATES:
{candidate_list}

STUDENT RESPONSE:
- Question: {row['QuestionText'][:400]}
- Student's Answer: {row['MC_Answer']}
- Explanation: {row['StudentExplanation'][:300]}

TASK: Analyze the student's response and rerank the candidates.
- Consider the student's reasoning and any misconceptions shown
- The format is "Category:Misconception" 
- "NA" means no specific misconception (correct answer or unclear)

Return your TOP 3 choices in order of likelihood, using EXACTLY this format:
1. [exact label from candidates]
2. [exact label from candidates]  
3. [exact label from candidates]

Your reranking:"""
    
    return prompt


def parse_qwen_reranking(response, candidates):
    """Parse Qwen's reranking response"""
    
    # Get candidate labels for matching
    candidate_labels = [c['label'] for c in candidates]
    
    # Try to extract numbered items
    reranked = []
    
    # Look for patterns like "1. Label" or "1: Label"
    for line in response.split('\n'):
        match = re.search(r'^\s*[123][\.\:\)]\s*(.+)', line)
        if match:
            predicted = match.group(1).strip()
            # Match to closest candidate
            best_match = match_to_candidate(predicted, candidate_labels)
            if best_match and best_match not in reranked:
                reranked.append(best_match)
    
    # If parsing failed, fall back to DeBERTa order
    if len(reranked) < 3:
        for c in candidate_labels:
            if c not in reranked:
                reranked.append(c)
            if len(reranked) >= 3:
                break
    
    return reranked[:3]


def match_to_candidate(text, candidates):
    """Match extracted text to closest candidate"""
    text_lower = text.lower().strip()
    
    # Exact match
    for c in candidates:
        if c.lower() == text_lower:
            return c
    
    # Partial match (text contains candidate or vice versa)
    for c in candidates:
        if text_lower in c.lower() or c.lower() in text_lower:
            return c
    
    # Fuzzy match - check key parts
    for c in candidates:
        parts = c.lower().split(':')
        if any(p.strip() in text_lower for p in parts):
            return c
    
    return None


def predict_ensemble(row):
    """Make ensemble prediction: DeBERTa candidates + Qwen reranking"""
    
    # Step 1: Get DeBERTa candidates
    candidates = get_deberta_candidates(row, top_k=config.DEBERTA_TOP_K)
    
    # Step 2: Create Qwen reranking prompt
    prompt = create_qwen_rerank_prompt(row, candidates)
    
    # Step 3: Get Qwen's reranking
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes student responses."},
        {"role": "user", "content": prompt}
    ]
    
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
    
    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id
        )
    
    response = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Step 4: Parse reranking
    top3_reranked = parse_qwen_reranking(response, candidates)
    
    return {
        "deberta_top1": candidates[0]['label'],
        "deberta_top3": [c['label'] for c in candidates[:3]],
        "deberta_probs": [c['probability'] for c in candidates[:3]],
        "ensemble_top3": top3_reranked,
        "qwen_response": response
    }


def predict_deberta_only(row):
    """Faster inference using only DeBERTa (for comparison)"""
    candidates = get_deberta_candidates(row, top_k=3)
    return {
        "deberta_top1": candidates[0]['label'],
        "deberta_top3": [c['label'] for c in candidates],
        "deberta_probs": [c['probability'] for c in candidates],
    }

# ============================================================
# 5️⃣ RUN INFERENCE
# ============================================================

print("\n[4/5] Running Ensemble Inference...")
print("-" * 40)

results = []

for idx, row in tqdm(X_val.iterrows(), total=len(X_val), desc="Predicting"):
    try:
        result = predict_ensemble(row)
        result["true_label"] = row["label"]
        result["idx"] = idx
        results.append(result)
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        # Fallback to DeBERTa only
        result = predict_deberta_only(row)
        result["ensemble_top3"] = result["deberta_top3"]
        result["true_label"] = row["label"]
        result["idx"] = idx
        result["qwen_response"] = "ERROR"
        results.append(result)
    
    # Clear cache periodically
    if len(results) % 20 == 0:
        torch.cuda.empty_cache()

# ============================================================
# 6️⃣ EVALUATE
# ============================================================

print("\n[5/5] Evaluating...")
print("-" * 40)

def map_at_k(y_true, y_pred_top_k, k=3):
    """Calculate Mean Average Precision @ K"""
    scores = []
    for true, preds in zip(y_true, y_pred_top_k):
        score = 0.0
        for i, pred in enumerate(preds[:k]):
            if pred == true:
                score = 1.0 / (i + 1)
                break
        scores.append(score)
    return np.mean(scores)

# Extract results
true_labels = [r["true_label"] for r in results]
deberta_top3 = [r["deberta_top3"] for r in results]
ensemble_top3 = [r["ensemble_top3"] for r in results]

# Calculate metrics for DeBERTa alone
deberta_map3 = map_at_k(true_labels, deberta_top3, k=3)
deberta_top1_acc = np.mean([p[0] == t for p, t in zip(deberta_top3, true_labels)])
deberta_top3_acc = np.mean([t in p[:3] for p, t in zip(deberta_top3, true_labels)])

# Calculate metrics for Ensemble
ensemble_map3 = map_at_k(true_labels, ensemble_top3, k=3)
ensemble_top1_acc = np.mean([p[0] == t for p, t in zip(ensemble_top3, true_labels)])
ensemble_top3_acc = np.mean([t in p[:3] for p, t in zip(ensemble_top3, true_labels)])

print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)
print(f"{'Metric':<20} {'DeBERTa Only':<15} {'Ensemble':<15} {'Diff':<10}")
print("-" * 60)
print(f"{'MAP@3':<20} {deberta_map3:<15.4f} {ensemble_map3:<15.4f} {ensemble_map3-deberta_map3:+.4f}")
print(f"{'Top-1 Accuracy':<20} {deberta_top1_acc:<15.4f} {ensemble_top1_acc:<15.4f} {ensemble_top1_acc-deberta_top1_acc:+.4f}")
print(f"{'Top-3 Accuracy':<20} {deberta_top3_acc:<15.4f} {ensemble_top3_acc:<15.4f} {ensemble_top3_acc-deberta_top3_acc:+.4f}")
print("=" * 60)

# Save results
results_df = pd.DataFrame(results)
results_path = f"{config.OUTPUT_DIR}/ensemble_predictions.csv"
results_df.to_csv(results_path, index=False)
print(f"\n✓ Results saved to: {results_path}")

# Sample predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(min(5, len(results))):
    r = results[i]
    print(f"\nSample {i+1}:")
    print(f"  True:         {r['true_label'][:50]}...")
    print(f"  DeBERTa Top1: {r['deberta_top1'][:50]}...")
    print(f"  Ensemble Top1: {r['ensemble_top3'][0][:50]}...")
    deberta_correct = r['deberta_top1'] == r['true_label']
    ensemble_correct = r['ensemble_top3'][0] == r['true_label']
    print(f"  DeBERTa: {'✓' if deberta_correct else '✗'} | Ensemble: {'✓' if ensemble_correct else '✗'}")

print("\n" + "=" * 70)
print("✅ ENSEMBLE INFERENCE COMPLETE!")
print("=" * 70)

if config.DEBUG:
    print("\n⚠️  This was a DEBUG run. Set DEBUG=False for full evaluation.")
