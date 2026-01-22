# ============================================================
# QWEN2.5-7B FAST RERANKER INFERENCE (KAGGLE READY)
# ============================================================

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1️⃣ SETUP
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*70)
print(" QWEN2.5-7B FAST RERANKER INFERENCE")
print("="*70)
print("Device:", device)

# ============================================================
# 2️⃣ LOAD DATA
# ============================================================

DATA_PATH = "/kaggle/input/dataset1/train.csv"  # 🔴 Update path
df = pd.read_csv(DATA_PATH)

df["StudentExplanation"] = df["StudentExplanation"].fillna("")
df["Misconception"] = df["Misconception"].fillna("NA")
df["label"] = df["Category"] + ":" + df["Misconception"]

# Remove rare classes
vc = df["label"].value_counts()
df = df[df["label"].isin(vc[vc >= 5].index)]
classes = sorted(df["label"].unique())
print("Total classes:", len(classes))

# Create validation set
X_train, X_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print("Validation samples:", len(X_val))

# ============================================================
# 3️⃣ FAST CLASSIFIER FOR TOP CANDIDATES
# ============================================================

# Use DeBERTa small for top candidates (fast)
MODEL_FAST = "microsoft/deberta-v3-small"
tokenizer_fast = AutoTokenizer.from_pretrained(MODEL_FAST)
model_fast = AutoModelForSequenceClassification.from_pretrained(
    MODEL_FAST,
    num_labels=len(classes)
).to(device)
model_fast.eval()

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(classes)

def prepare_text(row):
    return f"Question: {row['QuestionText']} | Answer: {row['MC_Answer']} | Explanation: {row['StudentExplanation']}"

texts_val = [prepare_text(r) for _, r in X_val.iterrows()]
labels_val = label_encoder.transform(X_val["label"])

# ============================================================
# 4️⃣ LOAD QWEN MODEL
# ============================================================

MODEL_QWEN = "Qwen/Qwen2.5-7B-Instruct"
tokenizer_qwen = AutoTokenizer.from_pretrained(MODEL_QWEN, trust_remote_code=True)
model_qwen = AutoModelForCausalLM.from_pretrained(
    MODEL_QWEN,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model_qwen.eval()

# ============================================================
# 5️⃣ INFERENCE FUNCTIONS
# ============================================================

def get_top_candidates(text, top_k=5):
    """Get top-K candidate labels from fast classifier"""
    inputs = tokenizer_fast(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model_fast(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    top_idx = np.argsort(-probs)[:top_k]
    return [label_encoder.classes_[i] for i in top_idx]

def make_prompt(row, candidates):
    label_block = "\n".join(f"- {l}" for l in candidates)
    return f"""
You are an expert at identifying student math misconceptions.

Choose the THREE most likely categories from the list below.
Return them in order from most likely to least likely.
Return ONLY the labels, one per line.

Categories:
{label_block}

Question: {row['QuestionText']}
Student Answer: {row['MC_Answer']}
Student Explanation: {row['StudentExplanation']}

Top 3 categories:
"""

def qwen_rerank(row, candidates):
    prompt = make_prompt(row, candidates)
    inputs = tokenizer_qwen(prompt, return_tensors="pt").to(model_qwen.device)
    with torch.no_grad():
        output = model_qwen.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0
        )
    text = tokenizer_qwen.decode(output[0], skip_special_tokens=True)
    preds = []
    for c in candidates:
        if c in text:
            preds.append(c)
        if len(preds) == 3:
            break
    while len(preds) < 3:
        preds.append(candidates[0])
    return preds

# ============================================================
# 6️⃣ BATCH INFERENCE
# ============================================================

BATCH_SIZE = 50
OUTPUT_DIR = "./qwen_rerank_batches"
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results = []

for batch_start in range(0, len(X_val), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(X_val))
    batch_df = X_val.iloc[batch_start:batch_end].copy()
    
    batch_preds = []
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_start}-{batch_end-1}"):
        top_candidates = get_top_candidates(prepare_text(row), top_k=5)
        preds = qwen_rerank(row, top_candidates)
        batch_preds.append(preds)
    
    batch_df["preds"] = batch_preds
    batch_file = os.path.join(OUTPUT_DIR, f"batch_{batch_start}_{batch_end-1}.csv")
    batch_df.to_csv(batch_file, index=False)
    print(f"Saved batch predictions: {batch_file}")
    all_results.append(batch_df)

# ============================================================
# 7️⃣ COMBINE BATCHES
# ============================================================

results_df = pd.concat(all_results).reset_index(drop=True)

# ============================================================
# 8️⃣ EVALUATE
# ============================================================

def map_at_3(true, preds):
    scores = []
    for t, p in zip(true, preds):
        score = 0
        for i, pred in enumerate(p[:3]):
            if pred == t:
                score = 1/(i+1)
                break
        scores.append(score)
    return np.mean(scores)

map3_score = map_at_3(results_df["label"], results_df["preds"])
top1_acc = np.mean([p[0] == t for p, t in zip(results_df["preds"], results_df["label"])])

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print("MAP@3:", round(map3_score,4))
print("Top-1 Accuracy:", round(top1_acc,4))

# Save combined results
results_df.to_csv("qwen_rerank_full_val_predictions.csv", index=False)
print("Saved final combined predictions: qwen_rerank_full_val_predictions.csv")

print("\n✅ Inference complete!")
