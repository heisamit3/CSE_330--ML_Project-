# ============================================================
# DEBERTA-V3-LARGE (OPTION 1 - FULLY FIXED, T4 SAFE)
# ============================================================

import pandas as pd
import numpy as np
import torch
import gc
from scipy.special import softmax

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# 0. CLEAN MEMORY
# ------------------------------------------------------------
torch.cuda.empty_cache()
gc.collect()

print("=" * 70)
print(" DEBERTA-V3-LARGE (OPTION 1 - FULLY FIXED, T4 SAFE)")
print("=" * 70)

# ------------------------------------------------------------
# 1. GPU CHECK
# ------------------------------------------------------------
print(f"\n🔍 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ Using: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"✓ GPU Memory: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")

# ------------------------------------------------------------
# 2. LOAD & CLEAN DATA
# ------------------------------------------------------------
print("\n[1/5] Loading data...")
train = pd.read_csv("train.csv")

train["StudentExplanation"] = train["StudentExplanation"].fillna("")
train["Misconception"] = train["Misconception"].fillna("NA")

train["label"] = train["Category"] + ":" + train["Misconception"]
train = train.dropna(subset=["label", "StudentExplanation"])

# Remove rare classes
class_counts = train["label"].value_counts()
rare_classes = class_counts[class_counts < 10].index
train = train[~train["label"].isin(rare_classes)]

print(f"✓ Samples: {len(train)}")
print(f"✓ Classes: {train['label'].nunique()}")

label_encoder = LabelEncoder()
train["label_id"] = label_encoder.fit_transform(train["label"])
num_labels = len(label_encoder.classes_)

X_train, X_val = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train["label_id"]
)

print(f"✓ Train: {len(X_train)} | Val: {len(X_val)}")

# ------------------------------------------------------------
# 3. PREPARE TEXT
# ------------------------------------------------------------
print("\n[2/5] Preparing text...")

def build_text(df):
    return (
        "Question: " + df["QuestionText"].fillna("") +
        " | Answer: " + df["MC_Answer"].fillna("") +
        " | Explanation: " + df["StudentExplanation"]
    )

X_train["text"] = build_text(X_train)
X_val["text"] = build_text(X_val)

train_dataset = Dataset.from_pandas(
    X_train[["text", "label_id"]]
    .rename(columns={"label_id": "labels"})
    .reset_index(drop=True)
)

val_dataset = Dataset.from_pandas(
    X_val[["text", "label_id"]]
    .rename(columns={"label_id": "labels"})
    .reset_index(drop=True)
)

print("✓ HuggingFace datasets created")

# ------------------------------------------------------------
# 4. LOAD MODEL (FULLY FIXED)
# ------------------------------------------------------------
print("\n[3/5] Loading DeBERTa-v3-large...")

MODEL_NAME = "microsoft/deberta-v3-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# 🔥 CRITICAL FIX 🔥
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.config.use_cache = False

print("✓ Model loaded with NON-reentrant gradient checkpointing")

# ------------------------------------------------------------
# 5. TOKENIZATION
# ------------------------------------------------------------
print("\nTokenizing...")

MAX_LEN = 160

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

print("✓ Tokenization complete")

# ------------------------------------------------------------
# 6. TRAINING CONFIG (T4 SAFE)
# ------------------------------------------------------------
print("\n[4/5] Training (T4-safe config)")
print("⏱ Expected time: ~30–40 min")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    optim="adafactor",
    logging_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    dataloader_num_workers=0,
    max_grad_norm=1.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

trainer.train()
print("✓ Training complete!")

# ------------------------------------------------------------
# 7. EVALUATION (MAP@3)
# ------------------------------------------------------------
print("\n[5/5] Evaluating...")

preds = trainer.predict(val_dataset)
probs = softmax(preds.predictions, axis=1)

top3_idx = np.argsort(-probs, axis=1)[:, :3]
top3_labels = [[label_encoder.classes_[i] for i in row] for row in top3_idx]

true_labels = label_encoder.inverse_transform(
    X_val["label_id"].values
)

def map_at_3(y_true, y_pred):
    score = 0
    for t, preds in zip(y_true, y_pred):
        for i, p in enumerate(preds):
            if p == t:
                score += 1 / (i + 1)
                break
    return score / len(y_true)

map3 = map_at_3(true_labels, top3_labels)
top1 = np.mean([p[0] == t for p, t in zip(top3_labels, true_labels)])

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"MAP@3: {map3:.4f}")
print(f"Top-1 Accuracy: {top1:.4f}")
print("=" * 70)
print("\n✅ COMPLETE — STABLE ON TESLA T4")
