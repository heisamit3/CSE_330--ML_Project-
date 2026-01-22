# ============================================================
# ULTRA-FAST GPU-OPTIMIZED VERSION (WITH EXPLICIT CHECKS)
# ============================================================

import pandas as pd
import numpy as np
from scipy.special import softmax
import torch
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

print("="*70)
print(" GPU-OPTIMIZED TRANSFORMER TRAINING")
print("="*70)

# ============================================================
# CRITICAL: CHECK GPU AVAILABILITY
# ============================================================
print("\n🔍 CHECKING HARDWARE...")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU FOUND: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    USE_GPU = True
else:
    device = torch.device("cpu")
    print("❌ NO GPU DETECTED - RUNNING ON CPU")
    print("⚠️  THIS WILL BE EXTREMELY SLOW (hours instead of minutes)")
    print("\n🔧 TO FIX:")
    print("   1. Runtime → Change runtime type")
    print("   2. Hardware accelerator → T4 GPU")
    print("   3. Save → Runtime will restart")
    print("   4. Re-run this code")
    print("\n⏸️  Do you want to continue on CPU anyway? (Not recommended)")
    
    import sys
    response = input("Type 'yes' to continue on CPU, or 'no' to stop: ").lower()
    if response != 'yes':
        print("❌ Stopping. Please enable GPU and try again.")
        sys.exit()
    
    USE_GPU = False
    print("\n⚠️  Continuing on CPU (this will be slow)...")

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")
train = pd.read_csv('train.csv')
train['StudentExplanation'] = train['StudentExplanation'].fillna('')
train['Misconception'] = train['Misconception'].fillna('NA')
train['label'] = train['Category'] + ':' + train['Misconception']
train = train.dropna(subset=['label', 'StudentExplanation'])

# Remove rare classes
class_counts = train['label'].value_counts()
rare_classes = class_counts[class_counts < 10].index
train = train[~train['label'].isin(rare_classes)]

print(f"✓ Samples: {len(train)}, Classes: {train['label'].nunique()}")

# Encode labels
label_encoder = LabelEncoder()
train['label_id'] = label_encoder.fit_transform(train['label'])
num_labels = len(label_encoder.classes_)

# Split data
X_train, X_val = train_test_split(
    train, test_size=0.2, random_state=42, stratify=train['label_id']
)

# Adjust sample size based on hardware
if USE_GPU:
    SAMPLE_FRAC = 1.0  # 30% on GPU = fast
    print("  Using 30% of data (GPU mode)")
else:
    SAMPLE_FRAC = 0.05  # Only 5% on CPU
    print("  Using only 5% of data (CPU mode - very limited)")

X_train = X_train.sample(frac=SAMPLE_FRAC, random_state=42)
print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}")

# ============================================================
# PREPARE TEXT
# ============================================================
print("\n[2/5] Preparing text...")
X_train['text'] = (
    X_train['QuestionText'].fillna('') + " [SEP] " + 
    X_train['StudentExplanation']
)
X_val['text'] = (
    X_val['QuestionText'].fillna('') + " [SEP] " + 
    X_val['StudentExplanation']
)

train_dataset = Dataset.from_pandas(
    X_train[['text', 'label_id']].rename(columns={'label_id': 'labels'}).reset_index(drop=True)
)
val_dataset = Dataset.from_pandas(
    X_val[['text', 'label_id']].rename(columns={'label_id': 'labels'}).reset_index(drop=True)
)

# ============================================================
# LOAD MODEL
# ============================================================
print("\n[3/5] Loading model...")
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# EXPLICITLY move model to device
model = model.to(device)
print(f"✓ Model loaded on: {device}")

# Tokenize
def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, max_length=96)

print("  Tokenizing...")
train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text'])
val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=['text'])
print("✓ Done")

# ============================================================
# TRAINING CONFIG (OPTIMIZED FOR HARDWARE)
# ============================================================
print("\n[4/5] Training...")

if USE_GPU:
    # GPU settings - FAST
    batch_size = 128
    eval_batch_size = 256
    epochs = 1
    print(f"  GPU mode: batch_size={batch_size}, {epochs} epoch")
    print(f"  Expected: 3-5 minutes")
else:
    # CPU settings - LIMITED
    batch_size = 8
    eval_batch_size = 16
    epochs = 1
    print(f"  CPU mode: batch_size={batch_size}, {epochs} epoch")
    print(f"  Expected: 30-60 minutes (still slow!)")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    fp16=USE_GPU,  # Only use fp16 on GPU
    logging_steps=10,
    report_to="none",
    dataloader_num_workers=2 if USE_GPU else 0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# Train
print("  Starting training...")
trainer.train()
print("✓ Training complete!")

# ============================================================
# INFERENCE
# ============================================================
print("\n[5/5] Inference...")
predictions = trainer.predict(val_dataset)
probs = softmax(predictions.predictions, axis=1)

# Top-3
top3_idx = np.argsort(-probs, axis=1)[:, :3]
top3_classes = [[label_encoder.classes_[i] for i in row] for row in top3_idx]

# Evaluate
true_labels = label_encoder.inverse_transform(X_val['label_id'].values)

def map_at_3(true, preds):
    scores = []
    for t, p in zip(true, preds):
        for i, pred in enumerate(p):
            if pred == t:
                scores.append(1/(i+1))
                break
        else:
            scores.append(0)
    return np.mean(scores)

map3 = map_at_3(true_labels, top3_classes)
top1 = np.mean([p[0] == t for p, t in zip(top3_classes, true_labels)])

# ============================================================
# RESULTS
# ============================================================
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Hardware: {'GPU ✅' if USE_GPU else 'CPU ⚠️'}")
print(f"Training samples: {len(X_train)} ({SAMPLE_FRAC*100:.0f}% of data)")
print(f"Validation samples: {len(X_val)}")
print(f"\nMAP@3: {map3:.4f}")
print(f"Top-1 Accuracy: {top1:.4f}")
print("="*70)

if not USE_GPU:
    print("\n⚠️  Results are limited because CPU was used.")
    print("   Enable GPU for much better performance!")
else:
    print("\n✅ Done!")
    if SAMPLE_FRAC < 1.0:
        print(f"\n💡 To improve accuracy:")
        print(f"   Change SAMPLE_FRAC from {SAMPLE_FRAC} to 0.5 or 1.0")

# Save results
results_df = pd.DataFrame({
    'true_label': true_labels,
    'pred_1': [p[0] for p in top3_classes],
    'pred_2': [p[1] for p in top3_classes],
    'pred_3': [p[2] for p in top3_classes],
})

for i in range(num_labels):
    results_df[f'prob_class_{i}'] = probs[:, i]

results_df.to_csv('transformer_predictions.csv', index=False)
print("\n✓ Saved: transformer_predictions.csv")