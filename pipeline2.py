import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import time

print("="*60)
print("MAP Competition - Pipeline with Cross-Validation")
print("="*60)

# Load data
print("\n[1/7] Loading data...")
train = pd.read_csv('train.csv')

train['StudentExplanation'] = train['StudentExplanation'].fillna('')
train['Misconception'] = train['Misconception'].fillna('NA')
train['label'] = train['Category'] + ':' + train['Misconception']
train = train.dropna(subset=['label', 'StudentExplanation'])

print(f"✓ Loaded {len(train)} samples")
print(f"✓ {train['label'].nunique()} unique classes")

# Remove rare classes
print("\n[2/7] Filtering rare classes...")
class_counts = train['label'].value_counts()
rare_classes = class_counts[class_counts < 2].index

if len(rare_classes) > 0:
    train = train[~train['label'].isin(rare_classes)]
    print(f"✓ Removed {len(rare_classes)} rare classes")
    print(f"✓ {train['label'].nunique()} classes remaining")

# Split data
print("\n[3/7] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    train['StudentExplanation'], 
    train['label'],
    test_size=0.2,
    random_state=42,
    stratify=train['label']
)
print(f"✓ Train: {len(X_train)}, Validation: {len(X_val)}")

# Feature extraction
print("\n[4/7] Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8,
    strip_accents='unicode'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print(f"✓ Features: {X_train_vec.shape}")

# Model 1: Logistic Regression with Cross-Validation
print("\n[5/7] Training Logistic Regression...")
print("    (Takes ~2-3 minutes with CV...)")
start = time.time()

lr_model = LogisticRegression(
    max_iter=500,
    random_state=42,
    class_weight='balanced',
    solver='lbfgs',
    verbose=0
)

# Perform 5-fold cross-validation on training data
print("    Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    lr_model, 
    X_train_vec, 
    y_train, 
    cv=cv, 
    scoring='accuracy'
)

print(f"    CV Fold Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"    Mean CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Train on full training set
print("    Training on full training set...")
lr_model.fit(X_train_vec, y_train)
lr_time = time.time() - start

# Evaluate on validation set
lr_pred = lr_model.predict(X_val_vec)
lr_acc = accuracy_score(y_val, lr_pred)
lr_f1 = f1_score(y_val, lr_pred, average='weighted', zero_division=0)

print(f"✓ Total Time: {lr_time:.1f}s")
print(f"✓ Validation Accuracy: {lr_acc:.4f}")
print(f"✓ Validation F1-Score: {lr_f1:.4f}")

# Model 2: Random Forest
print("\n[6/7] Training Random Forest...")
print("    (Takes ~30-60 seconds...)")
start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced',
    verbose=0
)

rf_model.fit(X_train_vec, y_train)
rf_time = time.time() - start

rf_pred = rf_model.predict(X_val_vec)
rf_acc = accuracy_score(y_val, rf_pred)
rf_f1 = f1_score(y_val, rf_pred, average='weighted', zero_division=0)

print(f"✓ Time: {rf_time:.1f}s")
print(f"✓ Validation Accuracy: {rf_acc:.4f}")
print(f"✓ Validation F1-Score: {rf_f1:.4f}")

# Comparison
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"{'Model':<20} {'Time':>10} {'CV Score':>12} {'Val Acc':>12} {'F1':>10}")
print("-"*60)
print(f"{'Logistic Regression':<20} {lr_time:>8.1f}s {cv_scores.mean():>11.4f} {lr_acc:>11.4f} {lr_f1:>9.4f}")
print(f"{'Random Forest':<20} {rf_time:>8.1f}s {'N/A':>11} {rf_acc:>11.4f} {rf_f1:>9.4f}")
print("-"*60)

# Select best
if lr_acc >= rf_acc:
    best_model = lr_model
    best_name = "Logistic Regression"
    best_acc = lr_acc
    best_cv = cv_scores.mean()
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_acc = rf_acc
    best_cv = None

print(f"\n🏆 Best: {best_name} (Val Acc: {best_acc:.4f})")

# Generate submission
print("\n[7/7] Generating submission...")
print("="*60)

test = pd.read_csv('test.csv')
test['StudentExplanation'] = test['StudentExplanation'].fillna('')

print(f"✓ Test samples: {len(test)}")

X_test_vec = vectorizer.transform(test['StudentExplanation'])
pred_proba = best_model.predict_proba(X_test_vec)

# Get top 3 predictions
top3_idx = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]

submissions = []
for indices in top3_idx:
    top3_labels = [best_model.classes_[i] for i in indices]
    submissions.append(' '.join(top3_labels))

sub_df = pd.DataFrame({'Category:Misconception': submissions})
sub_df.to_csv('submission.csv', index=False)

print(f"✓ Saved: submission.csv")
print(f"✓ Predictions: {len(sub_df)}")

# Sample predictions
print("\nSample predictions:")
for i in range(min(5, len(submissions))):
    print(f"  {i+1}. {submissions[i][:70]}...")

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Dataset:")
print(f"  • Total samples: {len(train)}")
print(f"  • Number of classes: {train['label'].nunique()}")
print(f"  • Train/Val split: {len(X_train)}/{len(X_val)}")
print(f"\nBest Model: {best_name}")
if best_cv:
    print(f"  • Cross-validation: {best_cv:.4f} (±{cv_scores.std():.4f})")
print(f"  • Validation Accuracy: {best_acc:.4f}")
print(f"  • Submission: {len(sub_df)} predictions")
print("="*60)
print("✅ COMPLETE!")
print("="*60)