# ============================================================
# MAP COMPETITION - COMPLETE PIPELINE (FIXED & OPTIMIZED)
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder  # <-- ADDED FOR XGBOOST
import time

import lightgbm as lgb
import xgboost as xgb

print("="*70)
print(" MAP COMPETITION - STUDENT MATH MISCONCEPTIONS")
print("="*70)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")

# For Colab: Uncomment to upload
#from google.colab import files
#uploaded = files.upload()

train = pd.read_csv('train.csv')

# Preprocess
train['StudentExplanation'] = train['StudentExplanation'].fillna('')
train['Misconception'] = train['Misconception'].fillna('NA')
train['label'] = train['Category'] + ':' + train['Misconception']
train = train.dropna(subset=['label', 'StudentExplanation'])

print(f"✓ Total samples: {len(train)}")
print(f"✓ Unique classes: {train['label'].nunique()}")

# Remove rare classes
class_counts = train['label'].value_counts()
rare_classes = class_counts[class_counts < 2].index
if len(rare_classes) > 0:
    train = train[~train['label'].isin(rare_classes)]
    print(f"✓ Removed {len(rare_classes)} rare classes → {train['label'].nunique()} remaining")

# ============================================================
# STEP 2: SPLIT DATA
# ============================================================
print("\n[2/6] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    train['StudentExplanation'], train['label'],
    test_size=0.2, random_state=42, stratify=train['label']
)
print(f"✓ Train: {len(X_train)} | Validation: {len(X_val)}")

# Encode labels for XGBoost
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# ============================================================
# STEP 3: FEATURE EXTRACTION
# ============================================================
print("\n[3/6] Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), 
    min_df=3, max_df=0.8, strip_accents='unicode'
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print(f"✓ Feature shape: {X_train_vec.shape}")

# ============================================================
# STEP 4: TRAIN ALL MODELS (OPTIMIZED FOR SPEED)
# ============================================================
print("\n[4/6] Training models...")
print("-"*70)

results = []

# Model 1: Logistic Regression with CV
print("\n[Model 1/4] Logistic Regression")
start = time.time()
lr = LogisticRegression(max_iter=300, random_state=42, class_weight='balanced', solver='lbfgs')
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr, X_train_vec, y_train, cv=cv, scoring='accuracy')
lr.fit(X_train_vec, y_train)
lr_time = time.time() - start
lr_pred = lr.predict(X_val_vec)
lr_acc = accuracy_score(y_val, lr_pred)
lr_f1 = f1_score(y_val, lr_pred, average='weighted', zero_division=0)
print(f"  CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"  Val Accuracy: {lr_acc:.4f} | F1: {lr_f1:.4f} | Time: {lr_time:.1f}s")
results.append(['Logistic Regression', lr_time, cv_scores.mean(), lr_acc, lr_f1])

# Model 2: Random Forest
print("\n[Model 2/4] Random Forest")
start = time.time()
rf = RandomForestClassifier(
    n_estimators=50,         # Reduced
    max_depth=15,            # Reduced
    min_samples_split=10,
    random_state=42, 
    class_weight='balanced', 
    n_jobs=-1
)
rf.fit(X_train_vec, y_train)
rf_time = time.time() - start
rf_pred = rf.predict(X_val_vec)
rf_acc = accuracy_score(y_val, rf_pred)
rf_f1 = f1_score(y_val, rf_pred, average='weighted', zero_division=0)
print(f"  Val Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f} | Time: {rf_time:.1f}s")
results.append(['Random Forest', rf_time, None, rf_acc, rf_f1])

# Model 3: LightGBM (HEAVILY OPTIMIZED)
print("\n[Model 3/4] LightGBM")
start = time.time()
lgbm = lgb.LGBMClassifier(
    n_estimators=50,         # REDUCED from 200
    max_depth=6,             # REDUCED from 10
    learning_rate=0.1,       # INCREASED from 0.05
    num_leaves=31,
    random_state=42, 
    class_weight='balanced',
    verbose=-1,
    n_jobs=-1
)
lgbm.fit(X_train_vec, y_train)
lgbm_time = time.time() - start
lgbm_pred = lgbm.predict(X_val_vec)
lgbm_acc = accuracy_score(y_val, lgbm_pred)
lgbm_f1 = f1_score(y_val, lgbm_pred, average='weighted', zero_division=0)
print(f"  Val Accuracy: {lgbm_acc:.4f} | F1: {lgbm_f1:.4f} | Time: {lgbm_time:.1f}s")
results.append(['LightGBM', lgbm_time, None, lgbm_acc, lgbm_f1])

# Model 4: XGBoost (FIXED - Uses encoded labels)
print("\n[Model 4/4] XGBoost")
start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=50,         # REDUCED from 200
    max_depth=6,             # REDUCED from 10
    learning_rate=0.1,       # INCREASED from 0.05
    random_state=42, 
    verbosity=0,
    n_jobs=-1
)
xgb_model.fit(X_train_vec, y_train_encoded)  # <-- USE ENCODED LABELS
xgb_time = time.time() - start
xgb_pred_encoded = xgb_model.predict(X_val_vec)
xgb_pred = label_encoder.inverse_transform(xgb_pred_encoded)  # <-- DECODE BACK
xgb_acc = accuracy_score(y_val, xgb_pred)
xgb_f1 = f1_score(y_val, xgb_pred, average='weighted', zero_division=0)
print(f"  Val Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f} | Time: {xgb_time:.1f}s")
results.append(['XGBoost', xgb_time, None, xgb_acc, xgb_f1])

# ============================================================
# STEP 5: RESULTS COMPARISON
# ============================================================
print("\n" + "="*70)
print("[5/6] RESULTS COMPARISON")
print("="*70)

results_df = pd.DataFrame(results, columns=['Model', 'Time(s)', 'CV_Score', 'Val_Acc', 'F1_Score'])
print(results_df.to_string(index=False))
print("-"*70)

best_idx = results_df['Val_Acc'].idxmax()
best_model_info = results_df.iloc[best_idx]
print(f"\n🏆 BEST MODEL: {best_model_info['Model']}")
print(f"   Validation Accuracy: {best_model_info['Val_Acc']:.4f}")
print(f"   F1-Score: {best_model_info['F1_Score']:.4f}")

# ============================================================
# STEP 6: ERROR ANALYSIS
# ============================================================
print("\n" + "="*70)
print("[6/6] ERROR ANALYSIS")
print("="*70)

misclassified = y_val != lr_pred
errors_sample = X_val[misclassified].head(5)
true_sample = y_val[misclassified].head(5)
pred_sample = pd.Series(lr_pred)[misclassified.values].head(5)

print("\nSample Misclassifications (Logistic Regression):")
for i, (text, true_label, pred_label) in enumerate(zip(errors_sample, true_sample, pred_sample), 1):
    print(f"\n{i}. Text: {text[:70]}...")
    print(f"   True: {true_label}")
    print(f"   Pred: {pred_label}")

print("\n" + "-"*70)
print("Top 5 Most Common Classes:")
print(train['label'].value_counts().head())

print("\n" + "="*70)
print("✅ PIPELINE COMPLETE!")
print("="*70)

print(f"\nSUMMARY:")
print(f"  • Dataset: {len(train)} samples, {train['label'].nunique()} classes")
print(f"  • Best Model: {best_model_info['Model']}")
print(f"  • Validation Accuracy: {best_model_info['Val_Acc']:.4f}")
print(f"  • Random Baseline: {1/train['label'].nunique():.4f}")
print(f"  • Improvement: {best_model_info['Val_Acc']/(1/train['label'].nunique()):.1f}x better")