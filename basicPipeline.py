import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
train = pd.read_csv('train.csv')

# Handle NaN values in the data
# Fill NaN in StudentExplanation with empty string
train['StudentExplanation'] = train['StudentExplanation'].fillna('')

# Fill NaN in Misconception with 'NA'
train['Misconception'] = train['Misconception'].fillna('NA')

# Create label
train['label'] = train['Category'] + ':' + train['Misconception']

# Remove any rows where label is still NaN (just in case)
train = train.dropna(subset=['label', 'StudentExplanation'])

# Print dataset info
print(f"Total samples: {len(train)}")
print(f"Unique labels: {train['label'].nunique()}")
print(f"\nFirst few labels:")
print(train['label'].value_counts().head())

# Simple train/val split
X_train, X_val, y_train, y_val = train_test_split(
    train['StudentExplanation'], 
    train['label'],
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Feature extraction
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

print(f"\nFeature matrix shape: {X_train_vec.shape}")

# Train model
print("\nTraining model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Predictions
print("\nMaking predictions...")
y_pred = model.predict(X_val_vec)

# Evaluation
print(f"\nAccuracy: {model.score(X_val_vec, y_val):.3f}")

# Show some predictions
print("\nSample predictions:")
for i in range(5):
    print(f"True: {y_val.iloc[i]}")
    print(f"Pred: {y_pred[i]}")
    print(f"Text: {X_val.iloc[i][:100]}...")
    print("-" * 50)