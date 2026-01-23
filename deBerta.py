"""
============================================================
DeBERTa-v3 Student Misconception Classification
============================================================
Kaggle-ready training script for classifying student misconceptions
Optimized for Kaggle GPU (Tesla T4/P100)

To run on Kaggle:
1. Create a new notebook
2. Add your dataset (train.csv, test.csv)
3. Enable GPU accelerator (Settings > Accelerator > GPU T4 x2 or P100)
4. Run all cells
"""

# ============================================================
# SECTION 1: INSTALL & IMPORT DEPENDENCIES
# ============================================================

# Uncomment these lines when running on Kaggle for the first time
# !pip install -q transformers datasets accelerate sentencepiece

import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import pickle

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================

class Config:
    """Central configuration for all hyperparameters"""
    
    # Paths - Update these for Kaggle
    TRAIN_PATH = "/kaggle/input/dataset2/train.csv"  # Update with your dataset path
    OUTPUT_DIR = "/kaggle/working/deberta_output"
    
    # For local testing, use these paths:
    #TRAIN_PATH = "train.csv"
    #OUTPUT_DIR = "./deberta_output"
    
    # Model
    MODEL_NAME = "microsoft/deberta-v3-base"  # Options: deberta-v3-base, deberta-v3-large
    MAX_LENGTH = 256
    
    # Training
    EPOCHS = 3
    BATCH_SIZE = 8          # Reduce to 4 if OOM
    EVAL_BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 2  # Effective batch = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    
    # Regularization
    DROPOUT = 0.1
    LABEL_SMOOTHING = 0.1
    MAX_GRAD_NORM = 1.0
    
    # Cross-validation
    N_FOLDS = 5
    USE_CV = False  # Set to True for full CV training
    
    # Misc
    SEED = 42
    FP16 = True
    NUM_WORKERS = 2
    MIN_CLASS_SAMPLES = 5  # Minimum samples per class
    
    # Debug mode - set True for quick testing
    DEBUG = False          # DISABLED for full training
    DEBUG_SAMPLES = 1000


def seed_everything(seed):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_gpu_info():
    """Print GPU information"""
    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: True")
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
        
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"✓ GPU Memory: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")
        except:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            print(f"✓ GPU Memory: {total_mem/1e9:.2f} GB total")
    else:
        print("⚠ No GPU available - training will be slow!")
    
    print("=" * 60)


# ============================================================
# SECTION 3: DATA LOADING & PREPROCESSING
# ============================================================

def load_and_preprocess_data(config):
    """Load and preprocess the training data"""
    
    print("\n[STEP 1/6] Loading Data...")
    print("-" * 40)
    
    # Load data
    train_df = pd.read_csv(config.TRAIN_PATH)
    
    print(f"Original dataset size: {len(train_df)}")
    
    # Fill missing values
    train_df["StudentExplanation"] = train_df["StudentExplanation"].fillna("")
    train_df["QuestionText"] = train_df["QuestionText"].fillna("")
    train_df["MC_Answer"] = train_df["MC_Answer"].fillna("")
    train_df["Misconception"] = train_df["Misconception"].fillna("NA")
    train_df["Category"] = train_df["Category"].fillna("Unknown")
    
    # Create combined label
    train_df["label"] = train_df["Category"] + " | " + train_df["Misconception"]
    
    # Remove rows with missing labels
    train_df = train_df.dropna(subset=["label", "StudentExplanation"])
    
    # Filter rare classes
    class_counts = train_df["label"].value_counts()
    rare_classes = class_counts[class_counts < config.MIN_CLASS_SAMPLES].index
    train_df = train_df[~train_df["label"].isin(rare_classes)]
    
    print(f"After filtering rare classes: {len(train_df)}")
    print(f"Number of classes: {train_df['label'].nunique()}")
    
    # Debug mode - use subset
    if config.DEBUG:
        print("\n⚡ DEBUG MODE ENABLED - Using small subset for quick testing")
        # Take top N most common classes to ensure enough samples
        top_classes = class_counts.head(20).index.tolist()
        train_df = train_df[train_df['label'].isin(top_classes)]
        # Sample from each class
        train_df = train_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), 10), random_state=config.SEED)
        ).reset_index(drop=True)
        print(f"DEBUG MODE: Using {len(train_df)} samples across {train_df['label'].nunique()} classes")
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_df["label_id"] = label_encoder.fit_transform(train_df["label"])
    num_labels = len(label_encoder.classes_)
    
    print(f"✓ Final samples: {len(train_df)}")
    print(f"✓ Number of classes: {num_labels}")
    
    return train_df, label_encoder, num_labels


def create_input_text(row):
    """Create formatted input text from dataframe row"""
    text = f"Question: {row['QuestionText']}\n"
    text += f"Student Answer: {row['MC_Answer']}\n"
    text += f"Student Explanation: {row['StudentExplanation']}"
    return text


def prepare_datasets(train_df, tokenizer, config):
    """Prepare HuggingFace datasets for training"""
    
    print("\n[STEP 2/6] Preparing Datasets...")
    print("-" * 40)
    
    # Create text column
    train_df = train_df.copy()
    train_df["text"] = train_df.apply(create_input_text, axis=1)
    
    # Split data
    X_train, X_val = train_test_split(
        train_df,
        test_size=0.15,
        random_state=config.SEED,
        stratify=train_df["label_id"]
    )
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")
    
    # Create HuggingFace datasets
    train_dataset = HFDataset.from_pandas(
        X_train[["text", "label_id"]].rename(columns={"label_id": "labels"}).reset_index(drop=True)
    )
    
    val_dataset = HFDataset.from_pandas(
        X_val[["text", "label_id"]].rename(columns={"label_id": "labels"}).reset_index(drop=True)
    )
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding=False  # Dynamic padding handled by DataCollator
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train"
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation"
    )
    
    print("✓ Tokenization complete")
    
    return train_dataset, val_dataset, X_train, X_val


# ============================================================
# SECTION 4: MODEL SETUP
# ============================================================

def setup_model_and_tokenizer(config, num_labels):
    """Initialize model and tokenizer"""
    
    print("\n[STEP 3/6] Loading Model & Tokenizer...")
    print("-" * 40)
    print(f"Model: {config.MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Load model config
    model_config = AutoConfig.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels,
        hidden_dropout_prob=config.DROPOUT,
        attention_probs_dropout_prob=config.DROPOUT,
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        config=model_config,
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print("✓ Gradient checkpointing enabled")
    
    return model, tokenizer


# ============================================================
# SECTION 5: TRAINING
# ============================================================

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }


def train_model(model, tokenizer, train_dataset, val_dataset, config):
    """Train the model using HuggingFace Trainer"""
    
    print("\n[STEP 4/6] Training Model...")
    print("-" * 40)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        
        # Training settings
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        
        # Optimizer settings
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM,
        optim="adamw_torch",  # or "adafactor" for memory efficiency
        
        # Regularization
        label_smoothing_factor=config.LABEL_SMOOTHING,
        
        # Evaluation & Saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        
        # Precision & Performance
        fp16=config.FP16,
        dataloader_num_workers=config.NUM_WORKERS,
        
        # Logging
        logging_dir=f"{config.OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="none",  # Set to "wandb" if using Weights & Biases
        
        # Misc
        seed=config.SEED,
        remove_unused_columns=True,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=2)
    ]
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    # Train
    print(f"Starting training for {config.EPOCHS} epochs...")
    print(f"Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION}")
    
    train_result = trainer.train()
    
    print("\n✓ Training complete!")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    
    return trainer


# ============================================================
# SECTION 6: EVALUATION & PREDICTION
# ============================================================

def evaluate_model(trainer, val_dataset, X_val, label_encoder):
    """Evaluate model and compute MAP@3"""
    
    print("\n[STEP 5/6] Evaluating Model...")
    print("-" * 40)
    
    # Get predictions
    predictions = trainer.predict(val_dataset)
    probs = softmax(predictions.predictions, axis=1)
    
    # Top-3 predictions
    top3_idx = np.argsort(-probs, axis=1)[:, :3]
    top3_labels = [[label_encoder.classes_[i] for i in row] for row in top3_idx]
    
    # True labels
    true_labels = label_encoder.inverse_transform(X_val["label_id"].values)
    
    # Compute MAP@3
    def mean_average_precision_at_k(y_true, y_pred, k=3):
        """Calculate Mean Average Precision @ K"""
        scores = []
        for true, preds in zip(y_true, y_pred):
            for i, pred in enumerate(preds[:k]):
                if pred == true:
                    scores.append(1.0 / (i + 1))
                    break
            else:
                scores.append(0.0)
        return np.mean(scores)
    
    map_at_3 = mean_average_precision_at_k(true_labels, top3_labels, k=3)
    top1_accuracy = np.mean([preds[0] == true for preds, true in zip(top3_labels, true_labels)])
    top3_accuracy = np.mean([true in preds[:3] for preds, true in zip(top3_labels, true_labels)])
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"MAP@3:          {map_at_3:.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
    print("=" * 50)
    
    return {
        "map_at_3": map_at_3,
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "predictions": probs,
        "top3_labels": top3_labels
    }


def save_validation_predictions(results, X_val, label_encoder, config):
    """Save validation predictions for analysis"""
    
    print("\n[STEP 6/6] Saving Validation Results...")
    print("-" * 40)
    
    # Create validation results DataFrame
    val_results = pd.DataFrame({
        "row_id": X_val["row_id"].values,
        "true_label": label_encoder.inverse_transform(X_val["label_id"].values),
        "prediction_1": [preds[0] for preds in results["top3_labels"]],
        "prediction_2": [preds[1] for preds in results["top3_labels"]],
        "prediction_3": [preds[2] for preds in results["top3_labels"]],
    })
    
    # Add correctness flags
    val_results["correct_top1"] = val_results["true_label"] == val_results["prediction_1"]
    val_results["correct_top3"] = val_results.apply(
        lambda x: x["true_label"] in [x["prediction_1"], x["prediction_2"], x["prediction_3"]], 
        axis=1
    )
    
    # Save
    results_path = f"{config.OUTPUT_DIR}/validation_predictions.csv"
    val_results.to_csv(results_path, index=False)
    print(f"✓ Validation predictions saved to: {results_path}")
    
    # Print some error analysis
    print("\n" + "=" * 50)
    print("ERROR ANALYSIS")
    print("=" * 50)
    
    incorrect = val_results[~val_results["correct_top1"]]
    print(f"Total incorrect (Top-1): {len(incorrect)}")
    
    if len(incorrect) > 0:
        print("\nSample misclassifications:")
        for _, row in incorrect.head(5).iterrows():
            print(f"  True: {row['true_label'][:50]}...")
            print(f"  Pred: {row['prediction_1'][:50]}...")
            print()
    
    return val_results


def save_artifacts(trainer, label_encoder, config):
    """Save model and artifacts"""
    
    print("\nSaving model artifacts...")
    
    # Save model
    model_path = f"{config.OUTPUT_DIR}/final_model"
    trainer.save_model(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save label encoder
    encoder_path = f"{config.OUTPUT_DIR}/label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Label encoder saved to: {encoder_path}")
    
    # Save config
    config_path = f"{config.OUTPUT_DIR}/config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump({
            'model_name': config.MODEL_NAME,
            'max_length': config.MAX_LENGTH,
            'num_labels': len(label_encoder.classes_)
        }, f)
    print(f"✓ Config saved to: {config_path}")


# ============================================================
# SECTION 7: MAIN EXECUTION
# ============================================================

def main():
    """Main execution function"""
    
    print("\n" + "=" * 60)
    print("DeBERTa Student Misconception Classification")
    print("=" * 60)
    
    # Initialize
    config = Config()
    seed_everything(config.SEED)
    print_gpu_info()
    
    # Clean GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load data
    train_df, label_encoder, num_labels = load_and_preprocess_data(config)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config, num_labels)
    
    # Prepare datasets
    train_dataset, val_dataset, X_train, X_val = prepare_datasets(
        train_df, tokenizer, config
    )
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)
    
    # Evaluate
    results = evaluate_model(trainer, val_dataset, X_val, label_encoder)
    
    # Save validation predictions for analysis
    val_results = save_validation_predictions(results, X_val, label_encoder, config)
    
    # Save artifacts
    save_artifacts(trainer, label_encoder, config)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    
    return trainer, results, label_encoder, val_results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    trainer, results, label_encoder, val_results = main()
