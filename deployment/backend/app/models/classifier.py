"""
Qwen3-4B Classification model with LoRA adapter.
Extracted from q3-4b-exp.ipynb notebook.
"""

import os
import gc
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from scipy.special import softmax

from ..config import (
    BASE_MODEL_PATH,
    ADAPTER_PATH,
    USE_4BIT_QUANTIZATION,
    TORCH_DTYPE,
    MAX_LEN,
    LABEL_ENCODER_PATH,
    HF_TOKEN,
    get_device,
)

logger = logging.getLogger(__name__)


class MisconceptionClassifier:
    """
    Qwen3-4B based misconception classifier with LoRA adapter.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.n_classes = 0
        self.device = get_device()
        self._loaded = False

    def load(self) -> bool:
        """
        Load the model, tokenizer, and label encoder.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            return True

        try:
            logger.info("Loading Qwen3-4B classifier...")

            # Import here to avoid issues if torch not installed
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from peft import PeftModel

            # Load label encoder
            self._load_label_encoder()

            # Load tokenizer
            logger.info(f"Loading tokenizer from {BASE_MODEL_PATH}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_PATH,
                trust_remote_code=True,
                token=HF_TOKEN,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine torch dtype
            dtype = torch.float16 if TORCH_DTYPE == "float16" else torch.float32

            # Load base model
            logger.info(f"Loading base model from {BASE_MODEL_PATH}...")
            load_kwargs = {
                "num_labels": self.n_classes,
                "trust_remote_code": True,
                "torch_dtype": dtype,
                "token": HF_TOKEN,
            }

            # Try 4-bit quantization if enabled and on GPU
            if USE_4BIT_QUANTIZATION and "cuda" in self.device:
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,
                    )
                    load_kwargs["quantization_config"] = bnb_config
                    load_kwargs["device_map"] = "auto"
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without quantization")
                    load_kwargs["device_map"] = self.device
            else:
                load_kwargs["device_map"] = self.device

            base_model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL_PATH,
                **load_kwargs
            )
            base_model.config.pad_token_id = self.tokenizer.pad_token_id

            # Load LoRA adapter if path exists
            adapter_path = Path(ADAPTER_PATH)
            if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
                logger.info(f"Loading LoRA adapter from {ADAPTER_PATH}...")
                self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            else:
                logger.warning(f"No adapter found at {ADAPTER_PATH}, using base model only")
                self.model = base_model

            self.model.eval()
            self._loaded = True
            logger.info("Classifier loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return False

    def _load_label_encoder(self):
        """Load or create the label encoder."""
        import joblib

        encoder_path = Path(LABEL_ENCODER_PATH)
        if encoder_path.exists():
            logger.info(f"Loading label encoder from {encoder_path}")
            self.label_encoder = joblib.load(encoder_path)
            self.n_classes = len(self.label_encoder.classes_)
        else:
            logger.warning("Label encoder not found, creating default...")
            # Create default label encoder from known labels
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            # Default classes (should be replaced with actual trained labels)
            default_classes = self._get_default_classes()
            self.label_encoder.fit(default_classes)
            self.n_classes = len(self.label_encoder.classes_)

            # Save for future use
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.label_encoder, encoder_path)

    def _get_default_classes(self) -> List[str]:
        """Get default class labels. Replace with actual from training."""
        # These should match the 65 classes from training
        categories = ["True_Correct", "True_Neither", "True_Misconception",
                     "False_Correct", "False_Neither", "False_Misconception"]
        misconceptions = [
            "NA", "Incomplete", "Additive", "Subtraction", "Inversion",
            "Irrelevant", "Mult", "Wrong_fraction", "Wrong_Fraction",
            "Denominator-only_change", "Num-only_change", "Adding_across",
            "Whole_numbers_larger", "Shorter_is_bigger", "Tacking",
            "Duplication", "Positive", "Negative", "Wrong_term", "Firstterm",
            "WNB", "SwapDividend", "Scale", "Not_variable", "Adding_terms",
            "CoeffIsExp", "Multiplying_by_4", "FlipChange",
            "Expand_single_bracket", "Unknowable"
        ]
        classes = []
        for cat in ["True_Correct", "True_Neither"]:
            classes.append(f"{cat}:NA")
        for cat in ["True_Misconception"]:
            for m in misconceptions[:10]:
                classes.append(f"{cat}:{m}")
        for cat in ["False_Correct", "False_Neither"]:
            classes.append(f"{cat}:NA")
        for cat in ["False_Misconception"]:
            for m in misconceptions:
                classes.append(f"{cat}:{m}")

        # Ensure we have at least 65 unique classes
        while len(set(classes)) < 65:
            classes.append(f"False_Misconception:Unknown_{len(classes)}")

        return list(set(classes))[:65]

    def predict(self, text: str) -> Tuple[List[str], np.ndarray]:
        """
        Run prediction on input text.

        Args:
            text: Formatted input text

        Returns:
            Tuple of (top labels sorted by probability, probabilities array)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.float().cpu().numpy()

        # Get probabilities and sorted indices
        probs = softmax(logits, axis=1)[0]
        sorted_indices = np.argsort(-probs)

        # Decode labels
        sorted_labels = self.label_encoder.inverse_transform(sorted_indices)

        return sorted_labels.tolist(), probs

    def predict_batch(self, texts: List[str]) -> List[Tuple[List[str], np.ndarray]]:
        """Run batch prediction."""
        return [self.predict(text) for text in texts]

    def get_labels(self) -> List[str]:
        """Get all label classes."""
        if self.label_encoder is None:
            return []
        return self.label_encoder.classes_.tolist()

    def cleanup(self):
        """Clean up model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("Classifier cleaned up")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton instance
_classifier: Optional[MisconceptionClassifier] = None


def get_classifier() -> MisconceptionClassifier:
    """Get the singleton classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = MisconceptionClassifier()
    return _classifier
