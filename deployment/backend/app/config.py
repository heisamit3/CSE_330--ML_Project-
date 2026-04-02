"""
Configuration module for the MAP Misconception Detection API.
All settings are loaded from environment variables.
"""

import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Model configuration
BASE_MODEL_PATH: str = os.getenv("BASE_MODEL_PATH", "Qwen/Qwen3-8B")
ADAPTER_PATH: str = os.getenv("ADAPTER_PATH", str(DATA_DIR / "adapter"))
USE_4BIT_QUANTIZATION: bool = os.getenv("USE_4BIT_QUANTIZATION", "true").lower() == "true"
TORCH_DTYPE: str = os.getenv("TORCH_DTYPE", "float16")

# Inference configuration
MAX_LEN: int = int(os.getenv("MAX_LEN", "256"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1"))

# Explainer configuration
USE_LLM_EXPLAINER: bool = os.getenv("USE_LLM_EXPLAINER", "true").lower() == "true"
USE_LLM_VERIFIER: bool = os.getenv("USE_LLM_VERIFIER", "false").lower() == "true"
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "320"))

# Data paths
TRAIN_DATA_PATH: str = os.getenv("TRAIN_DATA_PATH", str(DATA_DIR / "train.csv"))
KB_PATH: str = os.getenv("KB_PATH", str(DATA_DIR / "misconception_kb.json"))
LABEL_ENCODER_PATH: str = os.getenv("LABEL_ENCODER_PATH", str(DATA_DIR / "label_encoder.pkl"))
CORRECT_ANSWERS_PATH: str = os.getenv("CORRECT_ANSWERS_PATH", str(DATA_DIR / "correct_answers.json"))

# Server configuration
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# HuggingFace
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)


def get_device() -> str:
    """Get the appropriate device for inference."""
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"
