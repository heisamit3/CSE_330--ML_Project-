# DeepSeek Model Deployment Guide for Student App

## The Problem

The `deepseek_infer.ipynb` notebook uses a **7B parameter DeepSeek model** which requires:
- **~14-16 GB GPU VRAM** (for float16 inference)
- **~28+ GB RAM** (if running on CPU, extremely slow)

Your laptop likely doesn't have this capacity, so you need alternative approaches.

---

## Option 1: Kaggle API (Recommended for Demo/Testing)

Run the model on Kaggle and call it from your student_app.

### How it works:
1. Create a Kaggle notebook with the model
2. Use Kaggle's free T4 GPU (16GB VRAM)
3. Expose predictions via a simple API

### Steps:

**A. Create Kaggle Inference Notebook**
```python
# On Kaggle - save as "deepseek-inference-api"
# Enable GPU accelerator in notebook settings

import os
import json
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model once at startup
MODEL_PATH = "/kaggle/input/your-model-dataset"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
model.eval()

def predict(question_text, answer, explanation, is_correct):
    text = f"Question: {question_text}\nAnswer: {answer}\n{'Correct' if is_correct else 'Incorrect'}\nExplanation: {explanation}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    top3_idx = probs.argsort(descending=True)[0][:3]
    # Return top 3 predictions
    return [le.inverse_transform([i.item()])[0] for i in top3_idx]
```

**B. Use Kaggle's "Inference" feature** (if available) or create a scheduled notebook that processes a queue.

### Limitation:
- Kaggle doesn't support real-time APIs directly
- Workaround: Use file-based queue (Google Drive, Firebase, etc.)

---

## Option 2: Google Colab API (Free, Temporary)

### Steps:

1. **Colab Notebook** (`colab_inference_api.ipynb`):
```python
!pip install flask flask-ngrok transformers torch

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)
run_with_ngrok(app)

# Load model
MODEL_PATH = "/content/drive/MyDrive/deepseek_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to("cuda")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = f"Question: {data['question']}\nAnswer: {data['answer']}\nExplanation: {data['explanation']}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    top3 = probs.argsort()[-3:][::-1].tolist()
    return jsonify({"predictions": top3, "probabilities": probs[top3].tolist()})

app.run()
```

2. **In student_app**, call the ngrok URL:
```python
import requests

COLAB_API = "https://xxxx-xx-xxx.ngrok.io/predict"

def get_model_prediction(question, answer, explanation):
    response = requests.post(COLAB_API, json={
        "question": question,
        "answer": answer,
        "explanation": explanation
    })
    return response.json()
```

### Limitations:
- Colab sessions timeout after ~12 hours
- ngrok URLs change each session
- Not suitable for production

---

## Option 3: Hugging Face Inference API (Easiest)

Upload your model to Hugging Face and use their free Inference API.

### Steps:

1. **Upload Model to Hugging Face**:
```bash
pip install huggingface_hub
huggingface-cli login

# Upload your trained model
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='./saved_model/deepseek_model',
    repo_id='your-username/map-deepseek-misconception',
    repo_type='model'
)
"
```

2. **Call from student_app**:
```python
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/your-username/map-deepseek-misconception"
HF_TOKEN = "hf_xxxxx"  # Your HF token

def get_prediction(text):
    response = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text}
    )
    return response.json()
```

### Limitations:
- Free tier has rate limits
- Model must be public or you need Pro subscription
- Cold start can take 20-60 seconds

---

## Option 4: Modal.com / Replicate (Pay-per-use)

Best for production with minimal setup.

### Modal.com Example:
```python
# modal_inference.py
import modal

stub = modal.Stub("deepseek-inference")

@stub.function(gpu="T4", image=modal.Image.debian_slim().pip_install("torch", "transformers"))
def predict(text: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model = AutoModelForSequenceClassification.from_pretrained("your-model")
    tokenizer = AutoTokenizer.from_pretrained("your-model")

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(dim=-1).item()
```

### Pricing:
- Modal: ~$0.000225/second for T4 GPU
- Replicate: ~$0.00025/second for T4 GPU

---

## Option 5: Local with Quantization (If you have 8GB+ VRAM)

If you have an RTX 3060 or better:

```python
# Quantized inference (4-bit)
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
```

This reduces VRAM requirement to ~4-6GB but may slightly reduce accuracy.

---

## Recommended Architecture for Your Project

```
┌─────────────────┐         ┌─────────────────┐
│   Student App   │ ──────► │  FastAPI Proxy  │
│   (Frontend)    │   HTTP  │   (Your Laptop) │
└─────────────────┘         └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  Model Backend  │
                            │ (Colab/Kaggle/  │
                            │  HuggingFace)   │
                            └─────────────────┘
```

### Implementation in student_app:

Add to `app.py`:
```python
import requests
from typing import Optional

# Configuration
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "heuristic")  # "heuristic", "huggingface", "colab"
HF_API_URL = os.getenv("HF_API_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
COLAB_API_URL = os.getenv("COLAB_API_URL", "")

async def get_model_prediction(question: str, answer: str, explanation: str, is_correct: bool) -> Optional[dict]:
    """Get prediction from ML model backend."""

    if MODEL_BACKEND == "heuristic":
        return None  # Use existing heuristic

    text = f"Question: {question}\nAnswer: {answer}\n{'Correct' if is_correct else 'Incorrect'}\nExplanation: {explanation}"

    try:
        if MODEL_BACKEND == "huggingface":
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": text},
                timeout=30
            )
            return response.json()

        elif MODEL_BACKEND == "colab":
            response = requests.post(
                COLAB_API_URL,
                json={"text": text},
                timeout=30
            )
            return response.json()

    except Exception as e:
        print(f"Model backend error: {e}")
        return None  # Fallback to heuristic

# Modify build_feedback() to use model if available
async def build_feedback_with_model(question: dict, selected_answer: str, explanation: str) -> dict:
    is_correct = normalize_answer(selected_answer) == normalize_answer(question["correct_answer"])

    # Try ML model first
    model_result = await get_model_prediction(
        question["text"],
        selected_answer,
        explanation,
        is_correct
    )

    if model_result:
        # Use model prediction
        reasoning = {
            "reasoning_band": model_result.get("reasoning_band", "Neither"),
            "misconception": model_result.get("misconception", "NA"),
            "confidence": model_result.get("confidence", 0.8)
        }
    else:
        # Fallback to heuristic
        reasoning = classify_reasoning(question, selected_answer, explanation)

    # Rest of feedback building...
    category = f"{'True' if is_correct else 'False'}_{reasoning['reasoning_band']}"
    # ... (same as before)
```

---

## Quick Start Recommendation

For your ML project demo, I recommend:

1. **Development**: Use the existing heuristic (no model needed)
2. **Demo Day**: Run Google Colab notebook manually before demo, get ngrok URL
3. **Production**: Upload to Hugging Face and use their API

---

## Summary Table

| Option | Cost | Setup Time | Reliability | Best For |
|--------|------|------------|-------------|----------|
| Kaggle | Free | Medium | Low (no real-time) | Batch processing |
| Colab + ngrok | Free | Low | Low (sessions timeout) | Quick demos |
| Hugging Face | Free/Paid | Medium | Medium | Small-scale deployment |
| Modal/Replicate | ~$0.0003/req | Low | High | Production |
| Local (quantized) | Free | Low | High | If you have GPU |

---

## Files to Create

If you want me to help implement any of these options, let me know which one you prefer and I can:

1. Create the Colab notebook for inference API
2. Modify `student_app/app.py` to call the model backend
3. Create the Hugging Face upload script
4. Set up Modal.com deployment
