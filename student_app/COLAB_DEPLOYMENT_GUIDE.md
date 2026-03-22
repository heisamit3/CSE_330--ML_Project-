# Google Colab Deployment Guide for Student App

This guide is the Colab-specific next step for serving the DeepSeek model and connecting it to `student_app`.

It is meant for demo use, not production. Colab sessions can stop, GPU access is not guaranteed, and ngrok URLs change between sessions.

## Goal

Run the trained model in Google Colab, expose a small `/predict` API, and let `student_app/app.py` call it.

## What You Need First

Before opening Colab, make sure you have these two things:

1. Your trained DeepSeek model folder
2. `DS/train.csv`

Why both are needed:

- The model folder contains the trained weights and tokenizer.
- `DS/train.csv` is needed because the inference notebook rebuilds the label mapping from the training labels.

## Recommended Google Drive Layout

Upload these files to Google Drive in a simple folder:

```text
MyDrive/map_colab/
  deepseek_model/
  train.csv
```

If your saved model folder contains files like `config.json`, `tokenizer.json`, `model.safetensors`, and related tokenizer/config files, you are in good shape.

## Step 1 - Create a New Colab Notebook

1. Open Google Colab.
2. Create a new notebook.
3. Go to `Runtime > Change runtime type`.
4. Set hardware accelerator to `GPU`.

## Step 2 - Install Packages

Run this in the first Colab cell:

```python
!pip install -q fastapi uvicorn pyngrok transformers torch pandas scikit-learn requests
```

## Step 3 - Mount Google Drive

Run:

```python
from google.colab import drive
drive.mount('/content/drive')

MODEL_DIR = "/content/drive/MyDrive/map_colab/deepseek_model"
TRAIN_CSV = "/content/drive/MyDrive/map_colab/train.csv"
```

After this, confirm the paths are correct:

```python
import os

print(os.path.exists(MODEL_DIR))
print(os.path.exists(TRAIN_CSV))
```

Both should print `True`.

## Step 4 - Rebuild the Same Input Logic Used by the Notebook

Your Colab API should match the logic from `Final_Model/Inference/deepseek_infer.ipynb`.

Important behavior to keep:

- Build label names from `train.csv`
- Recreate the `is_correct` flag from `QuestionId` and `MC_Answer`
- Format the model input exactly like this:

```python
def format_input(question_text, answer, explanation, is_correct):
    status = "This answer is correct." if is_correct else "This answer is incorrect."
    return (
        f"Question: {question_text}\n"
        f"Answer: {answer}\n"
        f"{status}\n"
        f"Student Explanation: {explanation}"
    )
```

- Decode top predictions with the label encoder
- Apply the same family filtering:
  - If `is_correct == 1`, keep labels starting with `True_`
  - Otherwise keep labels starting with `False_`

## Step 5 - Load the Training Labels and Model

Use a cell like this:

```python
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

train = pd.read_csv(TRAIN_CSV)
train["Misconception"] = train["Misconception"].fillna("NA")
train["target"] = train["Category"] + ":" + train["Misconception"]

le = LabelEncoder()
le.fit(train["target"])

idx = train["Category"].apply(lambda c: c.split("_")[0]) == "True"
correct = train.loc[idx].copy()
correct["c"] = correct.groupby(["QuestionId", "MC_Answer"])["MC_Answer"].transform("count")
correct = correct.sort_values("c", ascending=False).drop_duplicates(["QuestionId"])
correct_by_q = dict(zip(correct["QuestionId"], correct["MC_Answer"]))

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=len(le.classes_),
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model = model.to(device)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
```

## Step 6 - Create a Single-Prediction Function

Build a function that accepts one student submission and returns the top 3 labels:

```python
def predict_one(question_id, question_text, answer, explanation, is_correct):
    text = format_input(question_text, answer, explanation, is_correct)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    order = np.argsort(-probs)
    decoded = le.inverse_transform(order)

    prefix = "True_" if is_correct else "False_"
    filtered = [label for label in decoded if label.startswith(prefix)]
    top3 = filtered[:3]

    while len(top3) < 3:
        filler = f"{prefix}Neither:NA"
        if filler not in top3:
            top3.append(filler)
        else:
            top3.append(f"{prefix}Correct:NA" if prefix == "True_" else f"{prefix}Misconception:NA")

    top1 = top3[0]
    top1_index = np.where(le.classes_ == top1)[0][0]
    confidence = float(probs[top1_index])

    return {
        "prediction": top1,
        "top3": top3,
        "confidence": confidence,
        "correct_choice": str(correct_by_q.get(question_id, "")),
    }
```

## Step 7 - Wrap It in a Small FastAPI App

Create a Colab cell like this:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    question_id: int
    question: str
    answer: str
    explanation: str
    is_correct: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    return predict_one(
        question_id=req.question_id,
        question_text=req.question,
        answer=req.answer,
        explanation=req.explanation,
        is_correct=req.is_correct,
    )
```

## Step 8 - Expose the API with ngrok

1. Create a free ngrok account.
2. Copy your ngrok auth token.
3. Run this in Colab:

```python
from pyngrok import ngrok

NGROK_AUTH_TOKEN = "PASTE_YOUR_TOKEN_HERE"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
```

4. Start the app:

```python
import threading
import uvicorn

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_api)
thread.start()
```

5. Open a tunnel:

```python
public_url = ngrok.connect(8000).public_url
print(public_url)
```

Your prediction URL will be:

```text
<public_url>/predict
```

## Step 9 - Test the API Inside Colab

Before touching the student app, test it directly:

```python
import requests

url = "<public_url>/predict"
payload = {
    "question_id": 32829,
    "question": "2y = 24. What is the value of y?",
    "answer": "12",
    "explanation": "because 24 divided by 2 is 12",
    "is_correct": True,
}

response = requests.post(url, json=payload, timeout=30)
print(response.status_code)
print(response.json())
```

You want to see:

- status code `200`
- a JSON response with `prediction`, `top3`, and `confidence`

## Step 10 - Connect student_app to the Colab API

Once the API works, the next code change should be in `student_app/app.py`.

Recommended environment variables:

```text
MODEL_BACKEND=colab
COLAB_API_URL=https://your-ngrok-url.ngrok-free.app/predict
```

You will also need `requests` in `student_app/requirements.txt`.

The app flow should become:

1. Student submits answer and explanation
2. `student_app` sends the request to Colab
3. If Colab responds successfully, use the model prediction
4. If Colab fails, fall back to the current heuristic logic

## Suggested Request Format from student_app

Send this JSON from `student_app`:

```json
{
  "question_id": 32829,
  "question": "2y = 24. What is the value of y?",
  "answer": "12",
  "explanation": "because 24 divided by 2 is 12",
  "is_correct": true
}
```

## Suggested Response Format from Colab

Return something like:

```json
{
  "prediction": "True_Correct:NA",
  "top3": [
    "True_Correct:NA",
    "True_Neither:NA",
    "True_Misconception:Irrelevant"
  ],
  "confidence": 0.91,
  "correct_choice": "12"
}
```

## Step 11 - Keep the First Version Small

For the first working version, do not try to move everything at once.

Do this first:

- Return only prediction, top3, confidence, and correct_choice

Add later:

- Explainable AI output
- Teacher-style feedback from the XAI notebook
- Better caching and error handling

## Demo Checklist

Before demo time:

1. Start Colab with GPU
2. Run the install cell
3. Mount Google Drive
4. Load `train.csv` and model
5. Start FastAPI
6. Start ngrok
7. Copy the latest ngrok `/predict` URL
8. Start `student_app`
9. Make one manual test submission

## Known Limitations

- Colab sessions can disconnect
- ngrok URL changes when restarted
- Free GPU availability is not guaranteed
- This setup is suitable for demos and testing, not production deployment

## Best Next Step

After finishing this guide, the next practical task is:

1. create the Colab notebook cells
2. patch `student_app/app.py`
3. add `requests` to `student_app/requirements.txt`

If you want, the next step can be to implement the `student_app` side now.
