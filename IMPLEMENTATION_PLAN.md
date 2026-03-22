# Implementation Plan: Model Integration with UI

## Executive Summary

Convert the notebook-based Qwen3-4B inference pipeline (`q3-4b-exp.ipynb`) into a production-ready full-stack application integrated with the existing UI (`student_app`).

---

## 1. Repository Analysis

### 1.1 Source of Truth (Good Notebook)
**File:** `Final_Model/Inference/q3-4b-exp.ipynb`

This notebook contains:
- Model loading (Qwen3-4B base + LoRA adapter)
- Tokenization with MAX_LEN=256
- 65-class classification with family filtering (True_/False_ prefix)
- **Explainable AI (GOOD):**
  - Misconception Knowledge Base (KB) building using Qwen3-4B CausalLM
  - LLM-generated KB cards with: `plain_definition`, `typical_wrong_thought`, `correct_rule`, `fix_steps`, `quick_check`, `tiny_example`
  - Teacher feedback generation using same Qwen3-4B
  - Rule-based verification (`rule_based_verify`) - softens language for correct answers
  - Optional LLM verification (`llm_verify`)

### 1.2 Current UI (Bad Fallback Logic)
**File:** `student_app/app.py`

Uses:
- Rule-based similarity matching (SequenceMatcher + Jaccard)
- Hardcoded `MISCONCEPTION_FEEDBACK` dictionary (~25 entries)
- No real model inference
- No LLM-generated explanations

### 1.3 Frontend
**Files:** `student_app/templates/index.html`, `student_app/static/js/app.js`, `student_app/static/css/style.css`

- Vanilla JavaScript with KaTeX for math rendering
- FastAPI Jinja2 templating
- Clean UI with dark/light theme
- Score tracking, star rating

### 1.4 Model Artifacts
| Artifact | Location | Format |
|----------|----------|--------|
| Base Model | Kaggle: `mdmarufhasanrubab/qwen3-4b-base` | HuggingFace format (safetensors) |
| LoRA Adapter | Kaggle: `mkhanshovon43/map-qwen3-4b-trained` | PEFT adapter_config.json + weights |
| Label Encoder | Not saved | Need to reconstruct from train.csv |
| Knowledge Base | `misconception_kb.json` (generated) | JSON with 35+ entries |

### 1.5 Kaggle-Specific Code to Refactor
```python
# These hardcoded paths must become configurable:
TRAIN_PATH = "/kaggle/input/datasets/mkhanshovon43/map-ds/DS/train.csv"
TEST_PATH = "/kaggle/input/datasets/mkhanshovon43/map-ds/DS/test.csv"
BASE_MODEL_CANDIDATES = ["/kaggle/input/..."]
ADAPTER_CANDIDATES = ["/kaggle/input/..."]
```

---

## 2. Architecture Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Vercel)                          │
│  React App with:                                                   │
│  - Question display with KaTeX                                     │
│  - MCQ selection                                                   │
│  - Explanation input                                               │
│  - Feedback display (predictions + explainable AI)                 │
│  - Loading/error states                                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTPS API calls
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BACKEND (GPU Cloud - Modal/HF)                  │
│  FastAPI Service with:                                             │
│  - /health                                                         │
│  - /predict (classification + explainable AI)                      │
│  - /labels (optional - return all 65 classes)                      │
│  - /questions (proxy to questions data)                            │
│                                                                    │
│  Model Components:                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ Qwen3-4B        │  │ LoRA Adapter    │  │ Qwen3-4B CausalLM   │ │
│  │ Classifier      │  │ (PEFT)          │  │ (Explainer)         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│                                                                    │
│  Support:                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ Label Encoder   │  │ Misconception   │  │ Train Data          │ │
│  │ (65 classes)    │  │ KB (JSON)       │  │ (for correct_by_q)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Checklist

### Phase 1: Backend Service Creation

#### 3.1 Create Backend Directory Structure
```
model_backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Environment config
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py    # Qwen3-4B + LoRA classification
│   │   └── explainer.py     # Qwen3-4B CausalLM explainer
│   ├── services/
│   │   ├── __init__.py
│   │   ├── prediction.py    # Prediction pipeline
│   │   └── feedback.py      # Teacher feedback generation
│   ├── schemas/
│   │   └── __init__.py      # Pydantic models
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py # Text formatting
│       └── postprocessing.py # Family filtering, verification
├── data/
│   ├── train.csv            # Copy of DS/train.csv
│   ├── label_encoder.pkl    # Saved LabelEncoder
│   └── misconception_kb.json # Pre-built KB
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

#### 3.2 Extract Core Logic from Notebook

| Notebook Function | Backend Location | Notes |
|-------------------|------------------|-------|
| `format_input()` | `utils/preprocessing.py` | Input text formatting |
| `find_model_path()` | `config.py` | Make configurable via env vars |
| `find_adapter_path()` | `config.py` | Make configurable via env vars |
| Model loading | `models/classifier.py` | Load once on startup |
| `softmax` + `top_indices` | `services/prediction.py` | Prediction logic |
| Family filtering | `utils/postprocessing.py` | Filter by True_/False_ prefix |
| `load_explainer()` | `models/explainer.py` | 4-bit quantization support |
| `llm_generate()` | `models/explainer.py` | Text generation |
| KB building | Pre-compute, save as JSON | Don't build at runtime |
| `generate_teacher_feedback()` | `services/feedback.py` | Core explainable AI |
| `rule_based_verify()` | `utils/postprocessing.py` | Post-processing |
| `llm_verify()` | `services/feedback.py` | Optional LLM cleanup |

#### 3.3 API Endpoints

**POST /predict**
```json
// Request
{
  "question_id": "31772",
  "question_text": "What fraction of the shape is not shaded?",
  "mc_answer": "3/9",
  "student_explanation": "There is 3 not shaded and nine triangles",
  "is_correct": false  // optional, can be inferred
}

// Response
{
  "success": true,
  "prediction": {
    "top_predictions": [
      {"label": "False_Misconception:Incomplete", "confidence": 0.87},
      {"label": "False_Neither:NA", "confidence": 0.08},
      {"label": "False_Misconception:Wrong_Fraction", "confidence": 0.05}
    ],
    "predicted_category": "False_Misconception",
    "predicted_misconception": "Incomplete"
  },
  "feedback": {
    "teacher_feedback": "The student seems to think that...",
    "misconception_card": {
      "plain_definition": "...",
      "typical_wrong_thought": ["..."],
      "correct_rule": ["..."],
      "fix_steps": ["..."],
      "quick_check": "...",
      "tiny_example": "..."
    },
    "correct_answer": "1/3",
    "is_student_correct": false
  }
}
```

**GET /health**
```json
{
  "status": "ok",
  "model_loaded": true,
  "explainer_loaded": true,
  "kb_entries": 35,
  "label_classes": 65
}
```

**GET /labels**
```json
{
  "labels": ["True_Correct:NA", "True_Neither:NA", "False_Misconception:Incomplete", ...],
  "count": 65
}
```

#### 3.4 Configuration via Environment Variables

```bash
# .env.example
# Model paths (HuggingFace Hub IDs or local paths)
BASE_MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
ADAPTER_PATH=/app/adapter  # or HF Hub ID

# Quantization
USE_4BIT_QUANTIZATION=true
TORCH_DTYPE=float16

# Explainer
USE_LLM_EXPLAINER=true
USE_LLM_VERIFIER=false

# Data paths
TRAIN_DATA_PATH=/app/data/train.csv
KB_PATH=/app/data/misconception_kb.json
LABEL_ENCODER_PATH=/app/data/label_encoder.pkl

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1
```

---

### Phase 2: React Frontend

#### 3.5 Create React App Structure
```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── index.js
│   ├── App.js
│   ├── components/
│   │   ├── Navbar.jsx
│   │   ├── Hero.jsx
│   │   ├── QuestionCard.jsx
│   │   ├── AnswerForm.jsx
│   │   ├── OptionCard.jsx
│   │   ├── FeedbackPanel.jsx
│   │   ├── ResultBanner.jsx
│   │   ├── MisconceptionCard.jsx
│   │   ├── SolutionCard.jsx
│   │   ├── StarRating.jsx
│   │   ├── Toast.jsx
│   │   └── LoadingSpinner.jsx
│   ├── hooks/
│   │   ├── useQuestion.js
│   │   ├── useSubmit.js
│   │   └── useScore.js
│   ├── services/
│   │   └── api.js           # API client
│   ├── styles/
│   │   └── globals.css      # Port existing CSS
│   ├── utils/
│   │   └── math.js          # KaTeX rendering
│   └── config.js            # Environment config
├── package.json
├── .env.example
├── .env.local
├── vercel.json
└── README.md
```

#### 3.6 Key React Components

**Preserve existing UI features:**
- Dark/light theme toggle
- Score tracking (correct/wrong/total)
- Question display with image placeholders
- MCQ option cards with selection state
- Explanation textarea with character count
- Loading spinner during submission
- Feedback panel with:
  - Result banner (color-coded)
  - Answer comparison
  - Solution display
  - Misconception card (when detected)
  - Star rating

**New features to add:**
- Top-3 predictions display with confidence bars
- Full KB card display (from explainable AI)
- LLM-generated teacher feedback section
- API error handling with retry

#### 3.7 Environment Configuration

```bash
# frontend/.env.example
REACT_APP_API_BASE_URL=https://your-backend.modal.run
# or for local dev:
# REACT_APP_API_BASE_URL=http://localhost:8000
```

---

### Phase 3: Deployment

#### 3.8 Backend Deployment Options (Free for University Project)

| Platform | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Modal** | Free $30/month credits, serverless GPU, easy Python deployment | Cold starts | **Best for demo** |
| **Hugging Face Spaces** | Free T4 GPU tier, Docker support | 16GB RAM limit, may not fit model | Try, but may fail |
| **Google Colab + ngrok** | Free, familiar | Unreliable, session timeouts | **Backup option** |
| **RunPod** | Cheap GPUs, persistent | Requires payment | For serious use |
| **Kaggle Notebooks** | Free H100, already works | Can't run as API | Not suitable |

**Recommended: Modal**
- Free tier includes GPU compute credits
- Native Python support with `@modal.web_endpoint`
- Auto-scaling and cold start handling
- Easy HuggingFace model downloads

#### 3.9 Modal Deployment Files

```python
# modal_app.py
import modal

app = modal.App("map-misconception-detector")

image = modal.Image.debian_slim().pip_install(
    "fastapi", "torch", "transformers", "peft",
    "pandas", "scikit-learn", "scipy"
).run_commands(
    "pip install bitsandbytes"
)

@app.cls(
    image=image,
    gpu="T4",  # or "A10G" for better performance
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("hf-secret")]
)
class MisconceptionAPI:
    @modal.enter()
    def load_model(self):
        # Load model on container start
        pass

    @modal.web_endpoint(method="POST")
    def predict(self, request: dict):
        # Prediction endpoint
        pass
```

#### 3.10 Frontend Deployment (Vercel)

```json
// vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": "build",
  "framework": "create-react-app",
  "rewrites": [
    { "source": "/api/:path*", "destination": "https://your-backend.modal.run/:path*" }
  ]
}
```

---

### Phase 4: Data Preparation

#### 3.11 Pre-compute Artifacts

**Before deployment, generate these files:**

1. **Label Encoder** (`label_encoder.pkl`)
   ```python
   from sklearn.preprocessing import LabelEncoder
   import joblib
   import pandas as pd

   train_df = pd.read_csv("DS/train.csv")
   train_df['target'] = train_df['Category'] + ':' + train_df['Misconception'].fillna('NA')

   le = LabelEncoder()
   le.fit(train_df['target'])
   joblib.dump(le, "data/label_encoder.pkl")
   ```

2. **Misconception KB** (`misconception_kb.json`)
   - Already generated by notebook, copy to `model_backend/data/`
   - Or re-generate with the KB building logic

3. **Correct Answers Map** (`correct_answers.json`)
   ```python
   import json
   # From notebook logic
   correct_by_q_full = {...}  # QuestionId -> correct MC_Answer
   with open("data/correct_answers.json", "w") as f:
       json.dump(correct_by_q_full, f)
   ```

4. **Train Data Subset** (`train.csv`)
   - Copy `DS/train.csv` to `model_backend/data/`
   - Used for inferring `is_correct` and lookup

---

## 4. File Creation Summary

### Files to Create

| File | Purpose |
|------|---------|
| `model_backend/app/main.py` | FastAPI application |
| `model_backend/app/config.py` | Environment configuration |
| `model_backend/app/models/classifier.py` | Qwen3-4B + LoRA loading |
| `model_backend/app/models/explainer.py` | CausalLM explainer |
| `model_backend/app/services/prediction.py` | Prediction pipeline |
| `model_backend/app/services/feedback.py` | Teacher feedback generation |
| `model_backend/app/schemas/__init__.py` | Pydantic request/response models |
| `model_backend/app/utils/preprocessing.py` | Input formatting |
| `model_backend/app/utils/postprocessing.py` | Family filtering, verification |
| `model_backend/requirements.txt` | Python dependencies |
| `model_backend/Dockerfile` | Docker image |
| `model_backend/.env.example` | Environment template |
| `model_backend/README.md` | Setup instructions |
| `model_backend/modal_app.py` | Modal deployment |
| `frontend/` | React application (full structure above) |
| `frontend/vercel.json` | Vercel configuration |

### Files to Modify

| File | Changes |
|------|---------|
| None | User requested no code changes, only plan |

### Files to Copy/Generate

| Source | Destination |
|--------|-------------|
| `DS/train.csv` | `model_backend/data/train.csv` |
| Generated | `model_backend/data/label_encoder.pkl` |
| Generated | `model_backend/data/misconception_kb.json` |
| Generated | `model_backend/data/correct_answers.json` |

---

## 5. Running Instructions (Preview)

### Local Development

**Backend:**
```bash
cd model_backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your model paths
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
cp .env.example .env.local
# Edit .env.local with backend URL
npm start
```

### Deployment

**Backend (Modal):**
```bash
cd model_backend
modal deploy modal_app.py
# Note the endpoint URL
```

**Frontend (Vercel):**
```bash
cd frontend
vercel --prod
# Set REACT_APP_API_BASE_URL in Vercel dashboard
```

---

## 6. Risk Mitigation

### 6.1 Model Size Challenge
- Qwen3-4B is ~8GB in float16
- **Mitigation:** Use 4-bit quantization (bitsandbytes) to reduce to ~2-3GB
- Modal T4 has 16GB VRAM, should fit

### 6.2 Cold Start Latency
- First request may take 30-60 seconds to load model
- **Mitigation:**
  - Keep container warm with periodic health checks
  - Show loading UI with expected wait time
  - Consider pre-warming on deploy

### 6.3 HuggingFace Model Access
- Base model `Qwen/Qwen3-4B-Instruct-2507` requires HF token
- LoRA adapter is on Kaggle, needs to be uploaded to HF Hub or stored elsewhere
- **Mitigation:**
  - Upload adapter to HuggingFace Hub (private repo)
  - Use HF_TOKEN secret in Modal

### 6.4 Explanation Quality Fallback
- If LLM generation fails, use structured fallback from KB card
- **Mitigation:** Already in notebook code, port to backend

---

## 7. Success Criteria Verification

| Criteria | How to Verify |
|----------|---------------|
| Run backend locally | `curl http://localhost:8000/health` returns 200 |
| Run frontend locally | `npm start` opens at localhost:3000 |
| Frontend connects to backend | Submit answer, get real predictions |
| Real model predictions | Response includes `top_predictions` with confidences |
| Explainable AI outputs | Response includes `teacher_feedback` and `misconception_card` |
| Deploy frontend to Vercel | Visit vercel URL, app loads |
| Deploy backend to GPU service | Health endpoint accessible, model_loaded: true |

---

## 8. Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1 | Backend service creation | 4-6 hours |
| Phase 2 | React frontend | 3-4 hours |
| Phase 3 | Deployment setup | 2-3 hours |
| Phase 4 | Data preparation | 1 hour |
| Testing | End-to-end testing | 2 hours |
| **Total** | | **12-16 hours** |

---

## 9. Questions/Decisions Needed

1. **LoRA Adapter Storage:** Upload to HuggingFace Hub or include in Docker image?
   - Recommendation: HF Hub (easier updates, smaller image)

2. **Questions Data:** Serve from backend or embed in frontend?
   - Recommendation: Backend (keeps data centralized)

3. **Pre-built KB vs Runtime KB:** Use existing KB or rebuild?
   - Recommendation: Pre-built (faster, consistent)

4. **Vercel API Proxy:** Use Vercel rewrites or direct CORS?
   - Recommendation: Direct CORS (simpler for demo)

---

## Appendix: Key Code Snippets to Extract

### A1. Input Formatting (from notebook)
```python
def format_input(row):
    x = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )
```

### A2. Family Filtering (from notebook)
```python
pref = 'True_' if is_correct == 1 else 'False_'
filtered = [lbl for lbl in top_labels if lbl.startswith(pref)]
top3 = filtered[:3]
```

### A3. Rule-Based Verification (from notebook)
```python
def rule_based_verify(feedback: str, is_correct: int, correct_choice: str) -> str:
    txt = feedback.strip()
    if is_correct == 1:
        txt = re.sub(r"\bincorrect\b", "not quite right", txt, flags=re.I)
        txt = re.sub(r"\bwrong\b", "not quite right", txt, flags=re.I)
    if is_correct == 0 and correct_choice and correct_choice not in txt:
        txt = f"Correct answer: {correct_choice}\n\n" + txt
    return txt
```

### A4. LLM Generation (from notebook)
```python
def llm_generate(prompt: str, max_new_tokens=320, explainer_mdl=None, explainer_tok=None):
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = explainer_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = explainer_tok(chat_prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(explainer_mdl.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = explainer_mdl.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=explainer_tok.eos_token_id
        )

    full_text = explainer_tok.decode(out[0], skip_special_tokens=True)
    # ... extract generated part
    return text
```

---

**Plan created:** 2026-03-22
**Status:** Ready for implementation approval
