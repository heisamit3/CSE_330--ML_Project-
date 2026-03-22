# MAP Misconception Detection - Deployment

Full-stack deployment for the MAP (Math Assessment Platform) Misconception Detection system with Qwen3-4B model and Explainable AI.

## Architecture

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│         FRONTEND                │     │         BACKEND                 │
│    (Vercel - Free tier)         │────▶│    (Modal - Free GPU)           │
│                                 │     │                                 │
│  • React 18                     │     │  • FastAPI                      │
│  • KaTeX math rendering         │     │  • Qwen3-4B + LoRA              │
│  • Dark/Light theme             │     │  • Explainable AI               │
│  • Score tracking               │     │  • 4-bit quantization           │
└─────────────────────────────────┘     └─────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git
- HuggingFace account (for model access)
- Modal account (free tier includes GPU credits)

### 1. Prepare Data Artifacts

```bash
cd deployment/backend

# Install dependencies
pip install pandas scikit-learn joblib

# Generate required data files
python prepare_data.py --train-path ../../DS/train.csv --output-dir ./data
```

This creates:
- `data/train.csv` - Training data copy
- `data/label_encoder.pkl` - 65-class label encoder
- `data/correct_answers.json` - Question → correct answer map
- `data/misconception_kb.json` - Basic knowledge base

### 2. Upload LoRA Adapter (Required for Real Model)

Your trained LoRA adapter needs to be accessible. Options:

**Option A: HuggingFace Hub (Recommended)**
```bash
# Install huggingface_hub
pip install huggingface-hub

# Login
huggingface-cli login

# Upload your adapter
huggingface-cli upload your-username/map-qwen3-4b-adapter ./path/to/adapter/
```

**Option B: Include in deployment**
Copy adapter files to `deployment/backend/data/adapter/`

### 3. Run Backend Locally

```bash
cd deployment/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings:
# - HF_TOKEN=your_huggingface_token
# - ADAPTER_PATH=your-username/map-qwen3-4b-adapter

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test the API:
```bash
curl http://localhost:8000/health
```

### 4. Run Frontend Locally

```bash
cd deployment/frontend

# Install dependencies
npm install

# Copy and configure environment
cp .env.example .env.local
# Edit .env.local:
# REACT_APP_API_BASE_URL=http://localhost:8000

# Run development server
npm start
```

Open http://localhost:3000 in your browser.

---

## Deployment

### Deploy Backend to Modal (Free GPU)

Modal provides free GPU credits - perfect for university projects.

#### Step 1: Install Modal CLI
```bash
pip install modal
modal setup  # Follow the prompts to authenticate
```

#### Step 2: Create HuggingFace Secret in Modal
```bash
modal secret create hf-secret HF_TOKEN=your_huggingface_token_here
```

#### Step 3: Deploy
```bash
cd deployment/backend
modal deploy modal_app.py
```

After deployment, you'll get URLs like:
- `https://your-username--map-misconception-detector-misconceptionapi-health.modal.run`
- `https://your-username--map-misconception-detector-misconceptionapi-predict.modal.run`

**Note the base URL pattern** - you'll need this for the frontend.

#### Step 4: Test Deployment
```bash
curl https://your-modal-url/health
```

### Deploy Frontend to Vercel

#### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

#### Step 2: Configure Environment
In Vercel dashboard or via CLI, set:
```
REACT_APP_API_BASE_URL=https://your-username--map-misconception-detector-misconceptionapi
```

**Important**: The Modal endpoints use a specific pattern. Your API calls will go to:
- `${REACT_APP_API_BASE_URL}-health.modal.run`
- `${REACT_APP_API_BASE_URL}-predict.modal.run`

You may need to update `frontend/src/services/api.js` to handle Modal's URL pattern.

#### Step 3: Deploy
```bash
cd deployment/frontend
vercel --prod
```

---

## Configuration Reference

### Backend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BASE_MODEL_PATH` | HuggingFace model ID | `Qwen/Qwen3-4B` |
| `ADAPTER_PATH` | Path to LoRA adapter | `./data/adapter` |
| `USE_4BIT_QUANTIZATION` | Enable 4-bit quantization | `true` |
| `TORCH_DTYPE` | Torch data type | `float16` |
| `USE_LLM_EXPLAINER` | Use LLM for explanations | `true` |
| `HF_TOKEN` | HuggingFace API token | Required |
| `PORT` | Server port | `8000` |

### Frontend Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `REACT_APP_API_BASE_URL` | Backend API URL | `http://localhost:8000` |

---

## API Endpoints

### Health Check
```
GET /health
Response: { status, model_loaded, explainer_loaded, kb_entries, label_classes }
```

### Prediction
```
POST /predict
Body: { question_id, question_text, mc_answer, student_explanation, is_correct? }
Response: { success, prediction, feedback }
```

### Submit (Legacy UI)
```
POST /api/submit
Body: { question_id, selected_answer, student_explanation }
Response: { success, feedback }
```

### Get Question
```
GET /api/question?question_id=123
Response: { id, text, options, has_image }
```

---

## Troubleshooting

### "Model not loaded" error
- Check HF_TOKEN is valid
- Ensure you have access to gated models
- Check GPU memory availability

### CORS errors
- Verify REACT_APP_API_BASE_URL matches your backend
- Check CORS middleware in backend allows your frontend origin

### Modal cold starts
- First request may take 30-60 seconds while model loads
- Consider keeping container warm with periodic health checks

### Memory issues
- Enable 4-bit quantization: `USE_4BIT_QUANTIZATION=true`
- Use smaller MAX_LEN if needed

---

## File Structure

```
deployment/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Configuration
│   │   ├── models/
│   │   │   ├── classifier.py    # Qwen3-4B classifier
│   │   │   └── explainer.py     # Explainable AI
│   │   ├── services/
│   │   │   └── prediction.py    # Prediction pipeline
│   │   ├── schemas/
│   │   │   └── __init__.py      # Pydantic models
│   │   └── utils/
│   │       ├── preprocessing.py
│   │       └── postprocessing.py
│   ├── data/                    # Generated data files
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── modal_app.py             # Modal deployment
│   ├── prepare_data.py          # Data preparation
│   └── .env.example
│
└── frontend/
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.js
    │   ├── index.js
    │   ├── config.js
    │   ├── components/
    │   │   ├── Navbar.jsx
    │   │   ├── Hero.jsx
    │   │   ├── QuestionCard.jsx
    │   │   ├── AnswerForm.jsx
    │   │   ├── FeedbackPanel.jsx
    │   │   └── Toast.jsx
    │   ├── hooks/
    │   │   ├── useScore.js
    │   │   ├── useQuestion.js
    │   │   └── useSubmit.js
    │   ├── services/
    │   │   └── api.js
    │   ├── styles/
    │   │   └── globals.css
    │   └── utils/
    │       └── math.js
    ├── package.json
    ├── vercel.json
    └── .env.example
```

---

## License

MIT License - Use freely for educational purposes.
