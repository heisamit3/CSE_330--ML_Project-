# MAP — Math Misconception Detector (Web UI)

A beautiful, modern web interface for the MAP pipeline that detects student math misconceptions using a DeBERTa + BGE ensemble model.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Beautiful glassmorphism UI** with dark/light theme
- **Real-time analysis** of student math explanations
- **Top-3 predictions** with confidence scores and animated visualizations
- **Explainable AI** section with detailed misconception descriptions
- **Model dashboard** showing pipeline architecture and techniques
- **Responsive design** — works on desktop, tablet, and mobile
- **One-click examples** for quick demos
- **Feedback collection** for model improvement

## Quick Start

### 1. Install dependencies

```bash
cd web_app
pip install -r requirements.txt
```

### 2. Run the server

```bash
python app.py
```

Or with uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open in browser

Navigate to **http://localhost:8000**

## Integrating Your ML Model

The app ships with a `MockPredictor` for demo purposes. To integrate your trained models:

### Step 1: Edit `app.py`

Replace the `MockPredictor` class with your real model:

```python
class RealPredictor:
    def __init__(self):
        # Load your trained DeBERTa + BGE models
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        self.model = AutoModelForSequenceClassification.from_pretrained("./fold_models/best")
        self.bge = SentenceTransformer("BAAI/bge-large-en-v1.5")
        # ... load ensemble weights, etc.

    def predict(self, question: str, answer: str, explanation: str):
        # 1. Build structured input text
        text = f"Question: {question}\nAnswer: {answer}\nStudent Explanation: {explanation}"
        
        # 2. Run DeBERTa classification
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        deberta_probs = torch.softmax(logits / TEMPERATURE, dim=-1).numpy()[0]
        
        # 3. Run BGE retrieval
        query_emb = self.bge.encode(text)
        similarities = cosine_similarity([query_emb], label_embeddings)[0]
        retrieval_probs = softmax(similarities / 20)
        
        # 4. Ensemble
        probs = alpha * deberta_probs + (1 - alpha) * retrieval_probs
        
        # 5. Apply family filter + get top-3
        top3_indices = np.argsort(probs)[::-1][:3]
        
        results = []
        for rank, idx in enumerate(top3_indices, 1):
            results.append({
                "rank": rank,
                "label": CLASSES[idx],
                "category": ...,
                "misconception": ...,
                "confidence": float(probs[idx]),
                ...
            })
        return results
```

### Step 2: Copy model weights

Place your trained model checkpoints in the `web_app/` directory or update paths accordingly.

### Step 3: Restart the server

```bash
python app.py
```

## Deployment

### Docker

```bash
cd web_app
docker build -t map-misconception-detector .
docker run -p 8000:8000 map-misconception-detector
```

### Render.com

1. Push to GitHub
2. Create a new Web Service on Render
3. Set **Build Command**: `pip install -r requirements.txt`  
4. Set **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy!

### Railway

```bash
# Install Railway CLI, then:
railway init
railway up
```

### Hugging Face Spaces

1. Create a new Space (SDK: Docker or Gradio)
2. Upload the `web_app/` folder
3. The Dockerfile handles everything

## Project Structure

```
web_app/
├── app.py                  # FastAPI backend + API routes
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container deployment
├── .dockerignore
├── README.md               # This file
├── templates/
│   └── index.html          # Main HTML template
└── static/
    ├── css/
    │   └── style.css       # All styles (glassmorphism, animations)
    └── js/
        └── app.js          # Frontend logic (SPA navigation, API calls)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main web interface |
| `POST` | `/api/predict` | Run misconception prediction |
| `POST` | `/api/feedback` | Submit feedback on predictions |
| `GET` | `/api/stats` | Get model & dataset statistics |
| `GET` | `/api/examples` | Get sample questions for demo |

### Prediction Request

```json
POST /api/predict
{
    "question": "What fraction of the shape is not shaded?",
    "answer": "1/3",
    "explanation": "1/3 because 6 over 9 is 2 thirds"
}
```

### Prediction Response

```json
{
    "success": true,
    "predictions": [
        {
            "rank": 1,
            "label": "True_Correct:NA",
            "category": "True_Correct",
            "misconception": "NA",
            "confidence": 0.82,
            "category_description": "Student answered correctly with proper reasoning",
            "misconception_description": "No specific misconception identified",
            "category_color": "#10b981"
        },
        ...
    ],
    "explanation": "The student's response was classified as True_Correct..."
}
```

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS (no build step needed!)
- **Design**: Glassmorphism, CSS animations, dark/light themes
- **Fonts**: Inter + JetBrains Mono (Google Fonts)
- **Icons**: Font Awesome 6
