# MAP — Math Practice Portal (Student App)

A student-facing web application where students practice math questions, provide their answers and reasoning, and receive AI-powered feedback that identifies misconceptions and provides learning guidance.

## Features

- **Practice Questions** — Randomly served from the training dataset
- **Answer + Explanation** — Students provide their answer and explain their reasoning
- **AI Feedback** — The system classifies the response and provides:
  - Whether the answer is correct or incorrect
  - The correct answer and solution
  - Detected misconceptions with student-friendly explanations
  - Tips on how to fix their thinking
- **Score Tracking** — Tracks correct/incorrect across the session
- **Star Ratings** — Students can rate the feedback quality
- **KaTeX Support** — Renders math expressions beautifully
- **Dark/Light Theme** — Toggle between themes
- **Responsive Design** — Works on mobile, tablet, and desktop

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (port 8001)
python app.py
```

Then open `http://localhost:8001` in your browser.

## Docker Deployment

```bash
# Build
docker build -t map-student-portal .

# Run
docker run -p 8001:8001 -v ./DS:/app/DS map-student-portal
```

> **Note:** Mount the `DS/` folder containing `train.csv` so the app can load questions. If the dataset isn't found, built-in example questions are used as a fallback.

## Integrating the Real Model

In `app.py`, replace the `MockPredictor` class with your trained model:

```python
class RealPredictor:
    def __init__(self):
        self.model = load_your_model()

    def predict(self, question, answer, explanation):
        # Run your DeBERTa + BGE ensemble
        # Return: { label, category, misconception, confidence }
        ...
```

## Project Structure

```
student_app/
├── app.py                  # FastAPI backend
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container deployment
├── README.md               # This file
├── templates/
│   └── index.html          # Main HTML template
└── static/
    ├── css/
    │   └── style.css       # Styles
    └── js/
        └── app.js          # Frontend logic
```

## API Endpoints

| Method | Path               | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/`               | Serve the student portal             |
| GET    | `/api/question`   | Get a random question                |
| GET    | `/api/questions/list` | List all available questions      |
| POST   | `/api/submit`     | Submit answer + explanation          |
| POST   | `/api/rate`       | Submit feedback rating               |
