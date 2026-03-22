"""
MAP Misconception Detection API
FastAPI backend for student misconception detection with explainable AI.
"""

import csv
import json
import logging
import random
from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import HOST, PORT, TRAIN_DATA_PATH
from .schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    LabelsResponse,
    QuestionResponse,
    QuestionOption,
    SubmitRequest,
    SubmitResponse,
)
from .services import get_prediction_service
from .models import get_classifier, get_explainer
from .utils import normalize_space, normalize_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Questions database (loaded from train data)
QUESTIONS_DB = {}


def load_questions():
    """Load questions from training data."""
    global QUESTIONS_DB
    QUESTIONS_DB.clear()

    train_path = Path(TRAIN_DATA_PATH)
    if not train_path.exists():
        logger.warning(f"Train data not found at {train_path}")
        return

    try:
        grouped_rows = defaultdict(list)
        with open(train_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                grouped_rows[row["QuestionId"]].append(row)

        for question_id, rows in grouped_rows.items():
            question_text = rows[0]["QuestionText"]

            # Get unique answers
            answer_counts = defaultdict(int)
            for row in rows:
                answer = normalize_space(row["MC_Answer"])
                category = normalize_space(row["Category"])
                if category.startswith("True"):
                    answer_counts[answer] += 10  # Weight correct answers
                else:
                    answer_counts[answer] += 1

            # Get top 4 answers as options
            sorted_answers = sorted(answer_counts.items(), key=lambda x: -x[1])
            options = [a[0] for a in sorted_answers[:4]]

            # Infer correct answer
            correct_answer = None
            for row in rows:
                if normalize_space(row["Category"]) == "True_Correct":
                    correct_answer = normalize_space(row["MC_Answer"])
                    break

            if correct_answer is None and options:
                correct_answer = options[0]

            QUESTIONS_DB[question_id] = {
                "id": question_id,
                "text": question_text,
                "options": options,
                "correct_answer": correct_answer,
            }

        logger.info(f"Loaded {len(QUESTIONS_DB)} questions")
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting MAP Misconception Detection API...")

    # Load questions
    load_questions()

    # Load models
    service = get_prediction_service()
    service.load()

    logger.info("API ready!")
    yield

    # Shutdown
    logger.info("Shutting down...")
    get_classifier().cleanup()
    get_explainer().cleanup()


app = FastAPI(
    title="MAP Misconception Detection API",
    description="API for detecting student misconceptions in math with explainable AI feedback",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    classifier = get_classifier()
    explainer = get_explainer()

    return HealthResponse(
        status="ok",
        model_loaded=classifier.is_loaded,
        explainer_loaded=explainer.is_loaded,
        kb_entries=explainer.kb_count,
        label_classes=classifier.n_classes if classifier.is_loaded else 0,
    )


@app.get("/labels", response_model=LabelsResponse)
async def get_labels():
    """Get all label classes."""
    classifier = get_classifier()
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    labels = classifier.get_labels()
    return LabelsResponse(labels=labels, count=len(labels))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run prediction on student answer and explanation.
    Returns classification result and explainable AI feedback.
    """
    service = get_prediction_service()
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Service not loaded")

    try:
        result = service.predict(
            question_id=request.question_id,
            question_text=request.question_text,
            mc_answer=request.mc_answer,
            student_explanation=request.student_explanation,
            is_correct=request.is_correct,
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/question", response_model=QuestionResponse)
async def get_question(question_id: Optional[str] = None):
    """Get a question (random or by ID)."""
    if not QUESTIONS_DB:
        raise HTTPException(status_code=500, detail="No questions available")

    if question_id and question_id in QUESTIONS_DB:
        q = QUESTIONS_DB[question_id]
    else:
        q = random.choice(list(QUESTIONS_DB.values()))

    options = [
        QuestionOption(id=chr(65 + i), text=opt)
        for i, opt in enumerate(q["options"])
    ]

    return QuestionResponse(
        id=q["id"],
        text=q["text"],
        options=options,
        has_image="[Image:" in q["text"] or "[image:" in q["text"].lower(),
    )


@app.get("/api/questions/list")
async def list_questions():
    """List all available questions."""
    questions = []
    for qid, q in QUESTIONS_DB.items():
        preview = q["text"][:100] + ("..." if len(q["text"]) > 100 else "")
        questions.append({
            "id": qid,
            "preview": preview,
            "option_count": len(q["options"]),
        })
    return {"questions": questions}


@app.post("/api/submit", response_model=SubmitResponse)
async def submit_answer(req: SubmitRequest):
    """
    Submit answer endpoint (legacy compatibility with existing UI).
    """
    if req.question_id not in QUESTIONS_DB:
        raise HTTPException(status_code=404, detail="Question not found")

    question = QUESTIONS_DB[req.question_id]
    selected_answer = normalize_space(req.selected_answer or req.student_answer or "")
    explanation = normalize_space(req.student_explanation)

    if not selected_answer:
        raise HTTPException(status_code=400, detail="Please choose one answer option")

    # Validate answer
    valid_answers = {normalize_answer(opt): opt for opt in question["options"]}
    if normalize_answer(selected_answer) not in valid_answers:
        # Try to match
        matched = valid_answers.get(normalize_answer(selected_answer), selected_answer)
        selected_answer = matched

    service = get_prediction_service()
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Service not loaded")

    try:
        feedback = service.get_legacy_feedback(
            question_id=req.question_id,
            question_text=question["text"],
            selected_answer=selected_answer,
            student_explanation=explanation,
            options=question["options"],
        )
        return SubmitResponse(success=True, feedback=feedback)
    except Exception as e:
        logger.error(f"Submit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rate")
async def rate_feedback(question_id: str, rating: int, comment: Optional[str] = None):
    """Rate feedback (placeholder)."""
    return {"success": True, "message": "Thanks for rating!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
