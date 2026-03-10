"""
MAP — Charting Student Math Misunderstandings
FastAPI Backend Server
"""

import os
import json
import random
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="MAP — Math Misconception Detector",
    description="Detect student math misconceptions using AI ensemble models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ============================================================
# LABEL DEFINITIONS (from the notebook)
# ============================================================
CLASSES = [
    "False_Correct:NA", "False_Misconception:Additive",
    "False_Misconception:Adding_across", "False_Misconception:Adding_terms",
    "False_Misconception:CoeffIsExp", "False_Misconception:Denominator-only_change",
    "False_Misconception:Duplication", "False_Misconception:Expand_single_bracket",
    "False_Misconception:Firstterm", "False_Misconception:FlipChange",
    "False_Misconception:Incomplete", "False_Misconception:Inversion",
    "False_Misconception:Irrelevant", "False_Misconception:Mult",
    "False_Misconception:Multiplying_by_4", "False_Misconception:Negative",
    "False_Misconception:Not_variable", "False_Misconception:Num-only_change",
    "False_Misconception:Positive", "False_Misconception:Scale",
    "False_Misconception:Shorter_is_bigger", "False_Misconception:Subtraction",
    "False_Misconception:SwapDividend", "False_Misconception:Tacking",
    "False_Misconception:Unknowable", "False_Misconception:WNB",
    "False_Misconception:Whole_numbers_larger", "False_Misconception:Wrong_Fraction",
    "False_Misconception:Wrong_fraction", "False_Misconception:Wrong_term",
    "False_Neither:Additive", "False_Neither:Incomplete",
    "False_Neither:Irrelevant", "False_Neither:NA",
    "False_Neither:Subtraction", "False_Neither:Wrong_fraction",
    "True_Correct:NA", "True_Misconception:Additive",
    "True_Misconception:Incomplete", "True_Misconception:Inversion",
    "True_Misconception:Irrelevant", "True_Misconception:Mult",
    "True_Misconception:Subtraction", "True_Misconception:Wrong_fraction",
    "True_Neither:Additive", "True_Neither:Incomplete",
    "True_Neither:Inversion", "True_Neither:Irrelevant",
    "True_Neither:Mult", "True_Neither:NA",
    "True_Neither:Subtraction", "True_Neither:Wrong_fraction",
]

MISCON_DESCRIPTIONS = {
    "NA": "No specific misconception identified",
    "Incomplete": "Incomplete reasoning — partial understanding but missing key steps",
    "Additive": "Additive reasoning — applies additive logic where multiplicative/proportional reasoning is needed",
    "Duplication": "Duplication error — repeats or duplicates numbers or operations incorrectly",
    "Subtraction": "Subtraction error — subtracts when a different operation is required",
    "Positive": "Positive number bias — treats negative numbers as positive or ignores sign rules",
    "Wrong_term": "Wrong term selection — identifies or uses the wrong term in a sequence",
    "Irrelevant": "Irrelevant reasoning — provides reasoning unrelated to the math concept",
    "Wrong_fraction": "Wrong fraction — forms or simplifies fractions incorrectly",
    "Inversion": "Inversion error — inverts fraction, ratio, or operation incorrectly",
    "Mult": "Multiplication error — makes errors in multiplication",
    "Denominator-only_change": "Only changes the denominator when both parts should change",
    "Whole_numbers_larger": "Believes longer decimals or more digits means larger value",
    "Adding_across": "Adds numerators and denominators separately when adding fractions",
    "WNB": "Wrong number bond — uses incorrect number bonds",
    "Tacking": "Appends digits without understanding place value",
    "Unknowable": "Claims the answer cannot be determined when it can",
    "SwapDividend": "Swaps the dividend and divisor in division",
    "Scale": "Misunderstands scale factors or proportional scaling",
    "Not_variable": "Treats a variable as a specific number rather than general value",
    "Wrong_Fraction": "Constructs the wrong fraction from given information",
    "Adding_terms": "Adds algebraic terms that should not be combined",
    "Firstterm": "Confuses the first term with the common difference",
    "Multiplying_by_4": "Multiplies by 4 when a different operation is needed",
    "FlipChange": "Incorrectly flips the sign or direction of change",
    "Shorter_is_bigger": "Believes shorter decimals represent larger values",
    "CoeffIsExp": "Confuses coefficients with exponents",
    "Negative": "Makes errors handling negative numbers or signs",
    "Expand_single_bracket": "Errors expanding a single bracket expression",
    "Num-only_change": "Only changes the numerator when both parts should change",
    "Smaller_is_bigger": "Confuses magnitude ordering of decimal numbers",
}

CATEGORY_DESCRIPTIONS = {
    "True_Correct": "Student answered correctly with proper reasoning",
    "True_Neither": "Student answered correctly but explanation is unclear",
    "True_Misconception": "Student answered correctly but reasoning has a misconception",
    "False_Correct": "Student answered incorrectly but shows correct reasoning process",
    "False_Neither": "Student answered incorrectly with unclear explanation",
    "False_Misconception": "Student answered incorrectly due to a specific misconception",
}

CATEGORY_COLORS = {
    "True_Correct": "#10b981",
    "True_Neither": "#f59e0b",
    "True_Misconception": "#f97316",
    "False_Correct": "#3b82f6",
    "False_Neither": "#8b5cf6",
    "False_Misconception": "#ef4444",
}


# ============================================================
# PLACEHOLDER MODEL (Replace with real model integration)
# ============================================================
class MockPredictor:
    """
    Placeholder predictor — returns mock predictions.
    Replace this with your actual model loading and inference.
    """

    def predict(self, question: str, answer: str, explanation: str):
        """
        TODO: Replace with actual model inference:
        1. Load DeBERTa + BGE models
        2. Build input text
        3. Run ensemble prediction
        4. Apply family filter + temperature scaling
        5. Return top-3 predictions
        """
        # --- Mock predictions for demo ---
        # In production, replace everything below with real model code
        import numpy as np

        # Simulate model output
        scores = np.random.dirichlet(np.ones(len(CLASSES)) * 0.3)

        # Apply a simple heuristic for demo purposes
        # (real model would use the actual input)
        if any(word in explanation.lower() for word in ["correct", "right", "yes"]):
            # Boost True_* classes
            for i, c in enumerate(CLASSES):
                if c.startswith("True"):
                    scores[i] *= 3.0
        else:
            for i, c in enumerate(CLASSES):
                if c.startswith("False"):
                    scores[i] *= 2.5

        # Normalize
        scores = scores / scores.sum()

        # Get top-3
        top_indices = np.argsort(scores)[::-1][:3]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            label = CLASSES[idx]
            parts = label.split(":", 1)
            cat = parts[0]
            miscon = parts[1] if len(parts) > 1 else "NA"

            results.append({
                "rank": rank,
                "label": label,
                "category": cat,
                "misconception": miscon,
                "confidence": float(scores[idx]),
                "category_description": CATEGORY_DESCRIPTIONS.get(cat, cat),
                "misconception_description": MISCON_DESCRIPTIONS.get(miscon, miscon),
                "category_color": CATEGORY_COLORS.get(cat, "#6b7280"),
            })

        return results

    def generate_explanation(self, question: str, answer: str,
                             explanation: str, prediction: dict):
        """
        TODO: Replace with actual LLM-based explanation (Qwen2.5-7B / Flan-T5)
        """
        cat = prediction["category"]
        miscon = prediction["misconception"]
        miscon_desc = prediction["misconception_description"]

        if miscon == "NA":
            return (
                f"The student's response was classified as **{cat}**. "
                f"No specific misconception was detected in their reasoning. "
                f"The explanation provided {'appears adequate' if 'Correct' in cat else 'lacks clear mathematical reasoning'}."
            )

        return (
            f"**Detected Misconception: {miscon}**\n\n"
            f"{miscon_desc}.\n\n"
            f"**Analysis:** The student's explanation suggests they may be "
            f"applying {miscon.lower().replace('_', ' ')} reasoning. "
            f"This is a common pattern where students "
            f"{'arrive at the correct answer through flawed logic' if cat.startswith('True') else 'reach an incorrect answer'} "
            f"because of this specific conceptual gap.\n\n"
            f"**Suggested Intervention:** Review the underlying concept and "
            f"provide worked examples that specifically address this type of error."
        )


predictor = MockPredictor()


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class PredictionRequest(BaseModel):
    question: str
    answer: str
    explanation: str


class FeedbackRequest(BaseModel):
    prediction_id: str
    helpful: bool
    comment: Optional[str] = None


# ============================================================
# API ROUTES
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/predict")
async def predict(req: PredictionRequest):
    """Run the ensemble model and return top-3 predictions."""
    predictions = predictor.predict(req.question, req.answer, req.explanation)

    # Generate explanation for top prediction
    top_explanation = predictor.generate_explanation(
        req.question, req.answer, req.explanation, predictions[0]
    )

    return JSONResponse({
        "success": True,
        "predictions": predictions,
        "explanation": top_explanation,
        "input_summary": {
            "question_preview": req.question[:200],
            "answer": req.answer,
            "explanation_preview": req.explanation[:200],
        },
    })


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """Collect user feedback on predictions."""
    # TODO: Store feedback in a database
    return JSONResponse({"success": True, "message": "Thank you for your feedback!"})


@app.get("/api/stats")
async def stats():
    """Return model and dataset statistics."""
    return JSONResponse({
        "total_classes": len(CLASSES),
        "categories": list(CATEGORY_DESCRIPTIONS.keys()),
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "misconception_count": len(MISCON_DESCRIPTIONS),
        "model_info": {
            "primary": "DeBERTa-v3-large (Classification)",
            "secondary": "BGE-Large-en-v1.5 (Retrieval)",
            "explainer": "Qwen2.5-7B-Instruct",
            "ensemble": "Weighted probability fusion",
            "metric": "MAP@3",
        },
        "techniques": [
            "Focal Loss (γ=2)",
            "Adversarial Weight Perturbation",
            "R-Drop Regularization",
            "Exponential Moving Average",
            "Layerwise LR Decay",
            "Temperature Scaling",
            "Family Filter",
            "GroupKFold Cross-Validation",
        ],
    })


@app.get("/api/examples")
async def examples():
    """Return sample questions for the demo."""
    return JSONResponse({
        "examples": [
            {
                "question": "What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]",
                "answer": "1/3",
                "explanation": "1/3 because 6 over 9 is 2 thirds and 1 third is not shaded.",
            },
            {
                "question": "Simplify the expression: 3x + 2x",
                "answer": "5x",
                "explanation": "I added 3 and 2 to get 5 and kept the x",
            },
            {
                "question": "What is 0.3 + 0.25?",
                "answer": "0.28",
                "explanation": "I added the 3 and the 25 to get 28 and put a decimal point",
            },
            {
                "question": "A rectangle has a length of 8cm and a width of 3cm. What is the area?",
                "answer": "22cm²",
                "explanation": "I added 8 + 3 + 8 + 3 = 22",
            },
            {
                "question": "What is the next term in the sequence: 2, 6, 18, 54, ...?",
                "answer": "162",
                "explanation": "You multiply by 3 each time so 54 times 3 is 162",
            },
        ]
    })


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
