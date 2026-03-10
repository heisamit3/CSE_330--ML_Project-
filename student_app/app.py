"""
MAP Student Portal — Math Practice & Feedback
A student-facing web app where students answer math questions,
provide explanations, and receive AI-powered feedback.
"""

import os
import csv
import random
from pathlib import Path
from typing import Optional
from collections import defaultdict

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
    title="MAP Student Portal — Math Practice",
    description="Practice math and get AI-powered feedback",
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
# LOAD QUESTIONS FROM DATASET
# ============================================================
DS_PATH = BASE_DIR.parent / "DS" / "train.csv"

# Stores: question_id -> { text, correct_answer, correct_explanations[], all_entries[] }
QUESTIONS_DB = {}


def load_questions():
    """Load and group questions from the training CSV."""
    if not DS_PATH.exists():
        print(f"[WARN] Dataset not found at {DS_PATH}, using built-in examples.")
        return _load_builtin_questions()

    raw = defaultdict(list)
    with open(DS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["QuestionId"]
            raw[qid].append(row)

    for qid, rows in raw.items():
        # Collect correct explanations (True_Correct with NA misconception)
        correct_explanations = []
        for r in rows:
            if r["Category"] == "True_Correct" and r["Misconception"] == "NA":
                expl = r["StudentExplanation"].strip()
                if len(expl) > 20:  # Only good-quality explanations
                    correct_explanations.append(expl)

        # Deduplicate and pick best explanations
        seen = set()
        unique_explanations = []
        for e in correct_explanations:
            lower = e.lower().strip()
            if lower not in seen:
                seen.add(lower)
                unique_explanations.append(e)

        # Sort by length (longer = more detailed) and take top 5
        unique_explanations.sort(key=len, reverse=True)
        best_explanations = unique_explanations[:5]

        QUESTIONS_DB[qid] = {
            "id": qid,
            "text": rows[0]["QuestionText"],
            "correct_answer": rows[0]["MC_Answer"],
            "correct_explanations": best_explanations,
            "entries": rows,
        }

    print(f"[INFO] Loaded {len(QUESTIONS_DB)} unique questions from dataset.")


def _load_builtin_questions():
    """Fallback questions if CSV is not found."""
    builtin = [
        {
            "id": "31772",
            "text": "What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]",
            "correct_answer": "\\( \\frac{1}{3} \\)",
            "correct_explanations": [
                "3 out of 9 triangles are not shaded. 3/9 simplifies to 1/3 because both 3 and 9 are divisible by 3.",
                "6 out of 9 are shaded, so 3 out of 9 are not shaded. 3/9 = 1/3 in simplest form.",
            ],
            "entries": [],
        },
        {
            "id": "EX001",
            "text": "Simplify the expression: 3x + 2x",
            "correct_answer": "5x",
            "correct_explanations": [
                "3x and 2x are like terms because they both have the variable x. When you add like terms, you add the coefficients: 3 + 2 = 5, so you get 5x.",
            ],
            "entries": [],
        },
        {
            "id": "EX002",
            "text": "What is 0.3 + 0.25?",
            "correct_answer": "0.55",
            "correct_explanations": [
                "Line up the decimal points: 0.30 + 0.25 = 0.55. You need to write 0.3 as 0.30 so both numbers have the same number of decimal places.",
            ],
            "entries": [],
        },
        {
            "id": "EX003",
            "text": "A rectangle has a length of 8cm and a width of 3cm. What is the area?",
            "correct_answer": "24cm²",
            "correct_explanations": [
                "Area of a rectangle = length × width = 8 × 3 = 24cm². The area is found by multiplying, not adding the sides.",
            ],
            "entries": [],
        },
        {
            "id": "EX004",
            "text": "What is the next term in the sequence: 2, 6, 18, 54, ...?",
            "correct_answer": "162",
            "correct_explanations": [
                "This is a geometric sequence where each term is multiplied by 3. So 54 × 3 = 162.",
            ],
            "entries": [],
        },
    ]
    for q in builtin:
        QUESTIONS_DB[q["id"]] = q
    print(f"[INFO] Loaded {len(QUESTIONS_DB)} built-in questions.")


# Load questions on startup
load_questions()


# ============================================================
# MISCONCEPTION KNOWLEDGE BASE
# ============================================================
MISCONCEPTION_FEEDBACK = {
    "NA": {
        "name": "No Misconception",
        "student_tip": "Your reasoning looks fine! Keep up the good work.",
        "what_went_wrong": None,
    },
    "Additive": {
        "name": "Additive Reasoning Error",
        "student_tip": "You used addition where multiplication or proportional reasoning was needed. When quantities are related proportionally (like scales, ratios, or percentages), you need to multiply or divide, not just add or subtract a fixed amount.",
        "what_went_wrong": "You applied an additive approach (adding/subtracting) to a problem that requires multiplicative thinking.",
    },
    "Subtraction": {
        "name": "Subtraction Error",
        "student_tip": "You subtracted when a different operation was needed. Re-read the question carefully and think about what operation the question is really asking for.",
        "what_went_wrong": "You used subtraction where a different mathematical operation was required.",
    },
    "Incomplete": {
        "name": "Incomplete Reasoning",
        "student_tip": "You were on the right track, but you didn't finish working through the problem completely. Make sure you carry out all the steps needed to reach the final answer.",
        "what_went_wrong": "Your reasoning had the right idea but was missing some important steps.",
    },
    "Inversion": {
        "name": "Inversion Error",
        "student_tip": "You flipped something that shouldn't have been flipped — maybe a fraction, a ratio, or the direction of an operation. Double-check which number goes on top/bottom or which value comes first.",
        "what_went_wrong": "You inverted a fraction, ratio, or operation when you shouldn't have.",
    },
    "Irrelevant": {
        "name": "Irrelevant Reasoning",
        "student_tip": "Your explanation doesn't connect to the math concept being tested. Re-read the question and think about which mathematical idea it's testing.",
        "what_went_wrong": "Your reasoning was about something different from what the question was asking.",
    },
    "Mult": {
        "name": "Multiplication Error",
        "student_tip": "There was a mistake in your multiplication. Try breaking the multiplication into smaller steps or using a different method to double-check your work.",
        "what_went_wrong": "You made an error when multiplying numbers.",
    },
    "Wrong_fraction": {
        "name": "Wrong Fraction",
        "student_tip": "The fraction you formed or simplified isn't correct. Remember: the numerator is the 'part' and the denominator is the 'whole'. When simplifying, divide both top and bottom by the same number.",
        "what_went_wrong": "You created or simplified a fraction incorrectly.",
    },
    "Wrong_Fraction": {
        "name": "Wrong Fraction",
        "student_tip": "The fraction you created from the given information isn't right. Think carefully about what number goes on top (numerator) and what goes on the bottom (denominator).",
        "what_went_wrong": "You constructed the wrong fraction from the given information.",
    },
    "Denominator-only_change": {
        "name": "Denominator-Only Change",
        "student_tip": "When you change a fraction (multiply, divide, simplify), you must do the same thing to BOTH the numerator and the denominator. You only changed the bottom number.",
        "what_went_wrong": "You only changed the denominator when both numerator and denominator should have been changed together.",
    },
    "Num-only_change": {
        "name": "Numerator-Only Change",
        "student_tip": "When you change a fraction, you must do the same thing to both the top and bottom numbers. You only changed the numerator.",
        "what_went_wrong": "You only changed the numerator when both parts of the fraction should have been changed.",
    },
    "Adding_across": {
        "name": "Adding Fractions Incorrectly",
        "student_tip": "When adding fractions, you can't just add the numerators and denominators separately. You need a common denominator first, then only add the numerators.",
        "what_went_wrong": "You added numerators and denominators separately (e.g., 1/2 + 1/3 = 2/5), which is incorrect.",
    },
    "Whole_numbers_larger": {
        "name": "Decimal Place Value Confusion",
        "student_tip": "More digits after the decimal point doesn't mean a bigger number. For example, 0.5 is bigger than 0.45, even though 45 > 5. Compare digit by digit from left to right.",
        "what_went_wrong": "You thought a decimal with more digits is always larger.",
    },
    "Shorter_is_bigger": {
        "name": "Shorter Decimal Confusion",
        "student_tip": "A shorter decimal isn't always bigger. For example, 0.3 is NOT bigger than 0.35. Compare decimals by adding zeros to make them the same length: 0.30 < 0.35.",
        "what_went_wrong": "You assumed that a decimal with fewer digits is always larger.",
    },
    "Tacking": {
        "name": "Tacking On Digits",
        "student_tip": "When working with decimals, you can't just stick digits together. Each position after the decimal point has a specific place value (tenths, hundredths, etc.).",
        "what_went_wrong": "You combined digits without considering place value.",
    },
    "Duplication": {
        "name": "Duplication Error",
        "student_tip": "You repeated or duplicated a number or operation when you shouldn't have. Carefully track each step to make sure you use each value only once.",
        "what_went_wrong": "You incorrectly repeated or duplicated numbers or operations.",
    },
    "Positive": {
        "name": "Positive Number Bias",
        "student_tip": "Remember that negative numbers follow different rules! When you multiply or divide two negatives, you get a positive. When you add a negative, it's like subtracting.",
        "what_went_wrong": "You treated negative numbers as if they were positive.",
    },
    "Negative": {
        "name": "Negative Number Error",
        "student_tip": "Be careful with negative signs! Remember the rules: negative × negative = positive, negative × positive = negative. Also watch signs when adding and subtracting.",
        "what_went_wrong": "You made an error handling negative numbers or signs.",
    },
    "Wrong_term": {
        "name": "Wrong Term in Sequence",
        "student_tip": "In sequences, make sure you're identifying the right term. The 'first term' is where the sequence starts, and the 'common difference' or 'common ratio' is how you get from one term to the next.",
        "what_went_wrong": "You identified or used the wrong term in a sequence.",
    },
    "Firstterm": {
        "name": "First Term Confusion",
        "student_tip": "Don't confuse the first term of a sequence with the common difference. The first term is the starting value, and the common difference is how much you add each time.",
        "what_went_wrong": "You confused the first term with the common difference in the sequence.",
    },
    "WNB": {
        "name": "Wrong Number Bond",
        "student_tip": "A number bond shows how a number can be broken into parts. Double-check that your parts add up correctly to the whole number.",
        "what_went_wrong": "You used incorrect number bonds.",
    },
    "SwapDividend": {
        "name": "Swapped Division",
        "student_tip": "In division, the order matters! 'A divided by B' means A ÷ B, not B ÷ A. The number being split up (dividend) comes first.",
        "what_went_wrong": "You swapped the dividend and divisor — you divided in the wrong direction.",
    },
    "Scale": {
        "name": "Scale Factor Error",
        "student_tip": "When scaling, all measurements must be multiplied by the same scale factor. Make sure you're applying the factor consistently to all dimensions.",
        "what_went_wrong": "You misunderstood how scale factors work.",
    },
    "Not_variable": {
        "name": "Variable Misunderstanding",
        "student_tip": "A variable (like x or n) represents any number, not a specific one. Don't substitute a particular value for it unless the question tells you to.",
        "what_went_wrong": "You treated a variable as a specific number instead of a general one.",
    },
    "Adding_terms": {
        "name": "Adding Unlike Terms",
        "student_tip": "You can only add terms that have the same variable part. For example, 3x + 2x = 5x, but 3x + 2y cannot be simplified further because x and y are different.",
        "what_went_wrong": "You combined algebraic terms that aren't like terms.",
    },
    "CoeffIsExp": {
        "name": "Coefficient vs. Exponent Confusion",
        "student_tip": "The coefficient (number in front) and the exponent (power) are different things. 3x means '3 times x', while x³ means 'x times x times x'.",
        "what_went_wrong": "You confused the coefficient with the exponent.",
    },
    "Multiplying_by_4": {
        "name": "Wrong Multiplication Factor",
        "student_tip": "Re-read the question carefully to find the correct number to multiply by. The operation you chose was right, but the number wasn't.",
        "what_went_wrong": "You multiplied by the wrong number.",
    },
    "FlipChange": {
        "name": "Direction of Change Error",
        "student_tip": "Check whether a value should increase or decrease. Sometimes students flip the direction — increasing when they should decrease or vice versa.",
        "what_went_wrong": "You got the direction of change backwards.",
    },
    "Expand_single_bracket": {
        "name": "Bracket Expansion Error",
        "student_tip": "When expanding brackets like 3(x + 2), multiply EVERY term inside the bracket by the number outside: 3 × x + 3 × 2 = 3x + 6.",
        "what_went_wrong": "You made an error when expanding a bracket expression.",
    },
    "Unknowable": {
        "name": "Premature Conclusion",
        "student_tip": "Don't give up too quickly! The problem CAN be solved with the information given. Look again at what you know and try a different approach.",
        "what_went_wrong": "You said the problem couldn't be solved, but it actually can be.",
    },
}

CATEGORY_FEEDBACK = {
    "True_Correct": {
        "icon": "check-circle",
        "color": "#10b981",
        "headline": "Correct! Great job!",
        "message": "Your answer is right and your reasoning is solid.",
    },
    "True_Neither": {
        "icon": "check-circle",
        "color": "#f59e0b",
        "headline": "Correct answer, but your explanation needs work",
        "message": "You got the right answer, but your explanation isn't clear enough. Try to explain your steps more precisely so someone else could follow your thinking.",
    },
    "True_Misconception": {
        "icon": "exclamation-triangle",
        "color": "#f97316",
        "headline": "Correct answer, but watch your reasoning!",
        "message": "You got the right answer, but your explanation reveals a misunderstanding. Getting the right answer with wrong reasoning can cause problems with harder questions later.",
    },
    "False_Correct": {
        "icon": "times-circle",
        "color": "#3b82f6",
        "headline": "Incorrect answer, but your method was close",
        "message": "Your reasoning approach was on the right track, but something went wrong in execution. Check your calculations carefully.",
    },
    "False_Neither": {
        "icon": "times-circle",
        "color": "#8b5cf6",
        "headline": "Incorrect — let's work through this together",
        "message": "Your answer is wrong and the explanation isn't clear. Don't worry — let's look at the correct approach step by step.",
    },
    "False_Misconception": {
        "icon": "times-circle",
        "color": "#ef4444",
        "headline": "Incorrect — we found a misconception",
        "message": "Your answer is wrong because of a specific misunderstanding. The good news: once you understand where your thinking went wrong, you'll be able to fix it!",
    },
}


# ============================================================
# LABEL DEFINITIONS
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


# ============================================================
# PREDICTOR (Replace with real model)
# ============================================================
class MockPredictor:
    """
    Placeholder predictor — returns mock predictions.
    Replace with actual model loading and inference in production.
    """

    def predict(self, question: str, answer: str, explanation: str):
        import numpy as np

        scores = np.random.dirichlet(np.ones(len(CLASSES)) * 0.3)

        if any(word in explanation.lower() for word in ["correct", "right", "yes", "because", "divided", "multiplied"]):
            for i, c in enumerate(CLASSES):
                if c.startswith("True"):
                    scores[i] *= 3.0
        else:
            for i, c in enumerate(CLASSES):
                if c.startswith("False"):
                    scores[i] *= 2.5

        scores = scores / scores.sum()
        top_idx = np.argsort(scores)[::-1][0]
        label = CLASSES[top_idx]
        parts = label.split(":", 1)
        cat = parts[0]
        miscon = parts[1] if len(parts) > 1 else "NA"

        return {
            "label": label,
            "category": cat,
            "misconception": miscon,
            "confidence": float(scores[top_idx]),
        }


predictor = MockPredictor()


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================
class SubmitRequest(BaseModel):
    question_id: str
    student_answer: str
    student_explanation: str


class FeedbackRequest(BaseModel):
    question_id: str
    rating: int  # 1-5
    comment: Optional[str] = None


# ============================================================
# ROUTES
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/question")
async def get_question(question_id: Optional[str] = None):
    """Get a random question (or specific one by ID)."""
    if not QUESTIONS_DB:
        return JSONResponse({"error": "No questions available"}, status_code=500)

    if question_id and question_id in QUESTIONS_DB:
        q = QUESTIONS_DB[question_id]
    else:
        q = random.choice(list(QUESTIONS_DB.values()))

    return JSONResponse({
        "id": q["id"],
        "text": q["text"],
        "has_image": "[Image:" in q["text"] or "[image:" in q["text"].lower(),
    })


@app.get("/api/questions/list")
async def list_questions():
    """List all available question IDs and previews."""
    questions = []
    for qid, q in QUESTIONS_DB.items():
        preview = q["text"][:100] + ("..." if len(q["text"]) > 100 else "")
        questions.append({"id": qid, "preview": preview})
    return JSONResponse({"questions": questions})


@app.post("/api/submit")
async def submit_answer(req: SubmitRequest):
    """Student submits their answer and explanation; receives AI feedback."""
    qid = req.question_id
    if qid not in QUESTIONS_DB:
        return JSONResponse({"error": "Question not found"}, status_code=404)

    q = QUESTIONS_DB[qid]

    # Run the model prediction
    prediction = predictor.predict(q["text"], req.student_answer, req.student_explanation)

    category = prediction["category"]
    misconception = prediction["misconception"]

    # Build the feedback response
    cat_feedback = CATEGORY_FEEDBACK.get(category, CATEGORY_FEEDBACK["False_Neither"])
    miscon_feedback = MISCONCEPTION_FEEDBACK.get(misconception, MISCONCEPTION_FEEDBACK["NA"])

    # Pick a correct explanation to show
    correct_explanation = ""
    if q["correct_explanations"]:
        correct_explanation = random.choice(q["correct_explanations"])

    # Determine if the student's answer is correct
    is_correct = category.startswith("True")

    feedback = {
        "is_correct": is_correct,
        "correct_answer": q["correct_answer"],
        "student_answer": req.student_answer,

        # Category feedback (overall result)
        "result_icon": cat_feedback["icon"],
        "result_color": cat_feedback["color"],
        "result_headline": cat_feedback["headline"],
        "result_message": cat_feedback["message"],

        # Misconception details (if any)
        "has_misconception": misconception != "NA" and "Misconception" in category,
        "misconception_name": miscon_feedback["name"],
        "misconception_tip": miscon_feedback["student_tip"],
        "what_went_wrong": miscon_feedback.get("what_went_wrong"),

        # Correct solution
        "correct_explanation": correct_explanation,

        # Confidence (for internal use, shown as a strength indicator)
        "confidence": prediction["confidence"],
    }

    return JSONResponse({"success": True, "feedback": feedback})


@app.post("/api/rate")
async def rate_feedback(req: FeedbackRequest):
    """Collect student ratings on the feedback quality."""
    # TODO: Store in database
    return JSONResponse({"success": True, "message": "Thanks for your rating!"})


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
