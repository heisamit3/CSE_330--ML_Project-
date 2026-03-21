"""
MAP Student Portal - MCQ practice with explanation-aware feedback.
"""

import csv
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


app = FastAPI(
    title="MAP Student Portal",
    description="Student-facing math practice with MCQ choices and explanation feedback",
    version="2.0.0",
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

DS_PATH = BASE_DIR.parent / "DS" / "train.csv"
QUESTIONS_DB = {}

MAX_EXAMPLES_PER_CATEGORY = 60
SIMILARITY_THRESHOLD = 0.18


QUESTION_GUIDES = {
    "31772": {
        "solution": "There are 9 equal small triangles in total and 6 are shaded, so 3 are not shaded. That gives 3/9, which simplifies to 1/3.",
    },
    "31774": {
        "solution": "Dividing by 6 is the same as multiplying by 1/6. So 1/2 div 6 = 1/2 x 1/6 = 1/12.",
    },
    "31777": {
        "solution": "First find one fifth of 120: 120 div 5 = 24. Then multiply by 3 because the question asks for 3/5, so 24 x 3 = 72.",
    },
    "31778": {
        "solution": "Simplify 9/15 to 3/5. Since A/10 = 3/5, multiply 10 by 3/5 to get A = 6.",
    },
    "32829": {
        "solution": "2y = 24 means 2 times y equals 24. Divide both sides by 2, so y = 12.",
    },
    "32833": {
        "solution": "Multiply 2/3 by 5: 2 x 5 = 10, so the answer is 10/3. As a mixed number, 10/3 = 3 1/3.",
    },
    "32835": {
        "solution": "Compare the decimals place by place. 6.2 is 6.2000, which is greater than 6.079, 6.0001, and 6.",
    },
    "33471": {
        "solution": "If 3/8 of 24 balls are yellow, then yellow balls = 24 x 3/8 = 9. The rest are green, so 24 - 9 = 15.",
    },
    "33472": {
        "solution": "Use a common denominator of 15. 1/3 = 5/15 and 2/5 = 6/15, so 5/15 + 6/15 = 11/15.",
    },
    "33474": {
        "solution": "Robert eats one third of the two-thirds piece, so you multiply the fractions: 1/3 x 2/3.",
    },
    "76870": {
        "solution": "In a regular polygon, exterior angle = 180 - 144 = 36 degrees. The number of sides is 360 div 36 = 10.",
    },
    "89443": {
        "solution": "Subtracting a negative is the same as adding the positive. So (-8) - (-5) = -8 + 5 = -3.",
    },
    "91695": {
        "solution": "The pattern goes up by 4 each time: 6, 10, 14, 18, 22, 26. So Pattern 6 has 26 dots.",
    },
    "104665": {
        "solution": "Three people taking 192 hours means the job needs 3 x 192 = 576 person-hours. With 12 people, 576 div 12 = 48 hours.",
    },
    "109465": {
        "solution": "A probability of 0.9 means a 90 percent chance. That is very likely, but it is not certain because there is still a 10 percent chance it will not happen.",
    },
    "EX001": {
        "solution": "3x and 2x are like terms because they both have x. Add the coefficients: 3 + 2 = 5, so the answer is 5x.",
    },
    "EX002": {
        "solution": "Write 0.3 as 0.30 so the decimal places line up. Then add: 0.30 + 0.25 = 0.55.",
    },
    "EX003": {
        "solution": "Area of a rectangle = length x width. So 8 x 3 = 24 cm^2.",
    },
    "EX004": {
        "solution": "Each term is multiplied by 3: 2, 6, 18, 54. So the next term is 54 x 3 = 162.",
    },
}


MISCONCEPTION_FEEDBACK = {
    "NA": {
        "name": "No specific misconception",
        "student_tip": "Your explanation does not point to a single misconception.",
        "what_went_wrong": None,
    },
    "Additive": {
        "name": "Additive reasoning error",
        "student_tip": "You used addition where multiplicative or proportional reasoning was needed.",
        "what_went_wrong": "You treated a multiplicative relationship like it only needed adding or subtracting.",
    },
    "Subtraction": {
        "name": "Subtraction error",
        "student_tip": "Check whether the question really calls for subtraction before using it.",
        "what_went_wrong": "Your explanation suggests you subtracted when another operation was needed.",
    },
    "Incomplete": {
        "name": "Incomplete reasoning",
        "student_tip": "You started with the right idea, but you did not carry it through to the full answer.",
        "what_went_wrong": "Your explanation stops too early or skips an essential step.",
    },
    "Inversion": {
        "name": "Inversion error",
        "student_tip": "Be careful not to flip a fraction, ratio, or operation unless the method really requires it.",
        "what_went_wrong": "Part of your reasoning reverses something that should have stayed in the original order.",
    },
    "Irrelevant": {
        "name": "Irrelevant reasoning",
        "student_tip": "Focus your explanation on the specific math idea the question is testing.",
        "what_went_wrong": "Your explanation does not connect tightly to the question being asked.",
    },
    "Mult": {
        "name": "Multiplication error",
        "student_tip": "Recheck the multiplication step and break it into smaller pieces if needed.",
        "what_went_wrong": "A multiplication step in the reasoning appears incorrect.",
    },
    "Wrong_fraction": {
        "name": "Wrong fraction",
        "student_tip": "Think carefully about the part and the whole, and simplify only after forming the correct fraction.",
        "what_went_wrong": "The fraction in your answer or explanation was set up incorrectly.",
    },
    "Wrong_Fraction": {
        "name": "Wrong fraction",
        "student_tip": "Make sure the numerator and denominator match the meaning of the question.",
        "what_went_wrong": "The fraction you constructed does not represent the situation correctly.",
    },
    "Denominator-only_change": {
        "name": "Denominator-only change",
        "student_tip": "When you scale or simplify a fraction, both parts have to change together.",
        "what_went_wrong": "You changed only the denominator when the numerator should have changed too.",
    },
    "Num-only_change": {
        "name": "Numerator-only change",
        "student_tip": "When you change a fraction, the numerator and denominator must stay in balance.",
        "what_went_wrong": "You changed only the numerator when both parts should have changed together.",
    },
    "Adding_across": {
        "name": "Adding fractions across",
        "student_tip": "When adding fractions, use a common denominator and only then add the numerators.",
        "what_went_wrong": "Your explanation suggests adding the top numbers and bottom numbers directly.",
    },
    "Whole_numbers_larger": {
        "name": "Decimal place value confusion",
        "student_tip": "A whole number is not always greater than a decimal, and more digits do not always mean a bigger value.",
        "what_went_wrong": "You compared decimals as if the number with fewer decimal places had to be larger or smaller.",
    },
    "Shorter_is_bigger": {
        "name": "Shorter decimal confusion",
        "student_tip": "Line decimals up by place value before comparing them.",
        "what_went_wrong": "You treated a shorter decimal as automatically larger.",
    },
    "Tacking": {
        "name": "Tacking digits together",
        "student_tip": "Each decimal place has meaning. Do not join digits together without place value.",
        "what_went_wrong": "Your explanation treats decimal digits like they can simply be attached together.",
    },
    "Duplication": {
        "name": "Duplication error",
        "student_tip": "Track each number once and check that you have not repeated a step.",
        "what_went_wrong": "A number or step seems to have been repeated incorrectly.",
    },
    "Positive": {
        "name": "Positive number bias",
        "student_tip": "Watch the signs carefully, especially with negative numbers.",
        "what_went_wrong": "Your explanation treats negative numbers as though they were positive.",
    },
    "Negative": {
        "name": "Negative number error",
        "student_tip": "Check the sign rules again for adding, subtracting, multiplying, or dividing negatives.",
        "what_went_wrong": "The sign handling in your reasoning is incorrect.",
    },
    "Wrong_term": {
        "name": "Wrong term in sequence",
        "student_tip": "Build the pattern carefully so you land on the exact term the question asks for.",
        "what_went_wrong": "Your explanation appears to stop on the wrong term in the sequence.",
    },
    "Firstterm": {
        "name": "First term confusion",
        "student_tip": "Keep the starting term separate from the amount the pattern changes by.",
        "what_went_wrong": "Your reasoning mixes up the first term and the rule of the sequence.",
    },
    "WNB": {
        "name": "Wrong number bond",
        "student_tip": "Check that the parts you found really combine to the whole in the question.",
        "what_went_wrong": "Your explanation splits the quantity into incorrect parts.",
    },
    "SwapDividend": {
        "name": "Swapped division order",
        "student_tip": "In division, order matters. Keep the dividend and divisor in the right places.",
        "what_went_wrong": "Your reasoning suggests the numbers in the division were reversed.",
    },
    "Scale": {
        "name": "Scale factor error",
        "student_tip": "Use the same scale factor consistently across the quantities involved.",
        "what_went_wrong": "Your explanation applies scaling inconsistently.",
    },
    "Not_variable": {
        "name": "Variable misunderstanding",
        "student_tip": "A variable stands for a value you need to solve for, not a label to ignore.",
        "what_went_wrong": "Your reasoning does not treat the variable correctly.",
    },
    "Adding_terms": {
        "name": "Adding unlike terms",
        "student_tip": "Only like terms can be combined in algebra.",
        "what_went_wrong": "Your explanation combines algebraic terms that should stay separate.",
    },
    "CoeffIsExp": {
        "name": "Coefficient and exponent confusion",
        "student_tip": "A coefficient multiplies the variable, while an exponent tells you repeated multiplication.",
        "what_went_wrong": "Your reasoning mixes up the role of a coefficient and an exponent.",
    },
    "Multiplying_by_4": {
        "name": "Wrong multiplication factor",
        "student_tip": "The idea to multiply may be right, but the factor has to match the question exactly.",
        "what_went_wrong": "You used the wrong factor in the multiplication step.",
    },
    "FlipChange": {
        "name": "Direction of change error",
        "student_tip": "Check whether the quantity should increase or decrease before doing the calculation.",
        "what_went_wrong": "Your explanation reverses the direction of change.",
    },
    "Expand_single_bracket": {
        "name": "Bracket expansion error",
        "student_tip": "When expanding brackets, multiply every term inside the bracket.",
        "what_went_wrong": "Your explanation expands the bracket incorrectly.",
    },
    "Unknowable": {
        "name": "Premature conclusion",
        "student_tip": "The question can be solved. Try identifying the key relationship and working step by step.",
        "what_went_wrong": "Your explanation says the answer cannot be found when enough information is actually given.",
    },
}


CATEGORY_FEEDBACK = {
    "True_Correct": {
        "color": "#10b981",
        "headline": "Correct choice and solid reasoning",
        "message": "You picked the correct option and your explanation matches a sound way to solve it.",
    },
    "True_Neither": {
        "color": "#f59e0b",
        "headline": "Correct choice, but explain it more clearly",
        "message": "You selected the right answer, but your explanation needs a clearer chain of steps.",
    },
    "True_Misconception": {
        "color": "#f97316",
        "headline": "Correct choice, but the reasoning needs attention",
        "message": "The final choice is right, but the explanation suggests a misconception that could cause trouble later.",
    },
    "False_Correct": {
        "color": "#3b82f6",
        "headline": "Not quite, but your method is close",
        "message": "You chose the wrong option, yet your explanation shows part of the correct method.",
    },
    "False_Neither": {
        "color": "#8b5cf6",
        "headline": "Not quite - let's tighten the explanation",
        "message": "Your selected answer is not correct, and the explanation does not clearly show a complete method yet.",
    },
    "False_Misconception": {
        "color": "#ef4444",
        "headline": "Incorrect answer - misconception detected",
        "message": "Your choice and explanation point to a specific misunderstanding that we can fix.",
    },
}


REASONING_SUMMARY = {
    "Correct": {
        "label": "Reasoning: strong",
        "message": "Your explanation lines up with a valid method.",
    },
    "Neither": {
        "label": "Reasoning: unclear",
        "message": "Your explanation needs a clearer sequence of steps.",
    },
    "Misconception": {
        "label": "Reasoning: misconception spotted",
        "message": "Your explanation points to a specific misunderstanding.",
    },
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_answer(text: str) -> str:
    cleaned = normalize_space(text)
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("\\", "")
    return cleaned.lower()


def tokenize_explanation(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_space(text).lower()))


def reasoning_band(category: str) -> str:
    return category.split("_", 1)[1] if "_" in category else "Neither"


def explanation_score(submitted_text: str, submitted_tokens: set[str], example: dict) -> float:
    if not submitted_text or not example["normalized"]:
        return 0.0

    ratio = SequenceMatcher(None, submitted_text, example["normalized"]).ratio()
    union = submitted_tokens | example["tokens"]
    jaccard = (len(submitted_tokens & example["tokens"]) / len(union)) if union else 0.0
    return (0.65 * ratio) + (0.35 * jaccard)


def infer_correct_answer(answer_stats: dict) -> str:
    ranked = []
    for answer, category_counts in answer_stats.items():
        true_count = sum(
            count for category, count in category_counts.items() if category.startswith("True")
        )
        total_count = sum(category_counts.values())
        ranked.append((true_count, total_count, answer))

    ranked.sort(reverse=True)
    return ranked[0][2]


def build_solution_text(question: dict) -> str:
    guide = QUESTION_GUIDES.get(question["id"], {})
    if guide.get("solution"):
        return guide["solution"]

    if question["correct_explanations"]:
        return question["correct_explanations"][0]

    return f"The correct answer is {question['correct_answer']}."


def build_builtin_question(
    question_id: str,
    text: str,
    options: list[str],
    correct_answer: str,
    correct_explanations: list[str],
) -> dict:
    profiles = {}
    for option in options:
        profiles[option] = {
            "examples": [],
            "default_reasoning_band": "Correct" if normalize_answer(option) == normalize_answer(correct_answer) else "Neither",
            "default_misconception": "NA",
        }

    return {
        "id": question_id,
        "text": text,
        "options": options,
        "correct_answer": correct_answer,
        "correct_explanations": correct_explanations,
        "answer_profiles": profiles,
    }


def build_question_record(question_id: str, rows: list[dict]) -> dict:
    answer_stats = defaultdict(Counter)
    answer_examples = defaultdict(list)
    answer_band_counts = defaultdict(Counter)
    answer_misconceptions = defaultdict(Counter)
    seen_examples = defaultdict(set)

    question_text = rows[0]["QuestionText"]

    for row in rows:
        answer = normalize_space(row["MC_Answer"])
        category = normalize_space(row["Category"])
        misconception = normalize_space(row["Misconception"]) or "NA"
        explanation = normalize_space(row["StudentExplanation"])

        answer_stats[answer][category] += 1
        answer_band_counts[answer][reasoning_band(category)] += 1
        if misconception != "NA" and reasoning_band(category) == "Misconception":
            answer_misconceptions[answer][misconception] += 1

        bucket_key = (answer, category)
        if explanation and explanation.lower() not in seen_examples[bucket_key]:
            if len(answer_examples[bucket_key]) < MAX_EXAMPLES_PER_CATEGORY:
                answer_examples[bucket_key].append(
                    {
                        "text": explanation,
                        "normalized": explanation.lower(),
                        "tokens": tokenize_explanation(explanation),
                        "category": category,
                        "reasoning_band": reasoning_band(category),
                        "misconception": misconception,
                    }
                )
            seen_examples[bucket_key].add(explanation.lower())

    correct_answer = infer_correct_answer(answer_stats)

    correct_explanations = []
    seen_correct = set()
    for row in rows:
        if normalize_answer(row["MC_Answer"]) != normalize_answer(correct_answer):
            continue
        if normalize_space(row["Category"]) != "True_Correct":
            continue

        explanation = normalize_space(row["StudentExplanation"])
        lowered = explanation.lower()
        if len(explanation) >= 18 and lowered not in seen_correct:
            correct_explanations.append(explanation)
            seen_correct.add(lowered)

    correct_explanations.sort(key=len, reverse=True)
    correct_explanations = correct_explanations[:8]

    options = list(answer_stats.keys())
    random.Random(f"options::{question_id}").shuffle(options)

    answer_profiles = {}
    for answer in options:
        examples = []
        for category in sorted(answer_stats[answer].keys()):
            examples.extend(answer_examples[(answer, category)])

        dominant_band = answer_band_counts[answer].most_common(1)[0][0]
        dominant_misconception = (
            answer_misconceptions[answer].most_common(1)[0][0]
            if answer_misconceptions[answer]
            else "NA"
        )

        answer_profiles[answer] = {
            "examples": examples,
            "default_reasoning_band": dominant_band,
            "default_misconception": dominant_misconception,
        }

    return {
        "id": question_id,
        "text": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "correct_explanations": correct_explanations,
        "answer_profiles": answer_profiles,
    }


def _load_builtin_questions() -> None:
    builtin_questions = [
        build_builtin_question(
            question_id="EX001",
            text="Simplify the expression: 3x + 2x",
            options=["5x", "6x", "x^5", "3x^2"],
            correct_answer="5x",
            correct_explanations=[
                "3x and 2x are like terms, so you add the coefficients: 3 + 2 = 5. The answer is 5x.",
            ],
        ),
        build_builtin_question(
            question_id="EX002",
            text="What is 0.3 + 0.25?",
            options=["0.55", "0.325", "0.28", "0.5"],
            correct_answer="0.55",
            correct_explanations=[
                "Line up the decimal points: 0.30 + 0.25 = 0.55.",
            ],
        ),
        build_builtin_question(
            question_id="EX003",
            text="A rectangle has a length of 8 cm and a width of 3 cm. What is the area?",
            options=["24 cm^2", "22 cm^2", "11 cm^2", "16 cm^2"],
            correct_answer="24 cm^2",
            correct_explanations=[
                "Area of a rectangle is length x width, so 8 x 3 = 24 cm^2.",
            ],
        ),
        build_builtin_question(
            question_id="EX004",
            text="What is the next term in the sequence: 2, 6, 18, 54, ...?",
            options=["162", "108", "57", "216"],
            correct_answer="162",
            correct_explanations=[
                "Each term is multiplied by 3, so 54 x 3 = 162.",
            ],
        ),
    ]

    for question in builtin_questions:
        QUESTIONS_DB[question["id"]] = question

    print(f"[INFO] Loaded {len(QUESTIONS_DB)} built-in questions.")


def load_questions() -> None:
    QUESTIONS_DB.clear()

    if not DS_PATH.exists():
        print(f"[WARN] Dataset not found at {DS_PATH}. Using built-in examples instead.")
        _load_builtin_questions()
        return

    grouped_rows = defaultdict(list)
    with open(DS_PATH, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped_rows[row["QuestionId"]].append(row)

    for question_id, rows in grouped_rows.items():
        QUESTIONS_DB[question_id] = build_question_record(question_id, rows)

    print(f"[INFO] Loaded {len(QUESTIONS_DB)} unique questions from dataset.")


def get_profile_for_answer(question: dict, selected_answer: str) -> Optional[dict]:
    if selected_answer in question["answer_profiles"]:
        return question["answer_profiles"][selected_answer]

    normalized_target = normalize_answer(selected_answer)
    for answer, profile in question["answer_profiles"].items():
        if normalize_answer(answer) == normalized_target:
            return profile

    return None


def fallback_reasoning(profile: dict, explanation: str, is_correct: bool) -> dict:
    explanation_text = normalize_space(explanation).lower()
    tokens = tokenize_explanation(explanation_text)
    reasoning_words = {
        "because",
        "divide",
        "divided",
        "multiply",
        "multiplied",
        "times",
        "add",
        "added",
        "subtract",
        "simplify",
        "common",
        "fraction",
        "decimal",
        "pattern",
        "probability",
        "angle",
        "equals",
    }
    has_reasoning_cue = any(word in explanation_text for word in reasoning_words) or any(
        character.isdigit() for character in explanation_text
    )

    if len(tokens) < 4 or len(explanation_text) < 18:
        return {
            "reasoning_band": "Neither",
            "misconception": "NA",
            "confidence": 0.32,
        }

    if is_correct:
        reasoning_band_name = "Correct" if has_reasoning_cue else "Neither"
        return {
            "reasoning_band": reasoning_band_name,
            "misconception": "NA",
            "confidence": 0.42 if reasoning_band_name == "Correct" else 0.34,
        }

    default_band = profile["default_reasoning_band"]
    default_misconception = profile["default_misconception"]

    if default_band == "Correct" and has_reasoning_cue:
        return {
            "reasoning_band": "Correct",
            "misconception": "NA",
            "confidence": 0.42,
        }

    if default_band == "Misconception" and default_misconception != "NA":
        return {
            "reasoning_band": "Misconception",
            "misconception": default_misconception,
            "confidence": 0.4,
        }

    return {
        "reasoning_band": "Neither",
        "misconception": "NA",
        "confidence": 0.34,
    }


def classify_reasoning(question: dict, selected_answer: str, explanation: str) -> dict:
    is_correct = normalize_answer(selected_answer) == normalize_answer(question["correct_answer"])
    profile = get_profile_for_answer(question, selected_answer)

    if profile is None:
        return {
            "reasoning_band": "Neither",
            "misconception": "NA",
            "confidence": 0.25,
        }

    normalized_explanation = normalize_space(explanation).lower()
    explanation_tokens = tokenize_explanation(normalized_explanation)

    heuristic_result = fallback_reasoning(profile, explanation, is_correct)

    if not normalized_explanation:
        return heuristic_result

    ranked_examples = []
    for example in profile["examples"]:
        score = explanation_score(normalized_explanation, explanation_tokens, example)
        if score >= SIMILARITY_THRESHOLD:
            ranked_examples.append((score, example))

    ranked_examples.sort(key=lambda item: item[0], reverse=True)
    ranked_examples = ranked_examples[:5]

    if ranked_examples:
        band_weights = Counter()
        misconception_weights = Counter()

        for score, example in ranked_examples:
            band_weights[example["reasoning_band"]] += score
            if example["misconception"] != "NA":
                misconception_weights[example["misconception"]] += score

        chosen_band = band_weights.most_common(1)[0][0]
        chosen_misconception = "NA"
        if chosen_band == "Misconception":
            chosen_misconception = (
                misconception_weights.most_common(1)[0][0]
                if misconception_weights
                else profile["default_misconception"]
            )

        result = {
            "reasoning_band": chosen_band,
            "misconception": chosen_misconception,
            "confidence": min(0.95, max(0.35, ranked_examples[0][0])),
        }
        if (
            result["reasoning_band"] == "Neither"
            and heuristic_result["reasoning_band"] != "Neither"
            and ranked_examples[0][0] < 0.7
        ):
            return heuristic_result

        return result

    return heuristic_result


def build_feedback(question: dict, selected_answer: str, explanation: str) -> dict:
    reasoning = classify_reasoning(question, selected_answer, explanation)
    is_correct = normalize_answer(selected_answer) == normalize_answer(question["correct_answer"])
    category = f"{'True' if is_correct else 'False'}_{reasoning['reasoning_band']}"

    category_feedback = CATEGORY_FEEDBACK[category]
    misconception_feedback = MISCONCEPTION_FEEDBACK.get(
        reasoning["misconception"], MISCONCEPTION_FEEDBACK["NA"]
    )
    reasoning_feedback = REASONING_SUMMARY[reasoning["reasoning_band"]]

    return {
        "is_correct": is_correct,
        "selected_answer": selected_answer,
        "correct_answer": question["correct_answer"],
        "result_color": category_feedback["color"],
        "result_headline": category_feedback["headline"],
        "result_message": category_feedback["message"],
        "reasoning_label": reasoning_feedback["label"],
        "reasoning_message": reasoning_feedback["message"],
        "answer_label": "Choice: correct" if is_correct else "Choice: incorrect",
        "has_misconception": reasoning["reasoning_band"] == "Misconception"
        and reasoning["misconception"] != "NA",
        "misconception_name": misconception_feedback["name"],
        "misconception_tip": misconception_feedback["student_tip"],
        "what_went_wrong": misconception_feedback["what_went_wrong"],
        "correct_explanation": build_solution_text(question),
        "confidence": reasoning["confidence"],
    }


load_questions()


class SubmitRequest(BaseModel):
    question_id: str
    selected_answer: Optional[str] = None
    student_answer: Optional[str] = None
    student_explanation: str


class FeedbackRequest(BaseModel):
    question_id: str
    rating: int
    comment: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "questions_loaded": len(QUESTIONS_DB)})


@app.get("/api/question")
async def get_question(question_id: Optional[str] = None):
    if not QUESTIONS_DB:
        return JSONResponse({"error": "No questions available."}, status_code=500)

    if question_id and question_id in QUESTIONS_DB:
        question = QUESTIONS_DB[question_id]
    else:
        question = random.choice(list(QUESTIONS_DB.values()))

    return JSONResponse(
        {
            "id": question["id"],
            "text": question["text"],
            "options": [
                {"id": chr(65 + index), "text": option}
                for index, option in enumerate(question["options"])
            ],
            "has_image": "[Image:" in question["text"] or "[image:" in question["text"].lower(),
        }
    )


@app.get("/api/questions/list")
async def list_questions():
    questions = []
    for question_id, question in QUESTIONS_DB.items():
        preview = question["text"][:100] + ("..." if len(question["text"]) > 100 else "")
        questions.append(
            {
                "id": question_id,
                "preview": preview,
                "option_count": len(question["options"]),
            }
        )

    return JSONResponse({"questions": questions})


@app.post("/api/submit")
async def submit_answer(req: SubmitRequest):
    if req.question_id not in QUESTIONS_DB:
        return JSONResponse({"error": "Question not found."}, status_code=404)

    question = QUESTIONS_DB[req.question_id]
    selected_answer = normalize_space(req.selected_answer or req.student_answer or "")
    explanation = normalize_space(req.student_explanation)

    if not selected_answer:
        return JSONResponse({"error": "Please choose one answer option."}, status_code=400)

    valid_answers = {normalize_answer(option): option for option in question["options"]}
    selected_answer = valid_answers.get(normalize_answer(selected_answer), selected_answer)
    if normalize_answer(selected_answer) not in valid_answers:
        return JSONResponse({"error": "Selected answer is not valid for this question."}, status_code=400)

    feedback = build_feedback(question, selected_answer, explanation)
    return JSONResponse({"success": True, "feedback": feedback})


@app.post("/api/rate")
async def rate_feedback(req: FeedbackRequest):
    return JSONResponse({"success": True, "message": "Thanks for rating the feedback."})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
