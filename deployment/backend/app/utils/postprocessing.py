"""
Postprocessing utilities for prediction filtering and verification.
Extracted from q3-4b-exp.ipynb notebook.
"""

import re
from typing import List, Tuple, Optional


def get_family_prefix(is_correct: bool) -> str:
    """Get the family prefix based on correctness."""
    return "True_" if is_correct else "False_"


def filter_by_family(labels: List[str], is_correct: bool) -> List[str]:
    """
    Filter predictions by family prefix (True_/False_).

    Args:
        labels: List of predicted labels
        is_correct: Whether the student's answer is correct

    Returns:
        Filtered list of labels matching the family
    """
    prefix = get_family_prefix(is_correct)
    return [lbl for lbl in labels if lbl.startswith(prefix)]


def get_top_k_with_fallback(labels: List[str], is_correct: bool, k: int = 3) -> List[str]:
    """
    Get top-k predictions with fallback filling.

    Args:
        labels: Filtered labels
        is_correct: Whether the answer is correct
        k: Number of predictions to return

    Returns:
        List of k labels
    """
    prefix = get_family_prefix(is_correct)
    top_k = labels[:k]

    # Fill with fallback labels if needed
    while len(top_k) < k:
        if is_correct:
            fallback = f"{prefix}Neither:NA"
            if fallback not in top_k:
                top_k.append(fallback)
            else:
                top_k.append(f"{prefix}Correct:NA")
        else:
            fallback = f"{prefix}Neither:NA"
            if fallback not in top_k:
                top_k.append(fallback)
            else:
                top_k.append(f"{prefix}Misconception:NA")

    return top_k[:k]


def parse_label(label: str) -> Tuple[str, str, str]:
    """
    Parse a label into its components.

    Args:
        label: Full label like "False_Misconception:Incomplete"

    Returns:
        Tuple of (correctness, category, misconception)
    """
    # Split by underscore first
    parts = label.split("_", 1)
    correctness = parts[0]  # "True" or "False"

    if len(parts) > 1:
        # Split by colon
        subparts = parts[1].split(":", 1)
        category = subparts[0]  # "Misconception", "Correct", "Neither"
        misconception = subparts[1] if len(subparts) > 1 else "NA"
    else:
        category = "NA"
        misconception = "NA"

    return correctness, category, misconception


def rule_based_verify(feedback: str, is_correct: bool, correct_choice: Optional[str] = None) -> str:
    """
    Rule-based post-hoc verification of feedback.
    - If student is correct, soften negative language
    - If student is incorrect and we know the answer, prepend it

    Args:
        feedback: Generated feedback text
        is_correct: Whether student's answer is correct
        correct_choice: The correct answer (optional)

    Returns:
        Verified/cleaned feedback text
    """
    txt = feedback.strip()

    if is_correct:
        # Soften negative language for correct answers
        txt = re.sub(r"\bincorrect\b", "not quite right", txt, flags=re.I)
        txt = re.sub(r"\bwrong\b", "not quite right", txt, flags=re.I)

    if not is_correct and correct_choice and correct_choice not in txt:
        # Prepend correct answer for incorrect responses
        txt = f"Correct answer: {correct_choice}\n\n" + txt

    return txt


def get_category_feedback(category: str) -> dict:
    """Get feedback template for a category."""
    templates = {
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
    return templates.get(category, templates["False_Neither"])


def get_reasoning_summary(reasoning_band: str) -> dict:
    """Get reasoning summary for a band."""
    summaries = {
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
    return summaries.get(reasoning_band, summaries["Neither"])
