"""
Preprocessing utilities for input text formatting.
Extracted from q3-4b-exp.ipynb notebook.
"""

import re
from typing import Dict, Any


def format_input(question_text: str, mc_answer: str, student_explanation: str, is_correct: bool) -> str:
    """
    Format input text for Qwen3-4B model.

    Args:
        question_text: The question text
        mc_answer: Student's selected MCQ answer
        student_explanation: Student's explanation
        is_correct: Whether the answer is correct

    Returns:
        Formatted input string for the model
    """
    correctness_text = "This answer is correct." if is_correct else "This answer is incorrect."
    return (
        f"Question: {question_text}\n"
        f"Answer: {mc_answer}\n"
        f"{correctness_text}\n"
        f"Student Explanation: {student_explanation}"
    )


def format_input_from_row(row: Dict[str, Any]) -> str:
    """
    Format input text from a row dictionary.

    Args:
        row: Dictionary with QuestionText, MC_Answer, StudentExplanation, is_correct

    Returns:
        Formatted input string
    """
    return format_input(
        question_text=str(row.get('QuestionText', '')),
        mc_answer=str(row.get('MC_Answer', '')),
        student_explanation=str(row.get('StudentExplanation', '')),
        is_correct=bool(row.get('is_correct', False))
    )


def normalize_space(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    cleaned = normalize_space(text)
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("\\", "")
    return cleaned.lower()


def truncate_text(text: str, max_length: int = 300) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
