"""
Prediction service that combines classifier and explainer.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ..config import TRAIN_DATA_PATH, CORRECT_ANSWERS_PATH
from ..models import get_classifier, get_explainer
from ..utils import (
    format_input,
    filter_by_family,
    get_top_k_with_fallback,
    parse_label,
    rule_based_verify,
    get_category_feedback,
    get_reasoning_summary,
    normalize_answer,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for running predictions and generating feedback."""

    def __init__(self):
        self.classifier = get_classifier()
        self.explainer = get_explainer()
        self.correct_answers: Dict[str, str] = {}
        self._loaded = False

    def load(self) -> bool:
        """Load all components."""
        if self._loaded:
            return True

        success = True

        # Load classifier
        if not self.classifier.load():
            logger.error("Failed to load classifier")
            success = False

        # Load explainer
        if not self.explainer.load():
            logger.error("Failed to load explainer")
            success = False

        # Load correct answers
        self._load_correct_answers()

        self._loaded = success
        return success

    def _load_correct_answers(self):
        """Load correct answers map from file or train data."""
        # Try loading from pre-built file
        ca_path = Path(CORRECT_ANSWERS_PATH)
        if ca_path.exists():
            try:
                self.correct_answers = json.loads(ca_path.read_text(encoding="utf-8"))
                logger.info(f"Loaded {len(self.correct_answers)} correct answers")
                return
            except Exception as e:
                logger.warning(f"Failed to load correct answers file: {e}")

        # Build from train data
        train_path = Path(TRAIN_DATA_PATH)
        if train_path.exists():
            try:
                import pandas as pd
                train_df = pd.read_csv(train_path)

                # Find correct answers (True_ categories)
                idx = train_df['Category'].apply(lambda c: c.split('_')[0]) == 'True'
                correct = train_df.loc[idx].copy()
                correct['c'] = correct.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
                correct = correct.sort_values('c', ascending=False).drop_duplicates(['QuestionId'])

                self.correct_answers = dict(zip(
                    correct['QuestionId'].astype(str),
                    correct['MC_Answer']
                ))
                logger.info(f"Built correct answers map with {len(self.correct_answers)} entries")

                # Save for future use
                ca_path.parent.mkdir(parents=True, exist_ok=True)
                ca_path.write_text(json.dumps(self.correct_answers, ensure_ascii=False, indent=2))
            except Exception as e:
                logger.error(f"Failed to build correct answers: {e}")
        else:
            logger.warning(f"Train data not found at {train_path}")

    def get_correct_answer(self, question_id: str) -> Optional[str]:
        """Get the correct answer for a question."""
        return self.correct_answers.get(str(question_id))

    def infer_is_correct(self, question_id: str, selected_answer: str) -> bool:
        """Infer if the selected answer is correct."""
        correct = self.get_correct_answer(question_id)
        if correct is None:
            return False
        return normalize_answer(selected_answer) == normalize_answer(correct)

    def predict(
        self,
        question_id: str,
        question_text: str,
        mc_answer: str,
        student_explanation: str,
        is_correct: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Run full prediction pipeline.

        Args:
            question_id: Question identifier
            question_text: The question text
            mc_answer: Student's selected answer
            student_explanation: Student's explanation
            is_correct: Whether the answer is correct (inferred if None)

        Returns:
            Dictionary with prediction results and feedback
        """
        if not self._loaded:
            raise RuntimeError("Service not loaded. Call load() first.")

        # Infer correctness if not provided
        if is_correct is None:
            is_correct = self.infer_is_correct(question_id, mc_answer)

        correct_answer = self.get_correct_answer(question_id)

        # Format input for model
        input_text = format_input(question_text, mc_answer, student_explanation, is_correct)

        # Run classifier
        sorted_labels, probs = self.classifier.predict(input_text)

        # Filter by family and get top-3
        filtered_labels = filter_by_family(sorted_labels, is_correct)
        top3_labels = get_top_k_with_fallback(filtered_labels, is_correct, k=3)

        # Get top prediction info
        top1_label = top3_labels[0]
        correctness, category, misconception = parse_label(top1_label)

        # Get confidence for top predictions
        top_predictions = []
        for label in top3_labels:
            idx = list(sorted_labels).index(label) if label in sorted_labels else -1
            conf = float(probs[idx]) if idx >= 0 else 0.0
            top_predictions.append({"label": label, "confidence": conf})

        # Generate teacher feedback
        feedback_text = self.explainer.generate_feedback(
            question_text=question_text,
            mc_answer=mc_answer,
            student_explanation=student_explanation,
            predicted_label=top1_label,
            top3_labels=top3_labels,
            is_correct=is_correct,
            correct_answer=correct_answer
        )

        # Apply rule-based verification
        feedback_text = rule_based_verify(feedback_text, is_correct, correct_answer)

        # Get KB card
        kb_card = self.explainer.get_kb_card(misconception)

        # Build category key for feedback
        category_key = f"{'True' if is_correct else 'False'}_{category}"
        category_feedback = get_category_feedback(category_key)
        reasoning_summary = get_reasoning_summary(category)

        return {
            "success": True,
            "prediction": {
                "top_predictions": top_predictions,
                "predicted_category": category_key,
                "predicted_misconception": misconception,
                "top1_label": top1_label,
            },
            "feedback": {
                "teacher_feedback": feedback_text,
                "misconception_card": kb_card,
                "correct_answer": correct_answer,
                "is_student_correct": is_correct,
            },
            "ui": {
                "result_color": category_feedback["color"],
                "result_headline": category_feedback["headline"],
                "result_message": category_feedback["message"],
                "reasoning_label": reasoning_summary["label"],
                "reasoning_message": reasoning_summary["message"],
            }
        }

    def get_legacy_feedback(
        self,
        question_id: str,
        question_text: str,
        selected_answer: str,
        student_explanation: str,
        options: List[str],
        is_correct: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Get feedback in legacy format for backward compatibility with existing UI.

        Args:
            question_id: Question identifier
            question_text: The question text
            selected_answer: Student's selected answer
            student_explanation: Student's explanation
            options: List of answer options
            is_correct: Whether the answer is correct

        Returns:
            Dictionary matching the legacy feedback format
        """
        result = self.predict(
            question_id=question_id,
            question_text=question_text,
            mc_answer=selected_answer,
            student_explanation=student_explanation,
            is_correct=is_correct
        )

        prediction = result["prediction"]
        feedback = result["feedback"]
        ui = result["ui"]

        misconception = prediction["predicted_misconception"]
        has_misconception = (
            "Misconception" in prediction["predicted_category"] and
            misconception != "NA"
        )

        # Get misconception tip from KB card
        kb_card = feedback.get("misconception_card", {})
        misconception_tip = ""
        what_went_wrong = None

        if kb_card:
            misconception_tip = kb_card.get("plain_definition", "")
            wrong_thoughts = kb_card.get("typical_wrong_thought", [])
            if wrong_thoughts:
                what_went_wrong = wrong_thoughts[0] if isinstance(wrong_thoughts, list) else str(wrong_thoughts)

        return {
            "is_correct": feedback["is_student_correct"],
            "selected_answer": selected_answer,
            "correct_answer": feedback["correct_answer"] or "See solution",
            "predicted_label": prediction["top1_label"],
            "predicted_category": prediction["predicted_category"],
            "predicted_misconception": prediction["predicted_misconception"],
            "result_color": ui["result_color"],
            "result_headline": ui["result_headline"],
            "result_message": ui["result_message"],
            "reasoning_label": ui["reasoning_label"],
            "reasoning_message": ui["reasoning_message"],
            "answer_label": "Choice: correct" if feedback["is_student_correct"] else "Choice: incorrect",
            "has_misconception": has_misconception,
            "misconception_name": misconception if has_misconception else "No specific misconception",
            "misconception_tip": misconception_tip or "Review your approach and try again.",
            "what_went_wrong": what_went_wrong,
            "correct_explanation": feedback["teacher_feedback"],
            "confidence": prediction["top_predictions"][0]["confidence"] if prediction["top_predictions"] else 0.5,
            # New fields from real model
            "top_predictions": prediction["top_predictions"],
            "teacher_feedback": feedback["teacher_feedback"],
            "misconception_card": kb_card,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton instance
_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get the singleton prediction service."""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service
