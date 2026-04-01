"""
Pydantic schemas for API request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    question_id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="The question text")
    mc_answer: str = Field(..., description="Student's selected MCQ answer")
    student_explanation: str = Field(..., description="Student's explanation of their reasoning")
    is_correct: Optional[bool] = Field(None, description="Whether the answer is correct (optional, can be inferred)")


class PredictionResult(BaseModel):
    """Single prediction with confidence."""
    label: str
    confidence: float


class MisconceptionCard(BaseModel):
    """Knowledge base card for a misconception."""
    plain_definition: str
    typical_wrong_thought: List[str]
    correct_rule: List[str]
    fix_steps: List[str]
    quick_check: str
    tiny_example: str


class FeedbackResult(BaseModel):
    """Generated feedback for the student."""
    teacher_feedback: str
    misconception_card: Optional[MisconceptionCard] = None
    correct_answer: Optional[str] = None
    is_student_correct: bool


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    success: bool
    prediction: Dict[str, Any] = Field(..., description="Prediction results")
    feedback: FeedbackResult


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    explainer_loaded: bool
    kb_entries: int
    label_classes: int


class QuestionOption(BaseModel):
    """MCQ option."""
    id: str
    text: str


class QuestionResponse(BaseModel):
    """Response schema for question endpoint."""
    id: str
    text: str
    options: List[QuestionOption]
    has_image: bool = False


class SubmitRequest(BaseModel):
    """Compatibility schema matching existing UI."""
    question_id: str
    selected_answer: Optional[str] = None
    student_answer: Optional[str] = None
    student_explanation: str
    # Optional fields for teacher-created questions
    question_text: Optional[str] = None
    is_correct: Optional[bool] = None
    options: Optional[List[str]] = None


class LegacyFeedback(BaseModel):
    """Legacy feedback format for backward compatibility with existing UI."""
    is_correct: bool
    selected_answer: str
    correct_answer: str
    result_color: str
    result_headline: str
    result_message: str
    reasoning_label: str
    reasoning_message: str
    answer_label: str
    has_misconception: bool
    misconception_name: str
    misconception_tip: str
    what_went_wrong: Optional[str] = None
    correct_explanation: str
    confidence: float
    # New fields from real model
    top_predictions: Optional[List[PredictionResult]] = None
    teacher_feedback: Optional[str] = None
    misconception_card: Optional[Dict[str, Any]] = None


class SubmitResponse(BaseModel):
    """Response schema for submit endpoint (legacy compatibility)."""
    success: bool
    feedback: LegacyFeedback


class LabelsResponse(BaseModel):
    """Response schema for labels endpoint."""
    labels: List[str]
    count: int
