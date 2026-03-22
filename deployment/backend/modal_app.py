"""
Modal Deployment for MAP Misconception Detection API.
Deploy with: modal deploy modal_app.py
"""

import modal
import os

# Create Modal app
app = modal.App("map-misconception-detector")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "torch>=2.0.0",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "accelerate>=0.25.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "huggingface-hub>=0.20.0",
        "bitsandbytes>=0.41.0",
    )
)

# Mount local data directory
data_mount = modal.Mount.from_local_dir(
    "./data",
    remote_path="/app/data",
)

# Mount app code
app_mount = modal.Mount.from_local_dir(
    "./app",
    remote_path="/app/app",
)


@app.cls(
    image=image,
    gpu="T4",  # Use T4 GPU (cheapest option with good performance)
    mounts=[data_mount, app_mount],
    container_idle_timeout=300,  # Keep warm for 5 minutes
    secrets=[modal.Secret.from_name("hf-secret")],  # HuggingFace token
    timeout=600,  # 10 minute timeout for requests
)
class MisconceptionAPI:
    """Modal class for the API deployment."""

    @modal.enter()
    def startup(self):
        """Load models on container startup."""
        import sys
        sys.path.insert(0, "/app")

        # Set environment variables
        os.environ["TRAIN_DATA_PATH"] = "/app/data/train.csv"
        os.environ["KB_PATH"] = "/app/data/misconception_kb.json"
        os.environ["LABEL_ENCODER_PATH"] = "/app/data/label_encoder.pkl"
        os.environ["CORRECT_ANSWERS_PATH"] = "/app/data/correct_answers.json"
        os.environ["USE_4BIT_QUANTIZATION"] = "true"

        from app.services import get_prediction_service
        from app.main import load_questions

        # Load questions
        load_questions()

        # Load prediction service (includes model loading)
        self.service = get_prediction_service()
        self.service.load()
        print("API initialized!")

    @modal.web_endpoint(method="GET")
    def health(self):
        """Health check endpoint."""
        from app.models import get_classifier, get_explainer

        classifier = get_classifier()
        explainer = get_explainer()

        return {
            "status": "ok",
            "model_loaded": classifier.is_loaded,
            "explainer_loaded": explainer.is_loaded,
            "kb_entries": explainer.kb_count,
            "label_classes": classifier.n_classes if classifier.is_loaded else 0,
        }

    @modal.web_endpoint(method="POST")
    def predict(self, request: dict):
        """Prediction endpoint."""
        try:
            result = self.service.predict(
                question_id=request.get("question_id", ""),
                question_text=request.get("question_text", ""),
                mc_answer=request.get("mc_answer", ""),
                student_explanation=request.get("student_explanation", ""),
                is_correct=request.get("is_correct"),
            )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @modal.web_endpoint(method="POST")
    def submit(self, request: dict):
        """Legacy submit endpoint for UI compatibility."""
        from app.main import QUESTIONS_DB
        from app.utils import normalize_space, normalize_answer

        question_id = request.get("question_id", "")
        if question_id not in QUESTIONS_DB:
            return {"success": False, "error": "Question not found"}

        question = QUESTIONS_DB[question_id]
        selected_answer = normalize_space(
            request.get("selected_answer") or request.get("student_answer") or ""
        )
        explanation = normalize_space(request.get("student_explanation", ""))

        if not selected_answer:
            return {"success": False, "error": "Please choose an answer"}

        try:
            feedback = self.service.get_legacy_feedback(
                question_id=question_id,
                question_text=question["text"],
                selected_answer=selected_answer,
                student_explanation=explanation,
                options=question["options"],
            )
            return {"success": True, "feedback": feedback}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @modal.web_endpoint(method="GET")
    def question(self, question_id: str = None):
        """Get a question."""
        import random
        from app.main import QUESTIONS_DB

        if not QUESTIONS_DB:
            return {"error": "No questions available"}

        if question_id and question_id in QUESTIONS_DB:
            q = QUESTIONS_DB[question_id]
        else:
            q = random.choice(list(QUESTIONS_DB.values()))

        options = [
            {"id": chr(65 + i), "text": opt}
            for i, opt in enumerate(q["options"])
        ]

        return {
            "id": q["id"],
            "text": q["text"],
            "options": options,
            "has_image": "[Image:" in q["text"] or "[image:" in q["text"].lower(),
        }

    @modal.web_endpoint(method="GET")
    def labels(self):
        """Get all labels."""
        from app.models import get_classifier

        classifier = get_classifier()
        if not classifier.is_loaded:
            return {"error": "Model not loaded"}

        labels = classifier.get_labels()
        return {"labels": labels, "count": len(labels)}


# For local testing with `modal serve`
@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("Modal app ready for deployment!")
    print("Deploy with: modal deploy modal_app.py")
    print("Serve locally with: modal serve modal_app.py")
