"""
Qwen3-4B CausalLM Explainer for generating teacher feedback.
Extracted from q3-4b-exp.ipynb notebook.
"""

import gc
import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from ..config import (
    BASE_MODEL_PATH,
    USE_4BIT_QUANTIZATION,
    USE_LLM_EXPLAINER,
    MAX_NEW_TOKENS,
    KB_PATH,
    HF_TOKEN,
    get_device,
)

logger = logging.getLogger(__name__)


class MisconceptionExplainer:
    """
    Qwen3-4B based explainer for generating teacher feedback.
    Uses the same base model as classifier but loaded as CausalLM.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.kb: Dict[str, Any] = {}
        self.device = get_device()
        self._loaded = False

    def load(self) -> bool:
        """
        Load the explainer model and knowledge base.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            return True

        # Load knowledge base first (doesn't require GPU)
        self._load_kb()

        if not USE_LLM_EXPLAINER:
            logger.info("LLM explainer disabled, using KB-only mode")
            self._loaded = True
            return True

        try:
            logger.info("Loading Qwen3-4B explainer (CausalLM)...")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_PATH,
                trust_remote_code=True,
                token=HF_TOKEN,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Try 4-bit quantization first
            if USE_4BIT_QUANTIZATION and "cuda" in self.device:
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL_PATH,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        token=HF_TOKEN,
                    )
                    logger.info("Explainer loaded with 4-bit quantization")
                except Exception as e:
                    logger.warning(f"4-bit quantization failed: {e}")
                    self.model = None

            # Fallback to float16 without quantization
            if self.model is None and "cuda" in self.device:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL_PATH,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        token=HF_TOKEN,
                    )
                    logger.info("Explainer loaded in float16")
                except Exception as e:
                    logger.warning(f"Float16 loading failed: {e}")
                    self.model = None

            # CPU fallback (will be slow)
            if self.model is None:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL_PATH,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        token=HF_TOKEN,
                    )
                    logger.info("Explainer loaded on CPU (will be slow)")
                except Exception as e:
                    logger.error(f"All loading methods failed: {e}")
                    logger.info("Falling back to KB-only mode")
                    self.model = None

            if self.model is not None:
                self.model.eval()

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load explainer: {e}")
            self._loaded = True  # Mark as loaded but in KB-only mode
            return True

    def _load_kb(self):
        """Load the misconception knowledge base."""
        kb_path = Path(KB_PATH)
        if kb_path.exists():
            try:
                self.kb = json.loads(kb_path.read_text(encoding="utf-8"))
                logger.info(f"Loaded KB with {len(self.kb)} entries")
            except Exception as e:
                logger.error(f"Failed to load KB: {e}")
                self.kb = {}
        else:
            logger.warning(f"KB not found at {kb_path}")
            self.kb = {}

    def _llm_generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """
        Generate text using the Qwen3-4B model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            return ""

        # Format prompt for Qwen3 chat template
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking for faster response
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and extract generated part
        full_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

        if full_text.startswith(input_text):
            text = full_text[len(input_text):].strip()
        else:
            text = full_text.strip()

        return text

    def get_kb_card(self, misconception: str) -> Optional[Dict[str, Any]]:
        """Get the knowledge base card for a misconception."""
        return self.kb.get(misconception)

    def generate_feedback(
        self,
        question_text: str,
        mc_answer: str,
        student_explanation: str,
        predicted_label: str,
        top3_labels: list,
        is_correct: bool,
        correct_answer: Optional[str] = None
    ) -> str:
        """
        Generate teacher feedback for a prediction.

        Args:
            question_text: The question text
            mc_answer: Student's selected answer
            student_explanation: Student's explanation
            predicted_label: Top predicted label
            top3_labels: Top 3 predicted labels
            is_correct: Whether the answer is correct
            correct_answer: The correct answer (optional)

        Returns:
            Generated feedback text
        """
        # Parse misconception from label
        misconception = predicted_label.split(":", 1)[1] if ":" in predicted_label else predicted_label

        # Get KB card
        card = self.kb.get(misconception, {})

        ic_str = "Yes" if is_correct else "No"

        if self.model is not None:
            # Use LLM to generate feedback
            prompt = f"""You are a kind, clear math teacher. Use ONLY the info below.

Question: {question_text[:300]}
Student chose: {mc_answer}
Is correct? {ic_str}
Correct choice: {correct_answer or 'Unknown'}
Student explanation: {student_explanation[:300]}
Top predictions: {top3_labels}
Misconception knowledge card:
{json.dumps(card, ensure_ascii=False, indent=2)[:600]}

Write:
- 1-2 lines: what the student seems to think
- 1 line: misconception in simple words
- 4-7 short steps: correct logic
- 1 line: self-check question
Keep it concise and encouraging.
"""
            feedback = self._llm_generate(prompt, MAX_NEW_TOKENS)
            if feedback:
                return feedback

        # Fallback to rule-based feedback
        return self._generate_fallback_feedback(
            mc_answer, is_correct, correct_answer, predicted_label, card
        )

    def _generate_fallback_feedback(
        self,
        mc_answer: str,
        is_correct: bool,
        correct_answer: Optional[str],
        predicted_label: str,
        card: Dict[str, Any]
    ) -> str:
        """Generate rule-based fallback feedback."""
        ic_str = "Yes" if is_correct else "No"
        correct_str = correct_answer or "See solution"

        # Build feedback from KB card if available
        if card:
            definition = card.get("plain_definition", "A common misconception was detected.")
            fix_steps = card.get("fix_steps", [
                "Re-read what the question is asking.",
                "Write down the correct mathematical rule.",
                "Apply it step-by-step.",
                "Check with a simpler example."
            ])
            quick_check = card.get("quick_check", "Does your rule still work if the numbers change slightly?")

            steps_text = "\n".join(f"{i+1}) {step}" for i, step in enumerate(fix_steps))

            return f"""Teacher Feedback
Student chose: {mc_answer} | Correct: {ic_str} | Correct answer: {correct_str}
Predicted misconception: {predicted_label}

What's happening: {definition}

Steps to fix:
{steps_text}

Self-check: {quick_check}
"""
        else:
            return f"""Teacher Feedback
Student chose: {mc_answer} | Correct: {ic_str} | Correct answer: {correct_str}
Predicted misconception: {predicted_label}

Steps to fix:
1) Re-read what the question is asking.
2) Write down the correct mathematical rule.
3) Apply it step-by-step.
4) Check with a simpler example.

Self-check: Does your rule still work if the numbers change slightly?
"""

    def cleanup(self):
        """Clean up model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("Explainer cleaned up")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_llm(self) -> bool:
        return self.model is not None

    @property
    def kb_count(self) -> int:
        return len(self.kb)


# Singleton instance
_explainer: Optional[MisconceptionExplainer] = None


def get_explainer() -> MisconceptionExplainer:
    """Get the singleton explainer instance."""
    global _explainer
    if _explainer is None:
        _explainer = MisconceptionExplainer()
    return _explainer
