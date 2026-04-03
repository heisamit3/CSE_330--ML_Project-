"""
Notebook-aligned explainer for generating teacher feedback.

Supports both:
- the original deployment KB format keyed by misconception name
- the richer list-based KB exported from `Extra/kaggle_notebooks/rf_results`
"""

import gc
import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

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
    Qwen3-8B based explainer for generating teacher feedback.
    Uses the same base model as classifier but loaded as CausalLM.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.kb: Dict[str, Any] = {}
        self.kb_entries: List[Dict[str, Any]] = []
        self._kb_entry_count = 0
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
            logger.info("Loading Qwen3-8B explainer (CausalLM)...")

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
                raw_kb = json.loads(kb_path.read_text(encoding="utf-8"))
                self.kb_entries = self._normalize_kb_payload(raw_kb)
                self.kb = self._build_kb_index(self.kb_entries)
                self._kb_entry_count = len(self.kb_entries)
                logger.info(
                    "Loaded KB with %s canonical entries and %s lookup keys",
                    self._kb_entry_count,
                    len(self.kb),
                )
            except Exception as e:
                logger.error(f"Failed to load KB: {e}")
                self.kb = {}
                self.kb_entries = []
                self._kb_entry_count = 0
        else:
            logger.warning(f"KB not found at {kb_path}")
            self.kb = {}
            self.kb_entries = []
            self._kb_entry_count = 0

    def _normalize_kb_payload(self, payload: Any) -> List[Dict[str, Any]]:
        """Normalize either dict-based or list-based KB payloads into a common schema."""
        if isinstance(payload, dict):
            raw_entries = []
            for key, value in payload.items():
                if isinstance(value, dict):
                    entry = dict(value)
                else:
                    entry = {"plain_definition": str(value)}
                entry.setdefault("misconception_name", key)
                entry.setdefault("misconception_id", key)
                raw_entries.append(entry)
        elif isinstance(payload, list):
            raw_entries = [entry for entry in payload if isinstance(entry, dict)]
        else:
            raw_entries = []

        return [self._normalize_kb_entry(entry) for entry in raw_entries]

    def _normalize_kb_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize one KB entry so downstream code can use a stable shape."""
        normalized = dict(entry)

        raw_id = str(
            entry.get("misconception_id")
            or entry.get("misconception_name")
            or ""
        ).strip()
        raw_name = str(
            entry.get("misconception_name")
            or (raw_id.split(":", 1)[1] if ":" in raw_id else raw_id)
            or ""
        ).strip()

        misconception_id = self._clean_misconception_value(raw_id, keep_na=True)
        misconception_name = self._clean_misconception_value(raw_name, keep_na=True)

        typical_wrong_thought = entry.get("typical_wrong_thought")
        if typical_wrong_thought is None:
            typical_wrong_thought = entry.get("evidence_examples", [])
        if isinstance(typical_wrong_thought, str):
            typical_wrong_thought = [typical_wrong_thought]
        elif not isinstance(typical_wrong_thought, list):
            typical_wrong_thought = []

        fix_steps = entry.get("fix_steps")
        if isinstance(fix_steps, str):
            fix_steps = [fix_steps]
        elif not isinstance(fix_steps, list):
            fix_steps = []
        if not fix_steps:
            fix_steps = self._build_fix_steps(entry)

        quick_check = (
            entry.get("quick_check")
            or entry.get("formal_counterexample")
            or "Does your method still work if you test it on a simpler example?"
        )

        plain_definition = (
            entry.get("plain_definition")
            or entry.get("short_description")
            or f"The student shows a pattern consistent with '{misconception_name}' reasoning."
        )

        normalized.update({
            "misconception_id": misconception_id,
            "misconception_name": misconception_name,
            "plain_definition": plain_definition,
            "typical_wrong_thought": [str(item) for item in typical_wrong_thought if str(item).strip()],
            "fix_steps": [str(item) for item in fix_steps if str(item).strip()],
            "quick_check": str(quick_check).strip(),
        })

        aliases = entry.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        elif not isinstance(aliases, list):
            aliases = []
        normalized["aliases"] = [str(item).strip() for item in aliases if str(item).strip()]

        return normalized

    def _build_fix_steps(self, entry: Dict[str, Any]) -> List[str]:
        """Create practical feedback steps for the richer rf_results KB schema."""
        steps: List[str] = []

        remediation_hint = str(entry.get("remediation_hint") or "").strip()
        if remediation_hint:
            steps.append(remediation_hint)

        prerequisite_concepts = entry.get("prerequisite_concepts", [])
        if isinstance(prerequisite_concepts, list) and prerequisite_concepts:
            concepts = ", ".join(str(item).strip() for item in prerequisite_concepts[:3] if str(item).strip())
            if concepts:
                steps.append(f"Review these ideas first: {concepts}.")

        trigger_patterns = entry.get("trigger_patterns", [])
        if isinstance(trigger_patterns, list) and trigger_patterns:
            patterns = ", ".join(str(item).strip() for item in trigger_patterns[:3] if str(item).strip())
            if patterns:
                steps.append(f"Check whether your explanation is relying on cues like: {patterns}.")

        formal_counterexample = str(entry.get("formal_counterexample") or "").strip()
        if formal_counterexample:
            steps.append("Test your method against a simple counterexample before finalizing your answer.")

        if not steps:
            steps = [
                "Re-read what the question is asking.",
                "Write down the correct mathematical rule.",
                "Apply it step by step to this problem.",
                "Check your answer with a simpler example.",
            ]

        return steps[:4]

    def _build_kb_index(self, entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build a flexible lookup index for misconception name/id/alias access."""
        index: Dict[str, Dict[str, Any]] = {}

        for entry in entries:
            for raw_key in self._kb_lookup_values(entry):
                lookup_key = self._normalize_lookup_key(raw_key)
                if not lookup_key:
                    continue
                # Skip notebook placeholder NA entries; the app handles NA generically.
                if lookup_key == "na":
                    continue
                index.setdefault(lookup_key, entry)

        return index

    def _kb_lookup_values(self, entry: Dict[str, Any]) -> List[str]:
        """Return candidate lookup strings for one KB entry."""
        values: List[str] = []

        for key in ("misconception_name", "misconception_id"):
            value = str(entry.get(key) or "").strip()
            if value:
                values.append(value)
                if ":" in value:
                    values.append(value.split(":", 1)[1].strip())

        for alias in entry.get("aliases", []):
            alias_text = str(alias).strip()
            if alias_text:
                values.append(alias_text)

        return values

    def _clean_misconception_value(self, value: str, keep_na: bool = False) -> str:
        """Normalize display values coming from different KB formats."""
        text = str(value or "").strip()
        if ":" in text:
            text = text.split(":", 1)[1].strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return "NA" if keep_na else ""
        return text

    def _normalize_lookup_key(self, value: str) -> str:
        """Normalize lookup keys for case-insensitive KB access."""
        cleaned = self._clean_misconception_value(value, keep_na=False)
        return cleaned.lower()

    def _llm_generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """
        Generate text using the Qwen3-8B model.

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
        return self.kb.get(self._normalize_lookup_key(misconception))

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
        card = self.get_kb_card(misconception) or {}

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

            steps_text = "\n".join(f"{i+1}) {step}" for i, step in enumerate(fix_steps))

            return f"""Teacher Feedback
Student chose: {mc_answer} | Correct: {ic_str} | Correct answer: {correct_str}
Predicted misconception: {predicted_label}

What's happening: {definition}

Steps to fix:
{steps_text}
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
        return self._kb_entry_count


# Singleton instance
_explainer: Optional[MisconceptionExplainer] = None


def get_explainer() -> MisconceptionExplainer:
    """Get the singleton explainer instance."""
    global _explainer
    if _explainer is None:
        _explainer = MisconceptionExplainer()
    return _explainer
