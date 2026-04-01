from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.utils.json_utils import heuristic_repair_json
from src.utils.text import tokenize


class BaseLLMBackend:
    backend_name = "base"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        raise NotImplementedError


class MockLLMBackend(BaseLLMBackend):
    backend_name = "mock"

    def _extract(self, prompt: str, label: str) -> str:
        pattern = re.compile(rf"{re.escape(label)}:\s*(.*)")
        match = pattern.search(prompt)
        return match.group(1).strip() if match else ""

    def _score_payload(self, prompt: str) -> dict[str, Any]:
        candidate_id = self._extract(prompt, "Candidate misconception id")
        candidate_name = self._extract(prompt, "Candidate misconception name")
        explanation_match = re.search(r"Student explanation:\s*(.*?)(?:\nKnowledge context:|\nReturn JSON only|$)", prompt, re.S)
        explanation = explanation_match.group(1).strip() if explanation_match else prompt
        trigger_match = re.search(r"Matched trigger patterns:\s*(.*)", prompt)
        trigger_text = trigger_match.group(1).strip() if trigger_match else ""
        ontology_support_match = re.search(r"Ontology support score:\s*([0-9.]+)", prompt)
        ontology_support = float(ontology_support_match.group(1)) if ontology_support_match else 0.0

        explanation_tokens = set(tokenize(explanation))
        candidate_tokens = set(tokenize(candidate_name + " " + prompt))
        overlap = len(explanation_tokens & candidate_tokens) / max(1, len(candidate_tokens))
        trigger_bonus = 0.25 if trigger_text and trigger_text.lower() != "none" else 0.0
        affinity = max(0.0, min(1.0, 0.18 + overlap * 2.0 + trigger_bonus + 0.20 * ontology_support))
        confidence = max(0.0, min(1.0, 0.30 + affinity * 0.40 + 0.15 * ontology_support + trigger_bonus))
        evidence = []
        if trigger_text and trigger_text.lower() != "none":
            evidence = [piece.strip() for piece in trigger_text.split("|") if piece.strip()][:2]
        elif explanation:
            evidence = [explanation[:80]]
        return {
            "misconception_id": candidate_id,
            "affinity_score": round(affinity, 4),
            "confidence": round(confidence, 4),
            "reasoning_short": f"Evidence aligns with {candidate_name or candidate_id}.".strip(),
            "evidence_spans": evidence,
        }

    def _counterfactual_payload(self, prompt: str) -> dict[str, Any]:
        original_match = re.search(r"Original explanation:\s*(.*?)(?:\nSuspected misconception:|$)", prompt, re.S)
        original = original_match.group(1).strip() if original_match else ""
        remediation = self._extract(prompt, "Remediation hint")
        trigger_match = re.search(r"Matched trigger patterns:\s*(.*)", prompt)
        trigger_text = trigger_match.group(1).strip() if trigger_match else ""
        sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", original) if piece.strip()]
        filtered = []
        triggers = [item.strip().lower() for item in trigger_text.split("|") if item.strip()]
        for sentence in sentences:
            lowered = sentence.lower()
            if triggers and any(trigger in lowered for trigger in triggers):
                continue
            filtered.append(sentence)
        if not filtered:
            replacement = remediation or "The student now applies the intended rule instead of the suspected misconception."
            filtered = [replacement]
        counterfactual_text = " ".join(filtered)
        original_tokens = set(tokenize(original))
        counterfactual_tokens = set(tokenize(counterfactual_text))
        style_score = len(original_tokens & counterfactual_tokens) / max(1, len(original_tokens)) if original_tokens else 0.6
        return {
            "counterfactual_text": counterfactual_text,
            "rewrite_confidence": 0.65,
            "style_preservation_score": round(style_score, 4),
            "explanation": "Removed misconception-heavy language while preserving most wording.",
        }

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if "TASK=json_repair" in prompt:
            malformed = prompt.split("Malformed JSON:", 1)[-1]
            return heuristic_repair_json(malformed)
        if "TASK=counterfactual_rewrite" in prompt:
            return json.dumps(self._counterfactual_payload(prompt), ensure_ascii=False)
        return json.dumps(self._score_payload(prompt), ensure_ascii=False)


class HuggingFaceLLMBackend(BaseLLMBackend):
    backend_name = "hf"

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        load_in_4bit: bool = False,
        attn_implementation: str | None = None,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_value = getattr(torch, dtype, torch.bfloat16)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype_value,
            "local_files_only": local_files_only,
        }
        if device != "cpu":
            model_kwargs["device_map"] = "auto"
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype_value,
                )
            except Exception:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _format_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        system_prefix = f"System: {system_prompt}\n\n" if system_prompt else ""
        return f"{system_prefix}User: {prompt}\nAssistant:"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        import torch

        formatted = self._format_prompt(prompt, system_prompt=system_prompt)
        encoded = self.tokenizer(formatted, return_tensors="pt")
        if self.device != "cpu":
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 0.01),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        completion = output[0][prompt_length:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


def _is_existing_local_model_ref(ref: str) -> bool:
    try:
        return Path(ref).exists()
    except Exception:
        return False


def build_backend(config: dict[str, Any], logger) -> BaseLLMBackend:
    llm_cfg = config.get("llm", {})
    if not llm_cfg.get("enabled", False) or llm_cfg.get("backend", "mock") == "mock":
        return MockLLMBackend()

    backend_name = llm_cfg.get("backend", "hf")
    if backend_name != "hf":
        logger.warning("Unsupported backend '%s'. Falling back to mock.", backend_name)
        return MockLLMBackend()

    model_candidates: list[str] = []
    primary_ref = str(llm_cfg.get("model_name_or_path") or "").strip()
    if primary_ref:
        model_candidates.append(primary_ref)

    for candidate in llm_cfg.get("model_candidates", []):
        text = str(candidate or "").strip()
        if text and text not in model_candidates:
            model_candidates.append(text)

    if not model_candidates:
        logger.warning("No llm.model_name_or_path or llm.model_candidates provided. Falling back to mock backend.")
        return MockLLMBackend()

    last_error: Exception | None = None
    for candidate_ref in model_candidates:
        is_local = _is_existing_local_model_ref(candidate_ref)
        if (candidate_ref.startswith('/') or candidate_ref.startswith('./') or candidate_ref.startswith('../')) and not is_local:
            logger.info("Skipping missing local model candidate: %s", candidate_ref)
            continue

        candidate_local_only = bool(llm_cfg.get("local_files_only", False)) if is_local else False
        logger.info("Trying HF backend candidate: %s", candidate_ref)
        try:
            return HuggingFaceLLMBackend(
                model_name_or_path=candidate_ref,
                device=llm_cfg.get("device", "cuda"),
                dtype=llm_cfg.get("dtype", "bfloat16"),
                max_new_tokens=int(llm_cfg.get("max_new_tokens", 256)),
                temperature=float(llm_cfg.get("temperature", 0.0)),
                load_in_4bit=bool(llm_cfg.get("load_in_4bit", False)),
                attn_implementation=llm_cfg.get("attn_implementation"),
                local_files_only=candidate_local_only,
            )
        except Exception as exc:
            last_error = exc
            logger.warning("Could not load HF backend candidate %s (%s)", candidate_ref, exc)

    if llm_cfg.get("allow_backend_fallback", True):
        logger.warning("All HF backend candidates failed. Falling back to mock backend.")
        return MockLLMBackend()
    if last_error is not None:
        raise last_error
    raise RuntimeError("No usable HF backend candidate was found.")
