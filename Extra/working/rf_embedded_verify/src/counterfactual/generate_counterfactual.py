"""Inspired by counterfactual evaluation work (Wu et al., 2024), MalAlgoQA-style reasoning checks,
and corrected-student-solution ideas for educational error analysis.

Original idea: validate whether a predicted misconception is causally tied to the observed answer.
Simplified version here: generate a minimal style-preserving rewrite where the target misconception is absent,
then reuse the same scorer on the rewritten answer.
"""

from __future__ import annotations

import json

import pandas as pd

from src.graphrag.prompt_templates import counterfactual_rewrite_prompt
from src.utils.cache import DiskCache, make_cache_key
from src.utils.json_utils import normalize_counterfactual_payload, parse_json_maybe
from src.utils.text import tokenize


CACHE_VERSION = "oamllm_cv_v2"


class CounterfactualGenerator:
    def __init__(self, config: dict, retriever, backend, cache: DiskCache, logger):
        self.config = config
        self.retriever = retriever
        self.backend = backend
        self.cache = cache
        self.logger = logger

    def _style_preservation_score(self, original: str, rewritten: str) -> float:
        original_tokens = set(tokenize(original))
        rewritten_tokens = set(tokenize(rewritten))
        if not original_tokens:
            return 0.6
        return len(original_tokens & rewritten_tokens) / max(1, len(original_tokens))

    def _heuristic_payload(self, row: pd.Series, candidate_entry: dict, context: dict) -> dict:
        original = str(row.get("StudentExplanation", "") or "")
        triggers = [item.lower() for item in context.get("matched_trigger_patterns", [])]
        sentences = [piece.strip() for piece in original.replace("!", ".").replace("?", ".").split(".") if piece.strip()]
        filtered = []
        for sentence in sentences:
            lowered = sentence.lower()
            if triggers and any(trigger in lowered for trigger in triggers):
                continue
            filtered.append(sentence)
        if not filtered:
            replacement = candidate_entry.get("remediation_hint") or candidate_entry.get("formal_counterexample")
            filtered = [replacement or "The student now applies the intended rule without the suspected misconception."]
        counterfactual_text = ". ".join(filtered).strip()
        if counterfactual_text and not counterfactual_text.endswith("."):
            counterfactual_text += "."
        return {
            "counterfactual_text": counterfactual_text,
            "rewrite_confidence": 0.65,
            "style_preservation_score": round(self._style_preservation_score(original, counterfactual_text), 4),
            "explanation": "Removed trigger-heavy statements while keeping most wording and length similar.",
        }

    def generate(self, row: pd.Series, candidate_id: str, cache_namespace: str = "counterfactual_rewrites") -> dict:
        cache_key = make_cache_key(
            {
                "row_id": row.get("row_id"),
                "candidate_id": candidate_id,
                "text": row.get("StudentExplanation", ""),
                "cache_version": CACHE_VERSION,
            }
        )
        if self.config.get("runtime", {}).get("resume", True):
            cached = self.cache.load_json(cache_namespace, cache_key)
            if cached is not None:
                return cached

        candidate_entry = self.retriever.get_candidate_entry(candidate_id)
        context = self.retriever.retrieve(
            row,
            candidate_id,
            top_k_neighbors=int(self.config.get("graphrag", {}).get("top_k_context_neighbors", 3)),
            max_trigger_matches=int(self.config.get("graphrag", {}).get("max_trigger_matches", 3)),
        )
        prompt = counterfactual_rewrite_prompt(row, candidate_entry, context["context_block"])

        raw_text = ""
        payload = None
        try:
            if self.backend.backend_name == "mock" and not self.config.get("llm", {}).get("enabled", False):
                raw_text = json.dumps(self._heuristic_payload(row, candidate_entry, context), ensure_ascii=False)
            else:
                raw_text = self.backend.generate(prompt)
            parsed = parse_json_maybe(raw_text)
            if parsed is not None:
                payload = normalize_counterfactual_payload(parsed)
        except Exception as exc:
            self.logger.warning("Counterfactual rewrite failed for row=%s candidate=%s: %s", row.get("row_id"), candidate_id, exc)

        if payload is None:
            payload = normalize_counterfactual_payload(self._heuristic_payload(row, candidate_entry, context))

        payload.update(
            {
                "row_id": row.get("row_id"),
                "misconception_id": candidate_id,
                "raw_response": raw_text,
                "context_block": context.get("context_block", ""),
                "counterfactual_method": "llm_rewrite" if raw_text else "heuristic_rewrite",
            }
        )
        self.cache.save_json(cache_namespace, cache_key, payload)
        return payload