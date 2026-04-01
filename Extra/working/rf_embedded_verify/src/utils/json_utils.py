from __future__ import annotations

import ast
import json
import re
from typing import Any


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json_maybe(text: str) -> dict[str, Any] | None:
    cleaned = strip_code_fences(text)
    if not cleaned:
        return None
    candidates = [cleaned, extract_first_json_object(cleaned)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
        try:
            loaded = ast.literal_eval(candidate)
            if isinstance(loaded, dict):
                return loaded
        except (ValueError, SyntaxError):
            pass
    return None


def heuristic_repair_json(text: str) -> str:
    cleaned = strip_code_fences(text)
    candidate = extract_first_json_object(cleaned) or cleaned
    candidate = re.sub(r"(\w+)\s*:", r'"\1":', candidate)
    candidate = candidate.replace("'", '"')
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    return candidate


def bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))


def normalize_scoring_payload(payload: dict[str, Any], expected_id: str | None = None) -> dict[str, Any]:
    evidence = payload.get("evidence_spans", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [str(item).strip() for item in evidence if str(item).strip()]
    return {
        "misconception_id": str(payload.get("misconception_id") or expected_id or "").strip(),
        "affinity_score": bounded_float(payload.get("affinity_score"), 0.0),
        "confidence": bounded_float(payload.get("confidence"), 0.0),
        "reasoning_short": str(payload.get("reasoning_short", "")).strip(),
        "evidence_spans": evidence[:3],
    }


def normalize_counterfactual_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "counterfactual_text": str(payload.get("counterfactual_text", "")).strip(),
        "rewrite_confidence": bounded_float(payload.get("rewrite_confidence"), 0.0),
        "style_preservation_score": bounded_float(payload.get("style_preservation_score"), 0.0),
        "explanation": str(payload.get("explanation", "")).strip(),
    }