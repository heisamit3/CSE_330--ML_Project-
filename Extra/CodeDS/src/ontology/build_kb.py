"""Inspired by Rittle-Johnson et al. (2025), neuro-symbolic KG work in education, and the MAP public writeups.

Original idea: use expert misconception labels plus structured conceptual links to support diagnosis.
Simplified version here: merge a manually curatable JSON seed with conservative heuristics extracted from
competition metadata and train responses. This keeps the ontology transparent, editable, and Kaggle-friendly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_data import load_competition_data
from src.utils.io import read_json, write_json
from src.utils.text import top_ngrams


REQUIRED_FIELDS = [
    "misconception_id",
    "misconception_name",
    "short_description",
    "prerequisite_concepts",
    "trigger_patterns",
    "formal_counterexample",
    "remediation_hint",
    "aliases",
    "concept_tags",
    "cognitive_roots",
    "graph_links",
    "evidence_examples",
    "source_notes",
]

TAG_RULES = [
    ("decimal", ["decimal", "place value", "compare decimals"]),
    ("fraction", ["fraction", "common denominator", "equivalent fractions"]),
    ("order_of_operations", ["order of operations", "operator precedence", "parentheses"]),
    ("algebra", ["equation", "variable", "algebra"]),
    ("ratio", ["ratio", "proportion", "rate"]),
    ("percentage", ["percent", "percentage"]),
    ("negative_numbers", ["negative", "minus", "sign"]),
    ("geometry", ["angle", "shape", "area", "perimeter"]),
]

ROOT_RULES = {
    "decimal": ["whole-number transfer", "place-value reasoning"],
    "fraction": ["part-whole reasoning", "structure-preserving arithmetic"],
    "order_of_operations": ["procedural sequencing", "symbol hierarchy"],
    "algebra": ["symbol interpretation", "equation balancing"],
    "ratio": ["multiplicative reasoning"],
    "percentage": ["part-whole reasoning", "scaling"],
    "negative_numbers": ["signed magnitude reasoning"],
    "geometry": ["spatial attribute mapping"],
}

COUNTEREXAMPLE_HINTS = {
    "decimal": "Rewrite both decimals with the same number of places, then compare from left to right.",
    "fraction": "Convert to equivalent fractions with a common denominator before combining them.",
    "order_of_operations": "Evaluate parentheses first, then multiplication/division, then addition/subtraction.",
    "algebra": "Check whether each operation preserves equality on both sides of the equation.",
    "ratio": "Compare multiplicative relationships rather than subtracting the two quantities.",
    "percentage": "Translate the percentage into a fraction or decimal before applying the operation.",
    "negative_numbers": "Use a number line or sign rules to verify the direction and magnitude of the result.",
}

REMEDIATION_HINTS = {
    "decimal": "Line up decimal places and compare place by place instead of comparing raw digit strings.",
    "fraction": "Focus on unit size and denominator meaning before adding or comparing fractions.",
    "order_of_operations": "Mark grouped operations first and verbalize the sequence before calculating.",
    "algebra": "State what each symbol represents, then justify each transformation step.",
    "ratio": "Describe the multiplicative relationship in words before computing.",
    "percentage": "Relate percent to out-of-100 meaning before simplifying.",
    "negative_numbers": "Say the sign rule aloud and check the result on a number line.",
}


def _normalize_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    normalized = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _lower_blob(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def _infer_concept_tags(misconception_id: str, misconception_name: str, description: str) -> list[str]:
    blob = _lower_blob(misconception_id, misconception_name, description)
    tags: list[str] = []
    for tag, keywords in TAG_RULES:
        if any(keyword in blob for keyword in keywords):
            tags.append(tag)
    return tags


def _infer_cognitive_roots(tags: list[str]) -> list[str]:
    roots: list[str] = []
    for tag in tags:
        for root in ROOT_RULES.get(tag, []):
            if root not in roots:
                roots.append(root)
    return roots


def _default_short_description(name: str, tags: list[str]) -> str:
    if tags:
        return f"Student reasoning appears consistent with the misconception '{name}' in the {tags[0]} concept area."
    return f"Student reasoning appears consistent with the misconception '{name}'."


def _default_counterexample(tags: list[str]) -> str:
    for tag in tags:
        if tag in COUNTEREXAMPLE_HINTS:
            return COUNTEREXAMPLE_HINTS[tag]
    return "Provide a minimal worked example that shows the intended rule and contrasts it with the mistaken rule."


def _default_remediation(tags: list[str]) -> str:
    for tag in tags:
        if tag in REMEDIATION_HINTS:
            return REMEDIATION_HINTS[tag]
    return "Ask the student to restate the rule in words and test it on a simple counterexample."


def _default_graph_links() -> dict[str, list[str]]:
    return {
        "causes": [],
        "exacerbates": [],
        "shares_cognitive_root_with": [],
    }


def _merge_entry(seed_entry: dict[str, Any] | None, generated_entry: dict[str, Any]) -> dict[str, Any]:
    if not seed_entry:
        return generated_entry
    merged = dict(generated_entry)
    for key, value in seed_entry.items():
        if key in {"trigger_patterns", "prerequisite_concepts", "aliases", "concept_tags", "cognitive_roots", "evidence_examples", "source_notes"}:
            merged[key] = _normalize_list(generated_entry.get(key, []))
            for item in _normalize_list(value):
                if item not in merged[key]:
                    merged[key].append(item)
        elif key == "graph_links":
            graph_links = _default_graph_links()
            for relation_name, relation_targets in generated_entry.get("graph_links", {}).items():
                graph_links[relation_name] = _normalize_list(relation_targets)
            for relation_name, relation_targets in (value or {}).items():
                graph_links.setdefault(relation_name, [])
                for target in _normalize_list(relation_targets):
                    if target not in graph_links[relation_name]:
                        graph_links[relation_name].append(target)
            merged[key] = graph_links
        else:
            seed_text = value if not isinstance(value, str) else value.strip()
            generated_text = generated_entry.get(key, "") if not isinstance(generated_entry.get(key, ""), str) else str(generated_entry.get(key, "")).strip()
            merged[key] = seed_text if seed_text not in [None, "", [], {}] else generated_text
    return merged


def validate_kb_entries(entries: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for index, entry in enumerate(entries):
        for field in REQUIRED_FIELDS:
            if field not in entry:
                issues.append(f"Entry {index} missing field '{field}'.")
        for list_field in [
            "prerequisite_concepts",
            "trigger_patterns",
            "aliases",
            "concept_tags",
            "cognitive_roots",
            "evidence_examples",
            "source_notes",
        ]:
            if list_field in entry and not isinstance(entry[list_field], list):
                issues.append(f"Entry {entry.get('misconception_id', index)} has non-list field '{list_field}'.")
        if "graph_links" in entry and not isinstance(entry["graph_links"], dict):
            issues.append(f"Entry {entry.get('misconception_id', index)} has non-dict field 'graph_links'.")
    return issues


def build_kb(train_df: pd.DataFrame, candidate_df: pd.DataFrame, output_path: str | Path) -> list[dict[str, Any]]:
    output_path = Path(output_path)
    existing_entries = read_json(output_path, default=[])
    existing_by_id = {entry.get("misconception_id"): entry for entry in existing_entries if entry.get("misconception_id")}

    entries: list[dict[str, Any]] = []
    grouped = train_df.groupby("target") if "target" in train_df.columns else None
    for _, candidate in candidate_df.iterrows():
        misconception_id = str(candidate.get("misconception_id", "")).strip()
        misconception_name = str(candidate.get("misconception_name", misconception_id)).strip()
        description = str(candidate.get("short_description", "") or "").strip()
        explanation_texts = []
        if grouped is not None and misconception_id in grouped.groups:
            explanation_texts = train_df.loc[grouped.groups[misconception_id], "StudentExplanation"].fillna("").astype(str).tolist()

        concept_tags = _infer_concept_tags(misconception_id, misconception_name, description)
        generated = {
            "misconception_id": misconception_id,
            "misconception_name": misconception_name,
            "family": str(candidate.get("family", "") or ""),
            "short_description": description or _default_short_description(misconception_name, concept_tags),
            "prerequisite_concepts": _normalize_list(candidate.get("prerequisite_concepts", [])),
            "trigger_patterns": top_ngrams(explanation_texts, top_n=6, min_count=1) if explanation_texts else [],
            "formal_counterexample": str(candidate.get("formal_counterexample", "") or "") or _default_counterexample(concept_tags),
            "remediation_hint": str(candidate.get("remediation_hint", "") or "") or _default_remediation(concept_tags),
            "aliases": _normalize_list([misconception_name] if misconception_name and misconception_name != misconception_id else []),
            "concept_tags": concept_tags,
            "cognitive_roots": _infer_cognitive_roots(concept_tags),
            "graph_links": _default_graph_links(),
            "evidence_examples": [text[:160].strip() for text in explanation_texts[:2] if text[:160].strip()],
            "support_count": len(explanation_texts),
            "source_notes": ["competition_metadata", "train_response_patterns"] if explanation_texts else ["competition_metadata"],
        }
        merged = _merge_entry(existing_by_id.get(misconception_id), generated)
        entries.append(merged)

    issues = validate_kb_entries(entries)
    if issues:
        raise ValueError("KB validation failed: " + " | ".join(issues[:10]))
    write_json(entries, output_path)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick_run.yaml")
    parser.add_argument("--output", default="data/misconception_kb.json")
    args = parser.parse_args()

    from src.utils.io import read_yaml
    from src.utils.kaggle_paths import resolve_runtime_paths
    from src.utils.logging_utils import setup_logger

    config = read_yaml(args.config)
    paths = resolve_runtime_paths(config)
    logger = setup_logger("build_kb", paths["logs_dir"])
    data_bundle = load_competition_data(config, paths, logger)
    entries = build_kb(data_bundle["train"], data_bundle["candidates"], args.output)
    logger.info("Wrote %s KB entries to %s", len(entries), args.output)


if __name__ == "__main__":
    main()