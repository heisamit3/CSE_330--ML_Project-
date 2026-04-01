from __future__ import annotations

from src.data.load_data import build_input_text


def scoring_json_schema() -> str:
    return '{"misconception_id":"...","affinity_score":0.0,"confidence":0.0,"reasoning_short":"...","evidence_spans":["..."]}'


def baseline_scoring_prompt(row, candidate_entry: dict) -> str:
    return "\n".join(
        [
            "TASK=score_candidate",
            "Score whether the candidate misconception matches the student explanation.",
            "Do not reveal chain-of-thought. Keep reasoning_short brief.",
            "Use only the student explanation and candidate label. Lower confidence when evidence is weak.",
            build_input_text(row),
            f"Candidate misconception id: {candidate_entry.get('misconception_id', '')}",
            f"Candidate misconception name: {candidate_entry.get('misconception_name', '')}",
            "Return JSON only using this schema:",
            scoring_json_schema(),
        ]
    )


def knowledge_guided_scoring_prompt(row, candidate_entry: dict, context_block: str) -> str:
    return "\n".join(
        [
            "TASK=score_candidate",
            "You are a careful math misconception rater.",
            "Use the ontology profile and bounded graph neighborhood as supporting evidence, not as a substitute for the student text.",
            "If the ontology context and student explanation disagree, prioritize the student explanation and lower confidence.",
            "Do not reveal chain-of-thought. Keep reasoning_short concise and evidence-focused.",
            build_input_text(row),
            f"Candidate misconception id: {candidate_entry.get('misconception_id', '')}",
            f"Candidate misconception name: {candidate_entry.get('misconception_name', '')}",
            "Knowledge context:",
            context_block,
            "Evidence spans must quote or closely paraphrase short parts of the student explanation only.",
            "Return JSON only using this schema:",
            scoring_json_schema(),
        ]
    )


def counterfactual_rewrite_prompt(row, candidate_entry: dict, context_block: str) -> str:
    return "\n".join(
        [
            "TASK=counterfactual_rewrite",
            "Rewrite the explanation as if the suspected misconception were absent.",
            "Keep wording, tone, and length as similar as possible.",
            "Do not add a new misconception. Do not add chain-of-thought. Make the minimal necessary edit.",
            f"Original explanation: {row.get('StudentExplanation', '')}",
            f"Suspected misconception: {candidate_entry.get('misconception_name', candidate_entry.get('misconception_id', ''))}",
            context_block,
            "Return JSON only using this schema:",
            '{"counterfactual_text":"...","rewrite_confidence":0.0,"style_preservation_score":0.0,"explanation":"..."}',
        ]
    )


def json_repair_prompt(raw_text: str) -> str:
    return "\n".join(
        [
            "TASK=json_repair",
            "Repair the malformed JSON below.",
            "Return JSON only.",
            "Malformed JSON:",
            raw_text,
        ]
    )