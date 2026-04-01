"""Inspired by RAG (Lewis et al., 2020), FaR calibration ideas, and retrieval-reranking work such as MiRAGE.

Original idea: retrieve evidence, reason over candidates, then rerank with confidence-aware signals.
Simplified version here: keep top baseline candidates, attach bounded ontology/graph context, request strict JSON,
and store extra neuro-symbolic features for downstream weighted fusion or XGBoost reranking.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.data.load_data import build_input_text
from src.graphrag.prompt_templates import (
    baseline_scoring_prompt,
    json_repair_prompt,
    knowledge_guided_scoring_prompt,
)
from src.utils.cache import DiskCache, make_cache_key
from src.utils.io import save_dataframe
from src.utils.json_utils import (
    heuristic_repair_json,
    normalize_scoring_payload,
    parse_json_maybe,
)
from src.utils.text import lexical_overlap_score, tokenize


CACHE_VERSION = "oamllm_cv_v2"


class CandidateScorer:
    def __init__(self, config: dict[str, Any], retriever, backend, cache: DiskCache, logger):
        self.config = config
        self.retriever = retriever
        self.backend = backend
        self.cache = cache
        self.logger = logger

    def _heuristic_payload(self, row: pd.Series, candidate_entry: dict[str, Any], context: dict[str, Any], baseline_score: float) -> dict[str, Any]:
        row_text = build_input_text(row)
        trigger_score, trigger_matches = lexical_overlap_score(row_text, candidate_entry.get("trigger_patterns", []))
        candidate_tokens = set(
            tokenize(
                " ".join(
                    [
                        candidate_entry.get("misconception_name", ""),
                        candidate_entry.get("short_description", ""),
                        " ".join(candidate_entry.get("aliases", [])),
                        " ".join(candidate_entry.get("concept_tags", [])),
                        " ".join(candidate_entry.get("cognitive_roots", [])),
                    ]
                )
            )
        )
        row_tokens = set(tokenize(row_text))
        lexical_score = len(candidate_tokens & row_tokens) / max(1, len(candidate_tokens)) if candidate_tokens else 0.0
        ontology_support_score = float(context.get("ontology_support_score", 0.0))
        concept_score = float(context.get("concept_score", 0.0))
        subgraph_density = float(context.get("subgraph_density", 0.0))
        affinity = min(
            1.0,
            max(0.0, 0.32 * baseline_score + 0.25 * trigger_score + 0.18 * ontology_support_score + 0.15 * lexical_score + 0.10 * concept_score + 0.05 * subgraph_density),
        )
        confidence = min(
            1.0,
            max(0.0, 0.24 + 0.25 * trigger_score + 0.20 * ontology_support_score + 0.12 * baseline_score + 0.10 * concept_score + 0.09 * subgraph_density),
        )
        evidence_spans = trigger_matches[:2] if trigger_matches else [str(row.get("StudentExplanation", ""))[:100]]
        if trigger_matches:
            reason = "Student wording matches ontology trigger patterns."
        elif context.get("matched_concept_terms"):
            reason = "Student wording overlaps with concept and prerequisite terms linked to this misconception."
        else:
            reason = "Weak direct evidence; score relies on baseline retrieval and ontology profile overlap."
        return {
            "misconception_id": candidate_entry.get("misconception_id", ""),
            "affinity_score": round(affinity, 4),
            "confidence": round(confidence, 4),
            "reasoning_short": reason,
            "evidence_spans": evidence_spans,
        }

    def _parse_response(self, raw_text: str, expected_id: str) -> dict[str, Any] | None:
        parsed = parse_json_maybe(raw_text)
        if parsed is None and self.config.get("llm", {}).get("json_repair", True):
            repaired = parse_json_maybe(heuristic_repair_json(raw_text))
            if repaired is not None:
                parsed = repaired
            else:
                try:
                    repaired_text = self.backend.generate(json_repair_prompt(raw_text))
                    parsed = parse_json_maybe(repaired_text)
                except Exception as exc:
                    self.logger.warning("JSON repair backend call failed for %s: %s", expected_id, exc)
        if parsed is None:
            return None
        return normalize_scoring_payload(parsed, expected_id=expected_id)

    def score_single_candidate(
        self,
        row: pd.Series,
        candidate_id: str,
        baseline_score: float = 0.0,
        use_kb_context: bool = True,
        cache_namespace: str = "kb_scores",
    ) -> dict[str, Any]:
        cache_key = make_cache_key(
            {
                "row_id": row.get("row_id"),
                "candidate_id": candidate_id,
                "use_kb_context": use_kb_context,
                "student_explanation": row.get("StudentExplanation", ""),
                "namespace": cache_namespace,
                "cache_version": CACHE_VERSION,
            }
        )
        if self.config.get("runtime", {}).get("resume", True):
            cached = self.cache.load_json(cache_namespace, cache_key)
            if cached is not None:
                return cached

        context = self.retriever.retrieve(
            row,
            candidate_id,
            top_k_neighbors=int(self.config.get("graphrag", {}).get("top_k_context_neighbors", 3)),
            max_trigger_matches=int(self.config.get("graphrag", {}).get("max_trigger_matches", 3)),
        )
        candidate_entry = self.retriever.get_candidate_entry(candidate_id)
        prompt = (
            knowledge_guided_scoring_prompt(row, candidate_entry, context["context_block"])
            if use_kb_context
            else baseline_scoring_prompt(row, candidate_entry)
        )

        raw_text = ""
        parsed = None
        try:
            if self.backend.backend_name == "mock" and not self.config.get("llm", {}).get("enabled", False):
                raw_text = json.dumps(self._heuristic_payload(row, candidate_entry, context, baseline_score), ensure_ascii=False)
            else:
                raw_text = self.backend.generate(prompt)
            parsed = self._parse_response(raw_text, expected_id=candidate_id)
        except Exception as exc:
            self.logger.warning("Candidate scoring failed for row=%s candidate=%s: %s", row.get("row_id"), candidate_id, exc)

        if parsed is None:
            parsed = normalize_scoring_payload(self._heuristic_payload(row, candidate_entry, context, baseline_score), expected_id=candidate_id)

        result = {
            **parsed,
            "row_id": row.get("row_id"),
            "baseline_score": float(baseline_score),
            "kb_score": float(parsed.get("affinity_score", 0.0)),
            "graph_neighbor_count": context.get("graph_neighbor_count", 0),
            "trigger_match_count": context.get("trigger_match_count", 0),
            "concept_match_count": context.get("concept_match_count", 0),
            "trigger_score": float(context.get("trigger_score", 0.0)),
            "concept_score": float(context.get("concept_score", 0.0)),
            "ontology_support_score": float(context.get("ontology_support_score", 0.0)),
            "subgraph_node_count": context.get("subgraph_node_count", 0),
            "subgraph_edge_count": context.get("subgraph_edge_count", 0),
            "subgraph_density": float(context.get("subgraph_density", 0.0)),
            "matched_trigger_patterns": " | ".join(context.get("matched_trigger_patterns", [])),
            "matched_concept_terms": " | ".join(context.get("matched_concept_terms", [])),
            "neighbor_relation_types": " | ".join(context.get("neighbor_relation_types", [])),
            "context_block": context.get("context_block", ""),
            "raw_response": raw_text,
            "prompt_type": "knowledge_guided" if use_kb_context else "baseline_prompt",
            "scoring_backend": self.backend.backend_name,
        }
        self.cache.save_json(cache_namespace, cache_key, result)
        return result

    def score_dataframe(
        self,
        df: pd.DataFrame,
        baseline_candidates: pd.DataFrame,
        use_kb_context: bool = True,
        partial_save_path: str | None = None,
        cache_namespace: str = "kb_scores",
    ) -> pd.DataFrame:
        top_n = int(self.config.get("graphrag", {}).get("retrieve_top_n", self.config.get("data", {}).get("top_k_candidates", 8)))
        save_every = int(self.config.get("runtime", {}).get("save_every", 25))
        row_lookup = {row["row_id"]: row for _, row in df.iterrows()}
        results: list[dict[str, Any]] = []
        sorted_candidates = baseline_candidates.sort_values(["row_id", "baseline_score"], ascending=[True, False])

        for row_id, group in sorted_candidates.groupby("row_id"):
            row = row_lookup.get(row_id)
            if row is None:
                continue
            for _, candidate_row in group.head(top_n).iterrows():
                scored = self.score_single_candidate(
                    row=row,
                    candidate_id=str(candidate_row["misconception_id"]),
                    baseline_score=float(candidate_row.get("baseline_score", 0.0)),
                    use_kb_context=use_kb_context,
                    cache_namespace=cache_namespace,
                )
                results.append(scored)
            if partial_save_path and len(results) % max(1, save_every) == 0:
                save_dataframe(pd.DataFrame(results), partial_save_path, index=False)

        scored_df = pd.DataFrame(results)
        if partial_save_path and not scored_df.empty:
            save_dataframe(scored_df, partial_save_path, index=False)
        return scored_df