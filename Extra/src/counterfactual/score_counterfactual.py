from __future__ import annotations

import pandas as pd

from src.utils.io import save_dataframe


class CounterfactualScorer:
    def __init__(self, config: dict[str, Any], generator, candidate_scorer, logger):
        self.config = config
        self.generator = generator
        self.candidate_scorer = candidate_scorer
        self.logger = logger

    def _causal_verdict(self, causal_delta: float, relative_delta: float) -> str:
        min_abs = float(self.config.get("counterfactual", {}).get("min_absolute_delta_confirm", 0.15))
        min_rel = float(self.config.get("counterfactual", {}).get("min_relative_delta_confirm", 0.30))
        if causal_delta >= min_abs and relative_delta >= min_rel:
            return "confirmed"
        if causal_delta > 0:
            return "weak_support"
        return "spurious_or_unstable"

    def score(self, df: pd.DataFrame, kb_scores: pd.DataFrame, partial_save_path: str | None = None) -> pd.DataFrame:
        top_k = int(self.config.get("counterfactual", {}).get("top_k", 3))
        min_affinity = float(self.config.get("counterfactual", {}).get("min_affinity_to_run", 0.15))
        save_every = int(self.config.get("runtime", {}).get("save_every", 25))
        row_lookup = {row["row_id"]: row for _, row in df.iterrows()}
        results: list[dict[str, object]] = []

        ranked = kb_scores.sort_values(["row_id", "affinity_score"], ascending=[True, False])
        for row_id, group in ranked.groupby("row_id"):
            row = row_lookup.get(row_id)
            if row is None:
                continue
            for _, kb_row in group.head(top_k).iterrows():
                original_affinity = float(kb_row.get("affinity_score", 0.0))
                if original_affinity < min_affinity:
                    continue
                candidate_id = str(kb_row["misconception_id"])
                rewrite = self.generator.generate(row, candidate_id)
                modified_row = row.copy()
                modified_row["StudentExplanation"] = rewrite["counterfactual_text"]
                rescored = self.candidate_scorer.score_single_candidate(
                    row=modified_row,
                    candidate_id=candidate_id,
                    baseline_score=float(kb_row.get("baseline_score", 0.0)),
                    use_kb_context=True,
                    cache_namespace="counterfactual_scores",
                )
                counterfactual_affinity = float(rescored.get("affinity_score", 0.0))
                causal_delta = original_affinity - counterfactual_affinity
                relative_delta = causal_delta / max(0.05, original_affinity)
                verdict = self._causal_verdict(causal_delta, relative_delta)
                style_preservation_score = float(rewrite.get("style_preservation_score", 0.0))
                causal_confidence = max(
                    0.0,
                    min(
                        1.0,
                        ((float(kb_row.get("confidence", 0.0)) + float(rewrite.get("rewrite_confidence", 0.0))) / 2.0)
                        * (0.45 + max(0.0, causal_delta))
                        * (0.50 + 0.50 * style_preservation_score),
                    ),
                )
                results.append(
                    {
                        "row_id": row_id,
                        "misconception_id": candidate_id,
                        "original_affinity": original_affinity,
                        "counterfactual_affinity": counterfactual_affinity,
                        "causal_delta": causal_delta,
                        "relative_delta": relative_delta,
                        "causal_confidence": causal_confidence,
                        "causal_verdict": verdict,
                        "counterfactual_text": rewrite.get("counterfactual_text", ""),
                        "counterfactual_explanation": rewrite.get("explanation", ""),
                        "rewrite_confidence": float(rewrite.get("rewrite_confidence", 0.0)),
                        "style_preservation_score": style_preservation_score,
                        "counterfactual_method": rewrite.get("counterfactual_method", ""),
                    }
                )
            if partial_save_path and len(results) % max(1, save_every) == 0:
                save_dataframe(pd.DataFrame(results), partial_save_path, index=False)

        result_df = pd.DataFrame(results)
        if partial_save_path and not result_df.empty:
            save_dataframe(result_df, partial_save_path, index=False)
        return result_df