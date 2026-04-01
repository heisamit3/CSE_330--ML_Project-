from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.data.load_data import build_input_text


class BaselineCandidateGenerator:
    """Reconstructs a lightweight baseline path inspired by the 45th-place public writeup.

    Original idea: combine model probabilities with family filtering and simple voting.
    Simplified version here: use optional external baseline artifacts when available; otherwise
    build a transparent TF-IDF candidate retriever that preserves family filtering.
    Why simplified: the local repo is empty, so the exact original ensemble artifacts are not present.
    """

    def __init__(self, config: dict[str, Any], paths: dict[str, Path], logger):
        self.config = config
        self.paths = paths
        self.logger = logger
        self.vectorizer: TfidfVectorizer | None = None
        self.candidate_df = pd.DataFrame()
        self.candidate_matrix = None

    def _candidate_text(self, row: pd.Series) -> str:
        return " ".join(
            [
                str(row.get("misconception_id", "")),
                str(row.get("misconception_name", "")),
                str(row.get("short_description", "")),
            ]
        ).strip()

    def _load_external_artifacts(self) -> pd.DataFrame | None:
        baseline_cfg = self.config.get("baseline", {})
        if not baseline_cfg.get("use_external_artifacts", False):
            return None
        artifact_dir = self.paths["baseline_artifacts_dir"]
        if not artifact_dir.exists():
            return None

        artifact_glob = baseline_cfg.get("artifact_glob", "*.csv")
        parsed_frames: list[pd.DataFrame] = []
        for csv_path in artifact_dir.glob(artifact_glob):
            frame = pd.read_csv(csv_path)
            if {"row_id", "misconception_id", "baseline_score"}.issubset(frame.columns):
                parsed_frames.append(frame[["row_id", "misconception_id", "baseline_score"]].copy())
                continue
            if {"row_id", "prediction"}.issubset(frame.columns):
                exploded_rows = []
                for _, row in frame.iterrows():
                    for rank, misconception_id in enumerate(str(row["prediction"]).split()):
                        exploded_rows.append(
                            {
                                "row_id": row["row_id"],
                                "misconception_id": misconception_id,
                                "baseline_score": 1.0 / (rank + 1),
                            }
                        )
                if exploded_rows:
                    parsed_frames.append(pd.DataFrame(exploded_rows))
                continue
            label_columns = [column for column in frame.columns if column.startswith("label_")]
            prob_columns = [column for column in frame.columns if column.startswith("prob_")]
            if "row_id" in frame.columns and label_columns and prob_columns:
                records = []
                for _, row in frame.iterrows():
                    for label_column, prob_column in zip(sorted(label_columns), sorted(prob_columns)):
                        label_value = row.get(label_column)
                        if pd.isna(label_value):
                            continue
                        records.append(
                            {
                                "row_id": row["row_id"],
                                "misconception_id": label_value,
                                "baseline_score": float(row.get(prob_column, 0.0)),
                            }
                        )
                if records:
                    parsed_frames.append(pd.DataFrame(records))
        if not parsed_frames:
            return None
        combined = pd.concat(parsed_frames, ignore_index=True)
        combined = combined.groupby(["row_id", "misconception_id"], as_index=False)["baseline_score"].mean()
        self.logger.info("Loaded %s external baseline candidate rows.", len(combined))
        return combined

    def fit(self, train_df: pd.DataFrame, candidate_df: pd.DataFrame) -> None:
        self.candidate_df = candidate_df.copy().reset_index(drop=True)
        prototype_texts = self.candidate_df.apply(self._candidate_text, axis=1).tolist()
        train_texts = train_df.apply(build_input_text, axis=1).tolist() if not train_df.empty else []
        corpus = prototype_texts + train_texts if train_texts else prototype_texts
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.vectorizer.fit(corpus)
        self.candidate_matrix = self.vectorizer.transform(prototype_texts)
        self.logger.info("Baseline TF-IDF retriever fitted on %s candidates.", len(self.candidate_df))

    def score_rows(self, df: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
        external = self._load_external_artifacts()
        if external is not None:
            merged = external.merge(self.candidate_df, on="misconception_id", how="left")
            merged = merged.sort_values(["row_id", "baseline_score"], ascending=[True, False])
            merged["rank"] = merged.groupby("row_id")["baseline_score"].rank(method="first", ascending=False)
            return merged

        if self.vectorizer is None or self.candidate_matrix is None:
            raise RuntimeError("BaselineCandidateGenerator.fit must be called before score_rows.")

        requested_top_k = top_k or int(self.config.get("data", {}).get("top_k_candidates", 8))
        heuristic_pool = max(requested_top_k, int(self.config.get("baseline", {}).get("heuristic_candidate_pool", requested_top_k)))
        row_matrix = self.vectorizer.transform(df.apply(build_input_text, axis=1).tolist())
        similarity = linear_kernel(row_matrix, self.candidate_matrix)

        records = []
        for row_index, (_, row) in enumerate(df.iterrows()):
            candidate_view = self.candidate_df.copy()
            candidate_view["baseline_score"] = similarity[row_index]
            if self.config.get("baseline", {}).get("family_filter", True):
                family = str(row.get("family", "") or "")
                if family:
                    filtered = candidate_view[candidate_view["family"].fillna("").str.startswith(family)]
                    if not filtered.empty:
                        candidate_view = filtered
            candidate_view = candidate_view.sort_values("baseline_score", ascending=False).head(heuristic_pool).copy()
            candidate_view["baseline_score"] = candidate_view["baseline_score"].clip(lower=0.0, upper=1.0)
            candidate_view["row_id"] = row["row_id"]
            candidate_view["rank"] = range(1, len(candidate_view) + 1)
            candidate_view["baseline_source"] = "heuristic_tfidf"
            records.append(candidate_view)

        combined = pd.concat(records, ignore_index=True)
        return combined.sort_values(["row_id", "baseline_score"], ascending=[True, False]).reset_index(drop=True)
