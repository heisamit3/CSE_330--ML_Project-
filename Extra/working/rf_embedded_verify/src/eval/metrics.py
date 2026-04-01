from __future__ import annotations

from typing import Iterable

import pandas as pd


def average_precision_at_k(actual: str, predicted: Iterable[str], k: int = 3) -> float:
    predicted = list(predicted)[:k]
    score = 0.0
    num_hits = 0.0
    for index, prediction in enumerate(predicted, start=1):
        if prediction == actual and prediction not in predicted[: index - 1]:
            num_hits += 1.0
            score += num_hits / index
    return score


def mean_average_precision_at_k(actuals: list[str], predictions: list[list[str]], k: int = 3) -> float:
    if not actuals:
        return 0.0
    values = [average_precision_at_k(actual, predicted, k=k) for actual, predicted in zip(actuals, predictions)]
    return float(sum(values) / len(values))


def map_at_3_from_frames(truth_df: pd.DataFrame, score_df: pd.DataFrame, truth_col: str = "target") -> float:
    top3 = (
        score_df.sort_values(["row_id", "final_score"], ascending=[True, False])
        .groupby("row_id")["misconception_id"]
        .apply(lambda items: [str(item) for item in items.head(3) if pd.notna(item)])
        .reset_index()
    )
    merged = truth_df[["row_id", truth_col]].merge(top3, on="row_id", how="left")
    merged["misconception_id"] = merged["misconception_id"].apply(lambda value: value if isinstance(value, list) else [])
    return mean_average_precision_at_k(merged[truth_col].astype(str).tolist(), merged["misconception_id"].tolist(), k=3)


def build_error_analysis(truth_df: pd.DataFrame, score_df: pd.DataFrame, truth_col: str = "target") -> pd.DataFrame:
    ranked = score_df.sort_values(["row_id", "final_score"], ascending=[True, False]).copy()
    top1 = ranked.groupby("row_id").head(1)[["row_id", "misconception_id", "final_score"]]
    top3 = (
        ranked.groupby("row_id")["misconception_id"]
        .apply(lambda items: " ".join(str(item) for item in items.head(3) if pd.notna(item)))
        .reset_index()
    )
    merged = truth_df[["row_id", truth_col, "QuestionText", "StudentExplanation"]].merge(top1, on="row_id", how="left")
    merged = merged.merge(top3, on="row_id", how="left", suffixes=("_top1", "_top3"))
    merged = merged.rename(columns={"misconception_id_top1": "pred_top1", "final_score": "pred_top1_score", "misconception_id_top3": "pred_top3"})
    merged["is_top1_correct"] = (merged[truth_col].astype(str) == merged["pred_top1"].astype(str)).astype(int)
    return merged
