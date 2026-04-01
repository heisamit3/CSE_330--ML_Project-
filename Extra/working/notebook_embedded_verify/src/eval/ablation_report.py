from __future__ import annotations

import pandas as pd

from src.utils.io import save_dataframe


def summarize_ablation(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["mode", "mean_map3", "std_map3", "folds"])
    summary = (
        metrics_df.groupby("mode")["map3"]
        .agg(mean_map3="mean", std_map3="std", folds="count")
        .reset_index()
    )
    return summary


def write_ablation_report(metrics_df: pd.DataFrame, summary_path: str, table_path: str) -> pd.DataFrame:
    summary = summarize_ablation(metrics_df)
    save_dataframe(metrics_df, table_path, index=False)
    save_dataframe(summary, summary_path, index=False)
    return summary