from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from src.data.load_data import build_candidate_table
from src.ensemble.fuse_scores import apply_xgb_binary_reranker, fit_xgb_binary_reranker
from src.eval.ablation_report import write_ablation_report
from src.eval.metrics import build_error_analysis, map_at_3_from_frames
from src.experiment import ExperimentRunner
from src.utils.io import save_dataframe


def _mode_config(base_config: dict[str, Any], mode: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["mode"] = mode
    config["experiment_name"] = f"{base_config.get('experiment_name', 'experiment')}_{mode}"
    config.setdefault("graphrag", {})
    config.setdefault("counterfactual", {})
    config.setdefault("ensemble", {})
    if mode == "baseline":
        config["graphrag"]["enabled"] = False
        config["counterfactual"]["enabled"] = False
        config["ensemble"]["use_xgboost"] = False
    elif mode == "kb_only":
        config["graphrag"]["enabled"] = True
        config["counterfactual"]["enabled"] = False
        config["ensemble"]["use_xgboost"] = False
    elif mode == "counterfactual":
        config["graphrag"]["enabled"] = True
        config["counterfactual"]["enabled"] = True
        config["ensemble"]["use_xgboost"] = False
    else:
        config["graphrag"]["enabled"] = True
        config["counterfactual"]["enabled"] = True
    return config


def _build_splitter(train_df: pd.DataFrame, config: dict[str, Any]):
    n_splits = int(config.get("data", {}).get("folds", 3))
    group_column = config.get("data", {}).get("group_column", "QuestionId")
    if group_column in train_df.columns and train_df[group_column].nunique() >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        return splitter.split(train_df, groups=train_df[group_column]), n_splits
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(config.get("seed", 42)))
    return splitter.split(train_df), n_splits


def run_cv(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    bootstrap_runner = ExperimentRunner(config)
    data_bundle = bootstrap_runner.load_data_bundle()
    train_df = data_bundle["train"].copy()
    base_candidates = data_bundle["candidates"].copy()
    if "target" not in train_df.columns:
        raise ValueError("Cross-validation requires labeled train rows with a target column.")

    split_iterator, n_splits = _build_splitter(train_df, config)
    metrics_records: list[dict[str, Any]] = []
    error_frames: list[pd.DataFrame] = []
    modes = ["baseline", "kb_only", "counterfactual", "full"]

    for fold_index, (train_idx, valid_idx) in enumerate(split_iterator):
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)
        fold_candidates = build_candidate_table(fold_train, base_candidates)

        for mode in modes:
            mode_config = _mode_config(config, mode)
            runner = ExperimentRunner(mode_config)
            train_result = runner.run_mode(fold_train, fold_train, fold_candidates, mode=mode, prefix=f"fold{fold_index}_train")
            valid_result = runner.run_mode(fold_train, fold_valid, fold_candidates, mode=mode, prefix=f"fold{fold_index}_valid")
            final_valid_scores = valid_result["fused_df"].copy()

            if mode == "full" and mode_config.get("ensemble", {}).get("use_xgboost", False):
                xgb_model, feature_columns = fit_xgb_binary_reranker(train_result["feature_df"], fold_train, logger=runner.logger)
                if xgb_model is not None:
                    reranked = valid_result["feature_df"].copy()
                    reranked["final_score"] = valid_result["fused_df"]["final_score"].values
                    final_valid_scores = apply_xgb_binary_reranker(reranked, xgb_model, feature_columns)
                    final_valid_scores = final_valid_scores.sort_values(["row_id", "final_score"], ascending=[True, False]).reset_index(drop=True)
                    save_dataframe(final_valid_scores, runner.paths["output_dir"] / f"fold{fold_index}_{mode}_reranked_scores.csv", index=False)

            fold_metric = map_at_3_from_frames(fold_valid, final_valid_scores, truth_col="target")
            metrics_records.append({"fold": fold_index, "mode": mode, "map3": fold_metric})

            error_df = build_error_analysis(fold_valid, final_valid_scores, truth_col="target")
            error_df["fold"] = fold_index
            error_df["mode"] = mode
            error_frames.append(error_df)
            runner.logger.info("Fold %s/%s mode=%s MAP@3=%.6f", fold_index + 1, n_splits, mode, fold_metric)

    metrics_df = pd.DataFrame(metrics_records)
    error_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    metrics_path = bootstrap_runner.paths["output_dir"] / "cv_metrics.csv"
    errors_path = bootstrap_runner.paths["output_dir"] / "cv_error_analysis.csv"
    summary_path = bootstrap_runner.paths["output_dir"] / "ablation_summary.csv"
    save_dataframe(metrics_df, metrics_path, index=False)
    if not error_df.empty:
        save_dataframe(error_df, errors_path, index=False)
    summary_df = write_ablation_report(metrics_df, str(summary_path), str(metrics_path))
    return {"metrics": metrics_df, "summary": summary_df, "errors": error_df}
