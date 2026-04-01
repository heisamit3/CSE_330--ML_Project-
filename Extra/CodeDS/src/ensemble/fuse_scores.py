from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_FEATURES = [
    "baseline_score",
    "kb_score",
    "confidence",
    "ontology_support_score",
    "trigger_score",
    "concept_score",
    "subgraph_density",
    "causal_delta",
    "relative_delta",
    "causal_confidence",
    "style_preservation_score",
    "graph_neighbor_count",
    "trigger_match_count",
    "concept_match_count",
]

COUNT_FEATURES = ["graph_neighbor_count", "trigger_match_count", "concept_match_count"]


def _ensure_columns(frame: pd.DataFrame, columns: list[str], id_columns: list[str]) -> pd.DataFrame:
    ensured = frame.copy()
    for column in columns:
        if column not in ensured.columns:
            ensured[column] = "" if column in id_columns else 0.0
    return ensured[columns]


def prepare_feature_table(
    baseline_df: pd.DataFrame,
    kb_df: pd.DataFrame | None = None,
    counterfactual_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    features = baseline_df[["row_id", "misconception_id", "baseline_score"]].copy()
    if kb_df is not None and not kb_df.empty:
        kb_cols = [
            "row_id",
            "misconception_id",
            "affinity_score",
            "confidence",
            "ontology_support_score",
            "trigger_score",
            "concept_score",
            "subgraph_density",
            "graph_neighbor_count",
            "trigger_match_count",
            "concept_match_count",
        ]
        kb_view = _ensure_columns(kb_df, kb_cols, ["row_id", "misconception_id"]).rename(columns={"affinity_score": "kb_score"})
        features = features.merge(kb_view, on=["row_id", "misconception_id"], how="left")
    if counterfactual_df is not None and not counterfactual_df.empty:
        cf_cols = [
            "row_id",
            "misconception_id",
            "causal_delta",
            "relative_delta",
            "causal_confidence",
            "style_preservation_score",
        ]
        cf_view = _ensure_columns(counterfactual_df, cf_cols, ["row_id", "misconception_id"])
        features = features.merge(cf_view, on=["row_id", "misconception_id"], how="left")

    for column in DEFAULT_FEATURES:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = features[column].fillna(0.0)

    for column in COUNT_FEATURES:
        max_value = max(1.0, float(features[column].max()))
        features[column] = features[column] / max_value
    return features


def weighted_fusion(feature_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    fused = feature_df.copy()
    fused["final_score"] = 0.0
    for feature_name, feature_weight in weights.items():
        if feature_name not in fused.columns:
            continue
        fused["final_score"] += fused[feature_name].fillna(0.0) * float(feature_weight)
    return fused


def fit_xgb_binary_reranker(feature_df: pd.DataFrame, truth_df: pd.DataFrame, logger=None):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        if logger:
            logger.warning("xgboost unavailable, skipping reranker: %s", exc)
        return None, DEFAULT_FEATURES

    if "target" not in truth_df.columns:
        if logger:
            logger.warning("truth_df has no target column, skipping xgboost reranker.")
        return None, DEFAULT_FEATURES

    training = feature_df.merge(truth_df[["row_id", "target"]], on="row_id", how="left")
    training = training.dropna(subset=["target"]).copy()
    if training.empty or training["row_id"].nunique() < 4:
        if logger:
            logger.warning("Not enough labeled rows to fit xgboost reranker.")
        return None, DEFAULT_FEATURES

    training["label"] = (training["misconception_id"] == training["target"]).astype(int)
    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(training[DEFAULT_FEATURES], training["label"])
    if logger:
        logger.info("Fitted optional XGBoost reranker on %s candidate rows.", len(training))
    return model, DEFAULT_FEATURES


def apply_xgb_binary_reranker(feature_df: pd.DataFrame, model, feature_columns: list[str] | None = None) -> pd.DataFrame:
    reranked = feature_df.copy()
    feature_columns = feature_columns or DEFAULT_FEATURES
    reranked["xgb_score"] = model.predict_proba(reranked[feature_columns])[:, 1]
    reranked["final_score"] = 0.7 * reranked["final_score"] + 0.3 * reranked["xgb_score"]
    return reranked


def fuse_score_tables(
    baseline_df: pd.DataFrame,
    kb_df: pd.DataFrame | None,
    counterfactual_df: pd.DataFrame | None,
    config: dict[str, Any],
    truth_df: pd.DataFrame | None = None,
    xgb_model=None,
    logger=None,
) -> pd.DataFrame:
    features = prepare_feature_table(baseline_df, kb_df=kb_df, counterfactual_df=counterfactual_df)
    weights = config.get("ensemble", {}).get("weights", {"baseline_score": 1.0})
    fused = weighted_fusion(features, weights)
    if xgb_model is not None:
        fused = apply_xgb_binary_reranker(fused, xgb_model)
    elif config.get("ensemble", {}).get("use_xgboost", False) and truth_df is not None:
        model, feature_columns = fit_xgb_binary_reranker(fused, truth_df, logger=logger)
        if model is not None:
            fused = apply_xgb_binary_reranker(fused, model, feature_columns)
    return fused.sort_values(["row_id", "final_score"], ascending=[True, False]).reset_index(drop=True)


def top_predictions(score_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    ranked = score_df.sort_values(["row_id", "final_score"], ascending=[True, False]).copy()
    ranked["rank"] = ranked.groupby("row_id")["final_score"].rank(method="first", ascending=False)
    top_df = ranked[ranked["rank"] <= top_k].copy()
    return top_df