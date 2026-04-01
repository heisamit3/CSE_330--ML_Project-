from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.baseline.pipeline import BaselineCandidateGenerator
from src.counterfactual.generate_counterfactual import CounterfactualGenerator
from src.counterfactual.score_counterfactual import CounterfactualScorer
from src.data.load_data import load_competition_data, make_submission
from src.ensemble.fuse_scores import fuse_score_tables, prepare_feature_table, top_predictions
from src.eval.metrics import map_at_3_from_frames
from src.graphrag.retrieve_context import ContextRetriever
from src.graphrag.score_candidates import CandidateScorer
from src.llm.backends import build_backend
from src.ontology.build_kb import build_kb
from src.ontology.graph_utils import build_graph_from_kb, export_graph
from src.utils.cache import DiskCache
from src.utils.io import read_json, save_dataframe
from src.utils.kaggle_paths import resolve_runtime_paths
from src.utils.logging_utils import setup_logger


class ExperimentRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = deepcopy(config)
        self.paths = resolve_runtime_paths(self.config)
        self.logger = setup_logger(self.config.get("experiment_name", "experiment"), self.paths["logs_dir"])
        self.cache = DiskCache(self.paths["cache_dir"])
        self.backend = build_backend(self.config, self.logger)

    def load_data_bundle(self) -> dict[str, pd.DataFrame]:
        return load_competition_data(self.config, self.paths, self.logger)

    def ensure_artifacts(self, train_df: pd.DataFrame, candidate_df: pd.DataFrame) -> None:
        kb_path = self.paths["kb_path"]
        graph_path = self.paths["graph_path"]
        summary_path = self.paths["graph_summary_path"]

        if self.config.get("runtime", {}).get("force_rebuild_kb", False) or not kb_path.exists():
            self.logger.info("Building misconception KB at %s", kb_path)
            build_kb(train_df, candidate_df, kb_path)

        if self.config.get("runtime", {}).get("force_rebuild_graph", False) or not graph_path.exists():
            self.logger.info("Building ontology graph at %s", graph_path)
            entries = read_json(kb_path, default=[])
            graph = build_graph_from_kb(entries)
            export_graph(graph, graph_path, summary_path)

    def run_mode(
        self,
        train_df: pd.DataFrame,
        inference_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        mode: str | None = None,
        prefix: str = "run",
    ) -> dict[str, Any]:
        active_mode = mode or self.config.get("mode", "full")
        self.logger.info("Running mode=%s on %s inference rows.", active_mode, len(inference_df))

        baseline_generator = BaselineCandidateGenerator(self.config, self.paths, self.logger)
        baseline_generator.fit(train_df, candidate_df)
        baseline_df = baseline_generator.score_rows(inference_df)
        baseline_path = self.paths["output_dir"] / f"{prefix}_{active_mode}_baseline_candidates.csv"
        save_dataframe(baseline_df, baseline_path, index=False)

        kb_df = pd.DataFrame()
        cf_df = pd.DataFrame()
        if active_mode != "baseline":
            self.ensure_artifacts(train_df, candidate_df)
            retriever = ContextRetriever.from_artifacts(self.paths["kb_path"], self.paths["graph_path"])
            scorer = CandidateScorer(self.config, retriever, self.backend, self.cache, self.logger)
            kb_path = self.paths["output_dir"] / f"{prefix}_{active_mode}_kb_scores.csv"
            kb_df = scorer.score_dataframe(
                inference_df,
                baseline_df,
                use_kb_context=True,
                partial_save_path=str(kb_path),
                cache_namespace=f"kb_scores_{prefix}_{active_mode}",
            )
            if active_mode in {"counterfactual", "full"} and self.config.get("counterfactual", {}).get("enabled", False):
                generator = CounterfactualGenerator(self.config, retriever, self.backend, self.cache, self.logger)
                cf_runner = CounterfactualScorer(self.config, generator, scorer, self.logger)
                cf_path = self.paths["output_dir"] / f"{prefix}_{active_mode}_counterfactual.csv"
                cf_df = cf_runner.score(inference_df, kb_df, partial_save_path=str(cf_path))

        fused_df = fuse_score_tables(
            baseline_df=baseline_df,
            kb_df=kb_df if not kb_df.empty else None,
            counterfactual_df=cf_df if not cf_df.empty else None,
            config=self.config,
            truth_df=None,
            logger=self.logger,
        )
        feature_df = prepare_feature_table(baseline_df, kb_df if not kb_df.empty else None, cf_df if not cf_df.empty else None)
        feature_df["final_score"] = fused_df["final_score"]
        score_path = self.paths["output_dir"] / f"{prefix}_{active_mode}_scores.csv"
        save_dataframe(fused_df, score_path, index=False)

        top_df = top_predictions(fused_df, top_k=int(self.config.get("data", {}).get("top_k_submit", 3)))
        submission = make_submission(fused_df, top_k=int(self.config.get("data", {}).get("top_k_submit", 3)))
        submission_path = self.paths["output_dir"] / f"{prefix}_{active_mode}_submission.csv"
        save_dataframe(submission, submission_path, index=False)

        metric = None
        if "target" in inference_df.columns:
            metric = map_at_3_from_frames(inference_df, fused_df, truth_col="target")
            self.logger.info("MAP@3 for %s/%s = %.6f", prefix, active_mode, metric)

        return {
            "mode": active_mode,
            "baseline_df": baseline_df,
            "kb_df": kb_df,
            "cf_df": cf_df,
            "feature_df": feature_df,
            "fused_df": fused_df,
            "top_df": top_df,
            "submission_df": submission,
            "metric": metric,
            "paths": {
                "baseline": baseline_path,
                "scores": score_path,
                "submission": submission_path,
            },
        }
