from __future__ import annotations

import argparse

from src.ontology.graph_utils import build_graph_from_kb, export_graph
from src.utils.io import read_json, read_yaml
from src.utils.kaggle_paths import resolve_runtime_paths
from src.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick_run.yaml")
    args = parser.parse_args()

    config = read_yaml(args.config)
    paths = resolve_runtime_paths(config)
    logger = setup_logger("build_graph", paths["logs_dir"])
    entries = read_json(paths["kb_path"], default=[])
    graph = build_graph_from_kb(entries)
    summary = export_graph(graph, paths["graph_path"], paths["graph_summary_path"])
    logger.info("Graph exported with %s nodes and %s edges.", summary["num_nodes"], summary["num_edges"])


if __name__ == "__main__":
    main()
