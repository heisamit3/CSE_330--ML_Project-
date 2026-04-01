from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import ensure_dir, resolve_path


def is_kaggle_environment() -> bool:
    return Path("/kaggle/input").exists() and Path("/kaggle/working").exists()


def default_input_dir() -> Path:
    return Path("/kaggle/input") if is_kaggle_environment() else Path.cwd()


def default_working_dir(experiment_name: str = "map_mvp") -> Path:
    if is_kaggle_environment():
        return Path("/kaggle/working") / experiment_name
    return Path.cwd() / "working" / experiment_name


def resolve_runtime_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths_cfg = config.get("paths", {})
    experiment_name = config.get("experiment_name", "map_mvp")

    input_dir = resolve_path(paths_cfg.get("input_dir") or default_input_dir())
    working_dir = resolve_path(paths_cfg.get("working_dir") or default_working_dir(experiment_name))
    logs_dir = resolve_path(paths_cfg.get("logs_dir") or working_dir / "logs")
    cache_dir = resolve_path(paths_cfg.get("cache_dir") or working_dir / "cache")
    output_dir = resolve_path(paths_cfg.get("output_dir") or working_dir / "output")
    graph_path = resolve_path(paths_cfg.get("graph_path") or working_dir / "graph" / "graph.gexf")
    graph_summary_path = resolve_path(paths_cfg.get("graph_summary_path") or working_dir / "graph" / "graph_summary.json")
    kb_path = resolve_path(paths_cfg.get("kb_path") or Path("data") / "misconception_kb.json")
    baseline_artifacts_dir = resolve_path(paths_cfg.get("baseline_artifacts_dir") or working_dir / "baseline_artifacts")

    for directory in [working_dir, logs_dir, cache_dir, output_dir, graph_path.parent, baseline_artifacts_dir]:
        ensure_dir(directory)

    return {
        "input_dir": input_dir,
        "working_dir": working_dir,
        "logs_dir": logs_dir,
        "cache_dir": cache_dir,
        "output_dir": output_dir,
        "graph_path": graph_path,
        "graph_summary_path": graph_summary_path,
        "kb_path": kb_path,
        "baseline_artifacts_dir": baseline_artifacts_dir,
    }
