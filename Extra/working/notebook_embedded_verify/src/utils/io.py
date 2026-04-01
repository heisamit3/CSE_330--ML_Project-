from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    target = Path(path)
    if target.is_absolute():
        return target
    if base is None:
        return Path.cwd() / target
    return Path(base) / target


def first_existing_path(candidates: str | Path | list[str | Path] | tuple[str | Path, ...] | None, base: str | Path | None = None) -> Path | None:
    if candidates is None:
        return None
    if isinstance(candidates, (str, Path)):
        candidate_values: list[str | Path] = [candidates]
    else:
        candidate_values = list(candidates)

    for candidate in candidate_values:
        if candidate is None:
            continue
        target = Path(candidate)
        if not target.is_absolute() and base is not None:
            target = Path(base) / target
        if target.exists():
            return target
    return None


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    with target.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
    return target


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(data: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return target


def write_text(text: str, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return target


def load_dataframe(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return pd.read_parquet(target)
    if target.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(target)
    return pd.read_csv(target, encoding="utf-8-sig")


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    if target.suffix.lower() == ".parquet":
        df.to_parquet(target, index=index)
    elif target.suffix.lower() in {".pkl", ".pickle"}:
        df.to_pickle(target)
    else:
        df.to_csv(target, index=index)
    return target
