from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from .io import ensure_dir


def make_cache_key(parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)

    def _path(self, namespace: str, key: str, suffix: str) -> Path:
        folder = ensure_dir(self.root / namespace)
        return folder / f"{key}{suffix}"

    def exists(self, namespace: str, key: str, suffix: str = ".json") -> bool:
        return self._path(namespace, key, suffix).exists()

    def load_json(self, namespace: str, key: str) -> Any:
        path = self._path(namespace, key, ".json")
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_json(self, namespace: str, key: str, payload: Any) -> Path:
        path = self._path(namespace, key, ".json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path

    def load_pickle(self, namespace: str, key: str) -> Any:
        path = self._path(namespace, key, ".pkl")
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def save_pickle(self, namespace: str, key: str, payload: Any) -> Path:
        path = self._path(namespace, key, ".pkl")
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path
