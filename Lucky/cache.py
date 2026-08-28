"""Tiny pickle-backed cache so re-runs don't re-download everything."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class Cache:
    def __init__(self, root: Path, namespace: str) -> None:
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.dir / f"{safe}.pkl"

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def get(self, key: str, default: Any = None) -> Any:
        p = self.path(key)
        if not p.exists():
            return default
        try:
            with p.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # corrupt / partial write
            log.warning("cache read failed for %s (%s); ignoring", key, exc)
            return default

    def put(self, key: str, value: Any) -> None:
        p = self.path(key)
        tmp = p.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)

    def keys(self) -> list[str]:
        return [p.stem for p in self.dir.glob("*.pkl")]
