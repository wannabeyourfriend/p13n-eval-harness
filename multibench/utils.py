"""Shared utilities used across all benchmarks."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_TRAILING_THINK_CLOSE = re.compile(r"</think>\s*")


def strip_think_tags(text: str) -> str:
    """Remove Qwen3-style <think>...</think> reasoning blocks from output."""
    if not text:
        return text
    text = _THINK_BLOCK.sub("", text)
    text = _TRAILING_THINK_CLOSE.sub("", text)
    return text.strip()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write JSON atomically so partial writes never corrupt saved progress."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, p)


def repo_root() -> Path:
    """Return the multi-bench repo root (parent of the multibench package)."""
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Return the data/ directory at the repo root."""
    return repo_root() / "data"


def benchmark_data_dir(name: str) -> Path:
    """Return data/<name>/. Raises if missing (expected to be a symlink to upstream)."""
    p = data_root() / name
    if not p.exists():
        raise FileNotFoundError(
            f"Missing data directory for benchmark '{name}' at {p}. "
            f"Create a symlink pointing to the upstream dataset."
        )
    return p
