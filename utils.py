"""Shared helpers: logging, seeding, device selection, JSON persistence."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def setup_logging(log_file: str | os.PathLike | None = None, level: int = logging.INFO) -> None:
    """Configure root logging to stdout and (optionally) a log file."""
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    # Flower installs its own coloured handler; route its records through ours instead so
    # every line appears once on stdout and is also captured in the log file.
    flwr_logger = logging.getLogger("flwr")
    for handler in list(flwr_logger.handlers):
        flwr_logger.removeHandler(handler)
    flwr_logger.propagate = True
    flwr_logger.setLevel(level)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve 'auto' | 'cpu' | 'cuda' into a torch.device, failing loudly if CUDA is unavailable."""
    preference = preference.lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False.")
    if preference not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown device '{preference}'. Use auto, cpu or cuda.")
    return torch.device(preference)


def _sanitize(obj: Any) -> Any:
    """Replace NaN/inf with None so the output is strictly valid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def save_json(obj: Any, path: str | os.PathLike) -> None:
    """Write obj as pretty, strictly valid JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(obj), f, indent=2, allow_nan=False)


def load_json(path: str | os.PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
