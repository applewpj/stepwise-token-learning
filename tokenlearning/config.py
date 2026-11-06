from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config.setdefault("_config_path", str(config_path.resolve()))
    return config


def deep_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def apply_overrides(
    config: dict[str, Any],
    *,
    device: str | None = None,
    epochs: int | None = None,
    max_tokens: int | None = None,
    subset_seed: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return a copy of ``config`` with command-line overrides applied."""
    result = copy.deepcopy(config)
    if device is not None:
        result.setdefault("fit", {})["device"] = str(device)
    if epochs is not None:
        result.setdefault("fit", {})["epochs"] = int(epochs)
    if max_tokens is not None:
        result.setdefault("input", {})["max_tokens"] = int(max_tokens)
    if subset_seed is not None:
        result.setdefault("input", {})["subset_seed"] = int(subset_seed)
    if output_dir is not None:
        result.setdefault("output", {})["dir"] = str(output_dir)
    return result
