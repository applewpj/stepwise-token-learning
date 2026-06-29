from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RUN_DIR = "loss_acquisition/llm_pretraining_sglang/runs/default"


def find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or __file__).resolve()
    candidates = [current] + list(current.parents)
    for path in candidates:
        if (path / "tokenlearning").is_dir() and (path / "loss_acquisition").is_dir():
            return path
    return Path.cwd()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir or Path.cwd()) / path_obj


def resolve_model_ref(value: str | Path | None, base_dir: Path | None = None) -> str | None:
    if value is None:
        return None
    text = str(value)
    path = Path(text)
    if path.is_absolute() or text.startswith(("./", "../")):
        return str(resolve_path(path, base_dir))
    candidate = resolve_path(path, base_dir)
    if candidate.exists():
        return str(candidate)
    return text


def run_dir_from_config(config: dict[str, Any], base_dir: Path | None = None) -> Path:
    output_cfg = dict(config.get("output", {}) or {})
    return resolve_path(output_cfg.get("dir", DEFAULT_RUN_DIR), base_dir)


def parse_value(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_values(items: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Empty key in KEY=VALUE item: {item}")
        parsed[key.replace("-", "_")] = parse_value(value)
    return parsed


def split_pythonpath(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(os.pathsep) if item]
