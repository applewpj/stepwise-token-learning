from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import find_repo_root, load_yaml, resolve_path, run_dir_from_config
from .data import prepare_validation_data
from .export import build_loss_npz_from_runs
from .scoring import score_checkpoint


def run_pipeline(config_path: str | Path, *, base_dir: Path | None = None) -> dict[str, Any]:
    config_path_obj = Path(config_path).resolve()
    config = load_yaml(config_path_obj)
    root = base_dir or find_repo_root(config_path_obj)
    run_dir = run_dir_from_config(config, root)
    data_path = prepare_validation_data(config, base_dir=root)
    checkpoint_dirs = []
    checkpoints = [dict(item) for item in config.get("checkpoints", [])]
    if not checkpoints:
        raise ValueError("Config must contain at least one checkpoint entry.")
    for checkpoint in checkpoints:
        checkpoint_dirs.append(score_checkpoint(config, checkpoint, base_dir=root))

    losses_path = None
    if bool(config.get("build", {}).get("enabled", True)) and len(checkpoint_dirs) >= 2:
        score_cfg = dict(config.get("score", {}) or {})
        build_cfg = dict(config.get("build", {}) or {})
        run_specs = [
            (float(checkpoint["axis_value"]), checkpoint_dir)
            for checkpoint, checkpoint_dir in zip(checkpoints, checkpoint_dirs, strict=True)
        ]
        losses_path = build_loss_npz_from_runs(
            run_specs,
            resolve_path(build_cfg.get("out", run_dir / "losses.npz"), root),
            axis_name=str(score_cfg.get("axis_name", "training step")),
            loss_subdir=str(build_cfg.get("loss_subdir", "losses/all")),
            token_limit=build_cfg.get("token_limit"),
            dataset_limit=build_cfg.get("dataset_limit"),
        )
    summary = {
        "output_dir": str(run_dir),
        "prepared_data": str(data_path),
        "checkpoint_loss_dirs": [str(path) for path in checkpoint_dirs],
        "losses_npz": None if losses_path is None else str(losses_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
