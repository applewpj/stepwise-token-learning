from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml

from tokenlearning.config import load_config
from tokenlearning.io.loss_matrix import save_loss_npz
from tokenlearning.types import LossMatrix
from loss_acquisition.synthetic_mano._mano import (
    DEFAULT_VALUE_MOD,
    ManoSample,
    generate_sample,
    iter_seeds,
    length_weights,
    parse_jsonl_line,
    resolve_jsonl_files,
    tokens_to_expression,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICSLM_ROOT = REPO_ROOT / "external" / "PhysicsLM4"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        base_candidate = (base_dir / path).resolve()
        if base_candidate.exists():
            return base_candidate
    return (REPO_ROOT / path).resolve()


def _config_base(config: dict[str, Any]) -> Path:
    return Path(config.get("_config_path", ".")).resolve().parent


def _deep_set(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _deep_update(mapping: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if "." in key:
            _deep_set(mapping, key, value)
            continue
        old_value = mapping.get(key)
        if isinstance(old_value, dict) and isinstance(value, dict):
            _deep_update(old_value, value)
        else:
            mapping[key] = value


def _parse_lens_config(lens_config: Any) -> dict[int, float] | None:
    if lens_config is None:
        return None
    if not isinstance(lens_config, dict):
        raise ValueError(f"dataset_config.lens only supports dict or null, got {type(lens_config)}")
    items = [(int(k), v) for k, v in lens_config.items() if v is not None]
    if not items:
        raise ValueError("dataset_config.lens cannot be empty.")
    total = sum(float(v) for _, v in items)
    if total <= 0:
        raise ValueError("dataset_config.lens weights must sum to a positive value.")
    return {k: float(v) / total for k, v in items}


def _seed_count(seed_cfg: dict[str, Any], *, base_dir: Path) -> int:
    mode = str(seed_cfg.get("mode", "incremental"))
    if mode == "file":
        file_path = _resolve_path(seed_cfg["file_path"], base_dir=base_dir)
        with file_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip() and not line.lstrip().startswith("#"))
    length = int(seed_cfg.get("length", 1000))
    if mode == "incremental":
        return length
    return length * int(seed_cfg.get("repeat", 1))


def _iter_seed_config(seed_cfg: dict[str, Any], *, is_validation: bool, base_dir: Path) -> Iterator[int]:
    mode = str(seed_cfg.get("mode", "incremental"))
    if mode == "file":
        seed_file = _resolve_path(seed_cfg["file_path"], base_dir=base_dir)
        with seed_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield int(line)
        return

    if mode not in {"incremental", "epoch_ordered", "epoch_shuffled"}:
        raise ValueError(f"Unknown seed_generation.mode: {mode}")
    base_seed = int(seed_cfg.get("validation_start_seed", 1_000_000)) if is_validation else int(seed_cfg.get("base_seed", 0))
    yield from iter_seeds(
        length=int(seed_cfg.get("length", 1000)),
        repeat=int(seed_cfg.get("repeat", 1)),
        base_seed=base_seed,
        mode=mode,
        shuffle_seed=int(seed_cfg.get("shuffle_seed", 42)),
        reverse=bool(is_validation),
    )


def _payload(sample: ManoSample, *, include_seed: bool) -> dict[str, Any]:
    row: dict[str, Any] = {"text": sample.tokens}
    if include_seed:
        row["seed"] = sample.seed
    return row


def _write_seed(handle, seed: int) -> None:
    if handle is not None:
        handle.write(f"{int(seed)}\n")


def _write_samples(
    out_file: Path,
    seeds: Iterable[int],
    *,
    weights: dict[int, float],
    value_mod: int,
    ops: str,
    knowledge_augment: bool,
    include_seed: bool,
    seed_file: Path | None,
    noise_ratio: float,
    noise_weights: dict[int, float] | None,
    log_every: int,
    limit: int | None = None,
) -> int:
    if out_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing data file: {out_file}")
    if seed_file is not None and seed_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing seed file: {seed_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    seed_handle = None
    count = 0
    try:
        if seed_file is not None:
            seed_file.parent.mkdir(parents=True, exist_ok=True)
            seed_handle = seed_file.open("w", encoding="utf-8")
        with out_file.open("w", encoding="utf-8") as f:
            for seed in seeds:
                if limit is not None and count >= int(limit):
                    break
                seed = int(seed)
                _write_seed(seed_handle, seed)
                sample = generate_sample(
                    seed,
                    weights,
                    value_mod=value_mod,
                    ops=ops,
                    knowledge_augment=knowledge_augment,
                    noise_ratio=noise_ratio,
                    noise_weights=noise_weights,
                )
                f.write(json.dumps(_payload(sample, include_seed=include_seed), ensure_ascii=False) + "\n")
                count += 1
                if log_every > 0 and count % int(log_every) == 0:
                    print(f"wrote {count} samples to {out_file}", flush=True)
    finally:
        if seed_handle is not None:
            seed_handle.close()
    return count


def generate_from_seed_config(
    seed_config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    log_every: int = 1_000_000,
) -> dict[str, Any]:
    seed_config_path = Path(seed_config_path).resolve()
    cfg = _load_yaml(seed_config_path)
    dataset = str(cfg.get("dataset", "mano"))
    if dataset != "mano":
        raise ValueError(f"synthetic_mano only supports dataset=mano, got {dataset!r}")

    is_validation = bool(cfg.get("is_validation", False))
    dataset_cfg = cfg.get("dataset_config", {}) or {}
    mano_cfg = dataset_cfg.get("mano", dataset_cfg) if isinstance(dataset_cfg, dict) else {}
    lens_dist = _parse_lens_config(mano_cfg.get("lens"))
    weights = length_weights(explicit=lens_dist)
    noise_lens = _parse_lens_config(mano_cfg.get("noise_lens"))
    noise_weights = length_weights(explicit=noise_lens) if noise_lens is not None else None
    noise_ratio = float(mano_cfg.get("noise_ratio", 0.0))

    seed_cfg = cfg.get("seed_generation", {}) or {}
    out_cfg = cfg.get("output", {}) or {}
    out_dir = _resolve_path(output_dir or out_cfg["dir"], base_dir=seed_config_path.parent)
    include_seed = bool(out_cfg.get("include_seed", False))
    nchunks = int(out_cfg.get("nchunks", 8))
    data_dir = out_dir / dataset

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    seeds = _iter_seed_config(seed_cfg, is_validation=is_validation, base_dir=seed_config_path.parent)
    total = _seed_count(seed_cfg, base_dir=seed_config_path.parent)
    if is_validation:
        out_file = data_dir / f"{dataset}.val.jsonl"
        seed_file = data_dir / f"{dataset}.val.seeds.txt"
        count = _write_samples(
            out_file,
            seeds,
            weights=weights,
            value_mod=DEFAULT_VALUE_MOD,
            ops="asm",
            knowledge_augment=True,
            include_seed=include_seed,
            seed_file=seed_file,
            noise_ratio=noise_ratio,
            noise_weights=noise_weights,
            log_every=log_every,
            limit=total,
        )
        files = [out_file]
    else:
        if nchunks < 1:
            raise ValueError("output.nchunks must be >= 1.")
        chunk_size = total // nchunks
        remainder = total % nchunks
        files = []
        count = 0
        for idx in range(nchunks):
            n_this = chunk_size + (1 if idx < remainder else 0)
            out_file = data_dir / f"{dataset}.chunk.{idx:02d}.jsonl"
            seed_file = data_dir / f"{dataset}.seeds.txt" if idx == 0 else None
            count += _write_samples(
                out_file,
                seeds,
                weights=weights,
                value_mod=DEFAULT_VALUE_MOD,
                ops="asm",
                knowledge_augment=True,
                include_seed=include_seed,
                seed_file=seed_file,
                noise_ratio=noise_ratio,
                noise_weights=noise_weights,
                log_every=log_every,
                limit=n_this,
            )
            files.append(out_file)

    return {
        "seed_config": str(seed_config_path),
        "dataset": dataset,
        "is_validation": is_validation,
        "output_dir": str(out_dir),
        "files": [str(p) for p in files],
        "samples": int(count),
        "length_weights": weights,
        "include_seed": include_seed,
    }


def _configured_seed_outputs(cfg: dict[str, Any]) -> tuple[Path | None, Path | None]:
    base_dir = _config_base(cfg)
    data_cfg = cfg.get("data", {}) or {}
    root_dir = data_cfg.get("root_dir")
    if root_dir is None:
        return None, None
    root = _resolve_path(root_dir, base_dir=base_dir)
    train_name = str(data_cfg.get("train_name", "incremental_5120w"))
    validation_name = str(data_cfg.get("validation_name", "validation_10k"))
    return root / train_name, root / validation_name


def _seed_config_path(cfg: dict[str, Any], key: str) -> Path | None:
    base_dir = _config_base(cfg)
    data_cfg = cfg.get("data", {}) or {}
    value = data_cfg.get(key)
    if value is None:
        return None
    return _resolve_path(value, base_dir=base_dir)


def generate_mano_data(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {}) or {}
    overwrite = bool(data_cfg.get("overwrite", False))
    log_every = int(data_cfg.get("log_every_samples", 1_000_000))
    train_out, val_out = _configured_seed_outputs(cfg)

    generated: dict[str, Any] = {}
    train_seed_config = _seed_config_path(cfg, "train_seed_config")
    validation_seed_config = _seed_config_path(cfg, "validation_seed_config")

    if train_seed_config is not None:
        generated["train"] = generate_from_seed_config(
            train_seed_config,
            output_dir=data_cfg.get("train_output_dir") or train_out,
            overwrite=overwrite,
            log_every=log_every,
        )
    if validation_seed_config is not None:
        generated["validation"] = generate_from_seed_config(
            validation_seed_config,
            output_dir=data_cfg.get("validation_output_dir") or val_out,
            overwrite=overwrite,
            log_every=log_every,
        )

    if not generated:
        raise ValueError("Set data.train_seed_config and/or data.validation_seed_config in the synthetic Mano config.")

    out_dir = _resolve_path(cfg.get("output", {}).get("dir", "loss_acquisition/synthetic_mano/runs/synthetic_mano"), base_dir=_config_base(cfg))
    manifest_path = out_dir / "data_manifest.json"
    _write_json(manifest_path, generated)
    generated["manifest"] = str(manifest_path)
    return generated


def _physicslm_root(cfg: dict[str, Any]) -> Path:
    root = cfg.get("physicslm_root") or os.environ.get("PHYSICSLM_ROOT") or str(DEFAULT_PHYSICSLM_ROOT)
    root_path = _resolve_path(root, base_dir=_config_base(cfg))
    lingua = root_path / "lingua_modified"
    if not lingua.is_dir():
        raise FileNotFoundError(
            f"PhysicsLM root not found or missing lingua_modified: {root_path}. "
            "Set physicslm_root in the config or PHYSICSLM_ROOT in the environment."
        )
    return root_path


def _with_physicslm_pythonpath(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(root / "lingua_modified"), str(root)]
    old = env.get("PYTHONPATH")
    if old:
        paths.append(old)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _generated_train_root(cfg: dict[str, Any]) -> Path | None:
    data_cfg = cfg.get("data", {}) or {}
    if data_cfg.get("train_output_dir"):
        return _resolve_path(data_cfg["train_output_dir"], base_dir=_config_base(cfg))
    train_out, _ = _configured_seed_outputs(cfg)
    return train_out


def _generated_validation_file(cfg: dict[str, Any]) -> Path | None:
    data_cfg = cfg.get("data", {}) or {}
    if data_cfg.get("validation_file"):
        return _resolve_path(data_cfg["validation_file"], base_dir=_config_base(cfg))
    if data_cfg.get("validation_output_dir"):
        return _resolve_path(data_cfg["validation_output_dir"], base_dir=_config_base(cfg)) / "mano" / "mano.val.jsonl"
    _, val_out = _configured_seed_outputs(cfg)
    return None if val_out is None else val_out / "mano" / "mano.val.jsonl"


def prepare_train_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    base_dir = _config_base(cfg)
    train_cfg = cfg.get("train", {}) or {}
    template = train_cfg.get("config_template")
    if template is None:
        raise ValueError("train.config_template is required.")
    template_path = _resolve_path(template, base_dir=base_dir)
    physics_cfg = _load_yaml(template_path)
    out_dir = _resolve_path(cfg.get("output", {}).get("dir", "loss_acquisition/synthetic_mano/runs/synthetic_mano"), base_dir=base_dir)
    train_root = _generated_train_root(cfg)
    validation_file = _generated_validation_file(cfg)
    dump_dir = _resolve_path(train_cfg.get("dump_dir", out_dir / "outputs" / str(physics_cfg.get("name", "mano_training"))), base_dir=base_dir)

    auto_overrides: dict[str, Any] = {"dump_dir": str(dump_dir)}
    if train_root is not None:
        auto_overrides["data.root_dir"] = str(train_root)
    if validation_file is not None:
        auto_overrides["eval.mano_eval.val_file"] = str(validation_file)
        auto_overrides["eval.validation.root_dir"] = str(validation_file.parent.parent)
    _deep_update(physics_cfg, auto_overrides)
    _deep_update(physics_cfg, train_cfg.get("overrides", {}) or {})

    config_out = _resolve_path(train_cfg.get("config_out", out_dir / "train_config.yaml"), base_dir=base_dir)
    _write_yaml(config_out, physics_cfg)
    return {
        "train_config": str(config_out),
        "dump_dir": str(dump_dir),
        "checkpoint_dir": str(dump_dir / "checkpoints"),
    }


def train_mano_model(config_path: str | Path, *, generate_data: bool | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    should_generate = bool(data_cfg.get("generate", True)) if generate_data is None else bool(generate_data)
    if should_generate:
        generate_mano_data(config_path)
    prepared = prepare_train_config(config_path)
    if not bool(train_cfg.get("enabled", True)):
        return {**prepared, "skipped": True}

    root = _physicslm_root(cfg)
    command = train_cfg.get("command")
    if command is None:
        nproc = train_cfg.get("torchrun_nproc_per_node")
        if nproc:
            command = ["torchrun", "--nproc-per-node", str(nproc), "-m", "apps.main.train"]
        else:
            command = [sys.executable, "-m", "apps.main.train"]
    command = [str(x) for x in command] + [f"config={prepared['train_config']}"]
    subprocess.run(
        command,
        cwd=str(root / "lingua_modified"),
        env=_with_physicslm_pythonpath(root),
        check=True,
    )
    return {**prepared, "command": command}


def _add_physicslm_to_syspath(root: Path) -> None:
    for path in [root / "lingua_modified", root]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _step_dirs(ckpt_dir: Path, *, min_step: int | None = None, max_step: int | None = None, max_checkpoints: int | None = None) -> list[Path]:
    dirs: list[Path] = []
    for path in sorted(ckpt_dir.iterdir()):
        if not path.is_dir() or not path.name.isdigit():
            continue
        step = int(path.name)
        if min_step is not None and step < int(min_step):
            continue
        if max_step is not None and step > int(max_step):
            continue
        dirs.append(path)
    if max_checkpoints is not None:
        dirs = dirs[: int(max_checkpoints)]
    if not dirs:
        raise FileNotFoundError(f"No PhysicsLM step checkpoint directories found under {ckpt_dir}")
    return dirs


def _load_validation_samples(path: str | Path, *, limit: int | None = None) -> list[ManoSample]:
    samples: list[ManoSample] = []
    for file in resolve_jsonl_files(path):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                if limit is not None and len(samples) >= int(limit):
                    return samples
                sample = parse_jsonl_line(line, fallback_seed=len(samples))
                if sample is not None:
                    samples.append(sample)
    return samples


def _validation_metadata(samples: list[ManoSample], *, value_mod: int) -> dict[str, np.ndarray]:
    return {
        "difficulty": np.asarray([sample.difficulty for sample in samples], dtype=np.int64),
        "seed": np.asarray([sample.seed for sample in samples], dtype=np.int64),
        "question": np.asarray([tokens_to_expression(sample.tokens[2:-2], value_mod=value_mod)[0] for sample in samples]),
    }


def score_mano_checkpoints(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    score_cfg = cfg.get("score", {}) or {}
    out_dir = _resolve_path(cfg.get("output", {}).get("dir", "loss_acquisition/synthetic_mano/runs/synthetic_mano"), base_dir=_config_base(cfg))
    if not bool(score_cfg.get("enabled", True)):
        return {"output_dir": str(out_dir), "skipped": True}
    train_prepared = prepare_train_config(config_path)
    ckpt_dir = _resolve_path(score_cfg.get("checkpoint_dir", train_prepared["checkpoint_dir"]), base_dir=_config_base(cfg))
    validation_file = _resolve_path(score_cfg.get("validation_file", _generated_validation_file(cfg)), base_dir=_config_base(cfg))
    value_mod = int(score_cfg.get("value_mod", DEFAULT_VALUE_MOD))
    max_samples = score_cfg.get("max_samples")
    batch_size = int(score_cfg.get("batch_size", 32))

    root = _physicslm_root(cfg)
    _add_physicslm_to_syspath(root)
    from apps.main.eval import eval_mano_accuracy_per_sample_and_variance
    from apps.main.generate import load_consolidated_model_and_tokenizer
    from apps.main.transformer import LMTransformer, LMTransformerArgs
    from lingua.checkpoint import CONSOLIDATE_FOLDER, CONSOLIDATE_NAME, consolidate_checkpoints
    import torch

    samples = _load_validation_samples(validation_file, limit=max_samples)
    metadata = _validation_metadata(samples, value_mod=value_mod)
    difficulties = metadata["difficulty"]
    step_dirs = _step_dirs(
        ckpt_dir,
        min_step=score_cfg.get("min_step"),
        max_step=score_cfg.get("max_step"),
        max_checkpoints=score_cfg.get("max_checkpoints"),
    )

    records: list[dict[str, Any]] = []
    axis_values: list[int] = []
    loss_columns: list[np.ndarray] = []
    for step_dir in step_dirs:
        consolidated_dir = step_dir / CONSOLIDATE_FOLDER
        if not (consolidated_dir / CONSOLIDATE_NAME).exists():
            consolidate_checkpoints(str(step_dir))
        model, tokenizer, _model_cfg = load_consolidated_model_and_tokenizer(
            str(consolidated_dir),
            model_cls=LMTransformer,
            model_args_cls=LMTransformerArgs,
        )
        try:
            res = eval_mano_accuracy_per_sample_and_variance(
                model,
                tokenizer,
                val_file=str(validation_file),
                value_mod=value_mod,
                max_samples=max_samples,
                batch_size=batch_size,
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not res or res.get("per_sample_loss") is None:
            continue
        step = int(step_dir.name)
        per_sample = np.asarray(res["per_sample_loss"], dtype=np.float32)
        if per_sample.shape[0] != len(samples):
            raise ValueError(f"Step {step} produced {per_sample.shape[0]} losses for {len(samples)} validation samples.")
        axis_values.append(step)
        loss_columns.append(per_sample)
        records.append(
            {
                "step": step,
                "loss": float(res["loss"]),
                "accuracy": float(res["accuracy"]),
                "by_length": res.get("by_length", {}),
                "per_sample_loss": per_sample.tolist(),
            }
        )

    if not records:
        raise RuntimeError("No checkpoint loss records were produced.")
    order = np.argsort(np.asarray(axis_values))
    axis = np.asarray(axis_values, dtype=np.float64)[order]
    loss_matrix = np.stack(loss_columns, axis=1)[:, order].astype(np.float32)
    records = [records[int(i)] for i in order]

    out_dir.mkdir(parents=True, exist_ok=True)
    losses_path = out_dir / "losses.npz"
    save_loss_npz(
        losses_path,
        LossMatrix(
            axis_values=axis,
            loss_matrix=loss_matrix,
            axis_name=str(score_cfg.get("axis_name", "training step")),
            sample_id=np.arange(loss_matrix.shape[0], dtype=np.int64),
            token_pos=np.zeros(loss_matrix.shape[0], dtype=np.int64),
            metadata=metadata,
        ),
    )
    records_path = out_dir / "checkpoint_loss_records.json"
    records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path = out_dir / "metrics.mano.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({k: v for k, v in rec.items() if k != "per_sample_loss"}, ensure_ascii=False) + "\n")
    summary = {
        "validation_file": str(validation_file),
        "validation_samples": len(samples),
        "validation_difficulty_counts": {str(int(k)): int(np.sum(difficulties == k)) for k in sorted(np.unique(difficulties))},
        "losses_npz": str(losses_path),
        "records": str(records_path),
        "metrics": str(metrics_path),
        "checkpoints": [str(p) for p in step_dirs],
    }
    _write_json(out_dir / "summary.json", summary)
    return {"output_dir": str(out_dir), **summary}


def run_mano_experiment(config_path: str | Path) -> dict[str, Any]:
    generate_info = generate_mano_data(config_path)
    train_info = train_mano_model(config_path, generate_data=False)
    score_info = score_mano_checkpoints(config_path)
    return {"data": generate_info, "training": train_info, **score_info}
