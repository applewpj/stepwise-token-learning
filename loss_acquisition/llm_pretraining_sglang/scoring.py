from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_model_ref, run_dir_from_config, split_pythonpath
from .data import prepare_validation_data, read_jsonl_field


def prepend_pythonpath(*values: str | None) -> list[str]:
    entries: list[str] = []
    for value in values:
        entries.extend(split_pythonpath(value))
    entries = [str(Path(entry).resolve()) for entry in entries]
    for entry in reversed(entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    if entries:
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(entries + ([existing] if existing else []))
    return entries


def encode_inputs(items: list[Any], tokenizer_path: str | None, *, trust_remote_code: bool = False) -> list[list[int]]:
    if not items:
        return []
    if isinstance(items[0], list):
        return [[int(x) for x in item] for item in items]
    if tokenizer_path is None:
        raise ValueError("A tokenizer is required when the JSONL field contains strings.")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    return [tokenizer.encode(str(item), add_special_tokens=False) for item in items]


def extract_input_logprobs(item: dict[str, Any]) -> list[float]:
    meta = item.get("meta_info", item)
    values = meta.get("input_token_logprobs")
    if values is None:
        raise KeyError(
            "SGLang output does not contain meta_info.input_token_logprobs. "
            "Use return_logprob=True and logprob_start_len=0."
        )
    out: list[float] = []
    for value in values:
        if value is None:
            out.append(float("nan"))
        elif isinstance(value, dict):
            out.append(float(value.get("logprob", value.get("log_prob", float("nan")))))
        elif isinstance(value, (list, tuple)):
            logprob = value[0] if value else None
            out.append(float(logprob) if logprob is not None else float("nan"))
        else:
            out.append(float(value))
    return out


def normalize_outputs(outputs: Any) -> list[dict[str, Any]]:
    if isinstance(outputs, dict):
        return [outputs]
    return list(outputs)


def checkpoint_model_config(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    model_cfg = dict(config.get("model", {}) or {})
    merged = dict(model_cfg)
    if "model" in checkpoint:
        merged["path"] = checkpoint["model"]
    if "path" in checkpoint:
        merged["path"] = checkpoint["path"]
    if "tokenizer" in checkpoint:
        merged["tokenizer"] = checkpoint["tokenizer"]
    if "trust_remote_code" in checkpoint:
        merged["trust_remote_code"] = checkpoint["trust_remote_code"]
    if "path" in merged:
        merged["path"] = resolve_model_ref(merged["path"], base_dir)
    if "tokenizer" in merged:
        merged["tokenizer"] = resolve_model_ref(merged["tokenizer"], base_dir)
    return merged


def score_checkpoint(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> Path:
    run_dir = run_dir_from_config(config, base_dir)
    data_path = prepare_validation_data(config, base_dir=base_dir)
    data_cfg = dict(config.get("data", {}) or {})
    score_cfg = dict(config.get("score", {}) or {})
    sglang_cfg = dict(config.get("sglang", {}) or {})
    model_cfg = checkpoint_model_config(config, checkpoint, base_dir=base_dir)
    checkpoint_name = str(checkpoint.get("name", f"axis_{checkpoint['axis_value']}"))
    out_dir = run_dir / "checkpoint_losses" / checkpoint_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pythonpath_entries = prepend_pythonpath(
        sglang_cfg.get("sglang_python_path"),
        sglang_cfg.get("sgl_kernel_python_path"),
        sglang_cfg.get("extra_pythonpath"),
    )
    if "model_type" in sglang_cfg:
        os.environ["MODEL_TYPE"] = str(sglang_cfg["model_type"])

    import sglang as sgl

    text_field = str(data_cfg.get("text_field", "text"))
    items = read_jsonl_field(data_path, text_field, score_cfg.get("limit"))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    input_ids = encode_inputs(items, model_cfg.get("tokenizer", model_cfg.get("path")), trust_remote_code=trust_remote_code)
    engine_kwargs: dict[str, Any] = {
        "model_path": model_cfg["path"],
        "tokenizer_path": model_cfg.get("tokenizer", model_cfg.get("path")),
        "trust_remote_code": trust_remote_code,
        "tp_size": int(sglang_cfg.get("tp_size", 1)),
    }
    optional_engine_keys = [
        "dp_size",
        "context_length",
        "mem_fraction_static",
        "chunked_prefill_size",
        "attention_backend",
        "disable_cuda_graph",
    ]
    for key in optional_engine_keys:
        if key in sglang_cfg:
            engine_kwargs[key] = sglang_cfg[key]
    engine_kwargs.update(dict(sglang_cfg.get("engine_args", {}) or {}))
    llm = sgl.Engine(**{k: v for k, v in engine_kwargs.items() if v is not None})

    rows: list[np.ndarray] = []
    sampling_params = {
        "temperature": float(score_cfg.get("temperature", 0.0)),
        "max_new_tokens": int(score_cfg.get("max_new_tokens", 0)),
    }
    sampling_params.update(dict(score_cfg.get("sampling_args", {}) or {}))
    try:
        batch_size = int(score_cfg.get("batch_size", 8))
        for start in range(0, len(input_ids), batch_size):
            batch_ids = input_ids[start : start + batch_size]
            outputs = normalize_outputs(
                llm.generate(
                    input_ids=batch_ids,
                    sampling_params=sampling_params,
                    return_logprob=True,
                    logprob_start_len=0,
                )
            )
            if len(outputs) != len(batch_ids):
                raise RuntimeError(f"SGLang returned {len(outputs)} outputs for batch size {len(batch_ids)}")
            for local_idx, (ids, output) in enumerate(zip(batch_ids, outputs, strict=True)):
                sample_idx = start + local_idx
                logprobs = extract_input_logprobs(output)
                n_tokens = min(len(ids), len(logprobs))
                for token_pos in range(1, n_tokens):
                    logprob = logprobs[token_pos]
                    if not np.isfinite(logprob):
                        continue
                    rows.append(np.asarray([sample_idx, sample_idx, token_pos, -float(logprob)], dtype=np.float32))
    finally:
        if hasattr(llm, "shutdown"):
            llm.shutdown()

    if not rows:
        raise RuntimeError("No finite token losses were produced.")
    token_losses = np.stack(rows, axis=0).astype(np.float32)
    np.save(out_dir / "token_losses.npy", token_losses)
    np.savez_compressed(
        out_dir / "losses.single_checkpoint.npz",
        axis_values=np.asarray([float(checkpoint["axis_value"])], dtype=np.float64),
        loss_matrix=token_losses[:, 3:4].astype(np.float32),
        axis_name=np.asarray([score_cfg.get("axis_name", "training step")]),
        sample_id=token_losses[:, 1].astype(np.int64),
        token_pos=token_losses[:, 2].astype(np.int64),
    )
    manifest = {
        "checkpoint": checkpoint,
        "model_path": model_cfg["path"],
        "tokenizer_path": model_cfg.get("tokenizer", model_cfg.get("path")),
        "dataset": str(data_path),
        "axis_value": float(checkpoint["axis_value"]),
        "axis_name": score_cfg.get("axis_name", "training step"),
        "num_samples": len(input_ids),
        "num_token_losses": int(token_losses.shape[0]),
        "sglang_pythonpath_entries": pythonpath_entries,
        "model_type_env": os.environ.get("MODEL_TYPE"),
        "engine_kwargs": {k: v for k, v in engine_kwargs.items() if k not in {"model_path", "tokenizer_path"}},
        "sampling_params": sampling_params,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir
