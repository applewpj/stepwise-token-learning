from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import resolve_path, run_dir_from_config


def read_jsonl_field(path: str | Path, text_field: str, limit: int | None = None) -> list[Any]:
    rows: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj[text_field])
    return rows


def prepare_validation_data(config: dict[str, Any], *, base_dir: Path | None = None) -> Path:
    data_cfg = dict(config.get("data", {}) or {})
    run_dir = run_dir_from_config(config, base_dir)
    prepared_path = resolve_path(data_cfg.get("prepared_jsonl", run_dir / "data" / "validation.jsonl"), base_dir)
    source = str(data_cfg.get("source", "jsonl")).lower()
    text_field = str(data_cfg.get("text_field", "text"))
    limit = data_cfg.get("limit")
    limit_int = None if limit is None else int(limit)
    overwrite = bool(data_cfg.get("overwrite", False))
    if prepared_path.exists() and not overwrite:
        return prepared_path
    prepared_path.parent.mkdir(parents=True, exist_ok=True)

    if source == "jsonl":
        input_path = resolve_path(data_cfg["path"], base_dir)
        texts = read_jsonl_field(input_path, text_field, limit_int)
    elif source in {"huggingface", "hf"}:
        from datasets import load_dataset

        dataset_name = str(data_cfg["dataset"])
        subset = data_cfg.get("subset")
        split = str(data_cfg.get("split", "validation"))
        kwargs = dict(data_cfg.get("load_kwargs", {}) or {})
        if subset is None:
            dataset = load_dataset(dataset_name, split=split, **kwargs)
        else:
            dataset = load_dataset(dataset_name, str(subset), split=split, **kwargs)
        texts = []
        for row in dataset:
            texts.append(row[text_field])
            if limit_int is not None and len(texts) >= limit_int:
                break
    else:
        raise ValueError(f"Unsupported data.source={source!r}")

    with prepared_path.open("w", encoding="utf-8") as f:
        for idx, text in enumerate(texts):
            f.write(json.dumps({"id": idx, text_field: text}, ensure_ascii=False) + "\n")
    return prepared_path
