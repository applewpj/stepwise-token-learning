#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "examples" / "manifest.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "examples" / "previews"
CORE_FIELDS = {"axis_values", "loss_matrix", "axis_name"}


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or "files" not in data:
        raise ValueError(f"Invalid examples manifest: {path}")
    return data


def preview_rows(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False)).astype(np.int64)


def _arrow_column(values: np.ndarray) -> Any:
    import pyarrow as pa

    if values.dtype.kind in {"U", "S", "O"}:
        return pa.array([str(v) for v in values.tolist()], type=pa.string())
    return pa.array(values)


def preview_table(npz_path: Path, *, max_rows: int, seed: int) -> tuple[Any, dict[str, Any]]:
    import pyarrow as pa

    with np.load(npz_path, allow_pickle=False) as data:
        loss_matrix = data["loss_matrix"]
        n_rows, n_points = loss_matrix.shape
        rows = preview_rows(n_rows, max_rows, seed)
        columns: dict[str, Any] = {"row": pa.array(rows)}
        for key in data.files:
            if key in CORE_FIELDS:
                continue
            values = data[key]
            if values.ndim == 1 and values.shape[0] == n_rows:
                columns[key] = _arrow_column(values[rows])
        losses = np.ascontiguousarray(loss_matrix[rows], dtype=np.float32)
        offsets = np.arange(0, losses.size + 1, n_points, dtype=np.int32)
        columns["loss"] = pa.ListArray.from_arrays(pa.array(offsets), pa.array(losses.reshape(-1)))
        axis_name = str(data["axis_name"][0]) if "axis_name" in data.files else "axis"
        axis_info = {
            "axis_name": axis_name,
            "axis_values": np.asarray(data["axis_values"], dtype=np.float64),
            "n_tokens": int(n_rows),
            "preview_rows": int(rows.size),
        }
    return pa.table(columns), axis_info


def export_previews(
    manifest: dict[str, Any],
    *,
    data_root: Path,
    out_dir: Path,
    max_rows: int,
    seed: int,
) -> list[Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    axes: dict[str, list[Any]] = {
        "asset": [],
        "axis_name": [],
        "n_points": [],
        "n_tokens": [],
        "preview_rows": [],
        "axis_values": [],
    }
    for entry in manifest.get("files", []):
        asset = str(entry["id"])
        npz_path = data_root / str(entry["local_path"])
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing example matrix: {npz_path}. Download it with tools/download_examples.py.")
        table, axis_info = preview_table(npz_path, max_rows=max_rows, seed=seed)
        path = out_dir / f"{asset}.parquet"
        pq.write_table(table, path, compression="zstd")
        written.append(path)
        axes["asset"].append(asset)
        axes["axis_name"].append(axis_info["axis_name"])
        axes["n_points"].append(int(axis_info["axis_values"].size))
        axes["n_tokens"].append(axis_info["n_tokens"])
        axes["preview_rows"].append(axis_info["preview_rows"])
        axes["axis_values"].append(axis_info["axis_values"].tolist())
        print(f"wrote {path} ({table.num_rows} of {axis_info['n_tokens']} rows)")
    axes_path = out_dir / "axes.parquet"
    pq.write_table(pa.table(axes), axes_path, compression="zstd")
    written.append(axes_path)
    print(f"wrote {axes_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Parquet previews of the example loss matrices for the Hugging Face Dataset Viewer."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--data-root", default=str(REPO_ROOT), help="Directory containing each manifest local_path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--rows", type=int, default=10000, help="Maximum rows per preview; smaller matrices are exported in full.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random row sample.")
    args = parser.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Exporting previews requires pyarrow: pip install pyarrow") from exc

    export_previews(
        load_manifest(Path(args.manifest)),
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        max_rows=int(args.rows),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
