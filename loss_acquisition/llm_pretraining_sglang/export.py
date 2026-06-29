from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenlearning.io.loss_matrix import save_loss_npz
from tokenlearning.types import LossMatrix


def list_dataset_loss_paths(run_dir: Path, loss_subdir: str) -> dict[str, Path]:
    direct_path = run_dir / "token_losses.npy"
    if direct_path.is_file():
        return {"default": direct_path}
    root = run_dir / loss_subdir
    if not root.is_dir():
        raise FileNotFoundError(f"Missing loss directory: {root}")
    paths = {path.parent.name: path for path in sorted(root.glob("*/token_losses.npy"))}
    if not paths:
        raise FileNotFoundError(f"No token_losses.npy files under {root}")
    return paths


def build_loss_npz_from_runs(
    run_specs: list[tuple[float, Path]],
    out_path: str | Path,
    *,
    axis_name: str,
    loss_subdir: str = "losses/all",
    token_limit: int | None = None,
    dataset_limit: int | None = None,
) -> Path:
    run_specs = sorted(run_specs, key=lambda item: item[0])
    axis_values = np.array([item[0] for item in run_specs], dtype=np.float64)
    path_maps = [list_dataset_loss_paths(item[1], loss_subdir) for item in run_specs]
    dataset_names = sorted(path_maps[0])
    for path_map in path_maps[1:]:
        missing = sorted(set(dataset_names) - set(path_map))
        if missing:
            raise ValueError(f"Run is missing datasets: {missing[:5]}")
    if dataset_limit is not None:
        dataset_names = dataset_names[: int(dataset_limit)]

    loss_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    token_pos_parts: list[np.ndarray] = []
    kept_total = 0
    limit = None if token_limit is None else int(token_limit)
    for dataset_idx, dataset_name in enumerate(dataset_names):
        if limit is not None and kept_total >= limit:
            break
        arrays = [np.load(path_map[dataset_name], mmap_mode="r") for path_map in path_maps]
        n_rows = int(arrays[0].shape[0])
        if limit is not None:
            n_rows = min(n_rows, limit - kept_total)
        if n_rows <= 0:
            continue
        base_keys = np.asarray(arrays[0][:n_rows, :3], dtype=np.int64)
        matrix = np.empty((n_rows, len(arrays)), dtype=np.float32)
        for run_idx, arr in enumerate(arrays):
            if arr.ndim != 2 or arr.shape[1] < 4:
                raise ValueError(f"Unexpected token loss shape for {dataset_name}: {arr.shape}")
            keys = np.asarray(arr[:n_rows, :3], dtype=np.int64)
            if not np.array_equal(keys, base_keys):
                raise ValueError(f"Token alignment mismatch in dataset {dataset_name}, run {run_idx}")
            matrix[:, run_idx] = np.asarray(arr[:n_rows, 3], dtype=np.float32)
        loss_parts.append(matrix)
        sample_parts.append((base_keys[:, 1] + dataset_idx * 10**12).astype(np.int64))
        token_pos_parts.append(base_keys[:, 2].astype(np.int64))
        kept_total += n_rows

    if not loss_parts:
        raise ValueError("No token rows were selected.")
    out = Path(out_path)
    save_loss_npz(
        out,
        LossMatrix(
            axis_values=axis_values,
            loss_matrix=np.concatenate(loss_parts, axis=0),
            axis_name=axis_name,
            sample_id=np.concatenate(sample_parts, axis=0),
            token_pos=np.concatenate(token_pos_parts, axis=0),
        ),
    )
    return out
