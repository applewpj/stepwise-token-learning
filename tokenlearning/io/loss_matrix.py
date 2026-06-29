from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenlearning.types import LossMatrix, TokenEventFit


_LOSS_RESERVED_KEYS = {"axis_values", "loss_matrix", "axis_name", "sample_id", "token_pos"}


def load_loss_npz(path: str | Path, *, axis_name: str | None = None, mmap_mode: str | None = None) -> LossMatrix:
    npz_path = Path(path)
    with np.load(npz_path, mmap_mode=mmap_mode) as data:
        if "axis_values" not in data or "loss_matrix" not in data:
            raise ValueError(f"{npz_path} must contain axis_values and loss_matrix")
        loaded_axis_name = axis_name
        if loaded_axis_name is None and "axis_name" in data:
            axis_value = np.asarray(data["axis_name"]).reshape(-1)
            loaded_axis_name = str(axis_value[0]) if axis_value.size else "axis"
        loss_matrix = np.asarray(data["loss_matrix"], dtype=np.float32)
        n_tokens = int(loss_matrix.shape[0])
        metadata: dict[str, np.ndarray] = {}
        for key in data.files:
            if key in _LOSS_RESERVED_KEYS:
                continue
            value = np.asarray(data[key])
            if value.ndim >= 1 and value.shape[0] == n_tokens:
                metadata[str(key)] = value
        result = LossMatrix(
            axis_values=np.asarray(data["axis_values"], dtype=np.float64),
            loss_matrix=loss_matrix,
            axis_name=loaded_axis_name or "axis",
            sample_id=np.asarray(data["sample_id"]) if "sample_id" in data else None,
            token_pos=np.asarray(data["token_pos"]) if "token_pos" in data else None,
            metadata=metadata or None,
        )
    result.validate()
    return result


def save_loss_npz(path: str | Path, loss_data: LossMatrix) -> None:
    loss_data.validate()
    payload: dict[str, np.ndarray] = {
        "axis_values": np.asarray(loss_data.axis_values, dtype=np.float64),
        "loss_matrix": np.asarray(loss_data.loss_matrix, dtype=np.float32),
        "axis_name": np.asarray([loss_data.axis_name]),
    }
    if loss_data.sample_id is not None:
        payload["sample_id"] = np.asarray(loss_data.sample_id)
    if loss_data.token_pos is not None:
        payload["token_pos"] = np.asarray(loss_data.token_pos)
    if loss_data.metadata is not None:
        for key, value in loss_data.metadata.items():
            key = str(key)
            if key in _LOSS_RESERVED_KEYS:
                raise ValueError(f"metadata field {key!r} conflicts with a reserved losses.npz key")
            payload[key] = np.asarray(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_fit_cache(
    path: str | Path,
    *,
    axis_values: np.ndarray,
    fit: TokenEventFit,
    raw_loss_mean: np.ndarray | None = None,
    fitted_loss_mean: np.ndarray | None = None,
) -> None:
    payload = fit.to_npz_payload()
    payload["axis_values"] = np.asarray(axis_values, dtype=np.float64)
    if raw_loss_mean is not None:
        payload["raw_loss_mean"] = np.asarray(raw_loss_mean, dtype=np.float64)
    if fitted_loss_mean is not None:
        payload["fitted_loss_mean"] = np.asarray(fitted_loss_mean, dtype=np.float64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_public_fit_cache(path: str | Path) -> tuple[np.ndarray, TokenEventFit, np.ndarray | None, np.ndarray | None]:
    data = np.load(path)
    axis_values = np.asarray(data["axis_values"], dtype=np.float64)
    fit = TokenEventFit.from_npz({key: data[key] for key in data.files})
    raw_loss_mean = np.asarray(data["raw_loss_mean"], dtype=np.float64) if "raw_loss_mean" in data else None
    fitted_loss_mean = np.asarray(data["fitted_loss_mean"], dtype=np.float64) if "fitted_loss_mean" in data else None
    return axis_values, fit, raw_loss_mean, fitted_loss_mean
