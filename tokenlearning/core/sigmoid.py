from __future__ import annotations

import numpy as np

from tokenlearning.types import TokenEventFit


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def sigmoid_loss(
    axis_values: np.ndarray,
    tau: np.ndarray,
    lambda_: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    x = np.asarray(axis_values, dtype=np.float64)[None, :]
    tau_arr = np.asarray(tau, dtype=np.float64)[:, None]
    lam_arr = np.asarray(lambda_, dtype=np.float64)[:, None]
    start_arr = np.asarray(start, dtype=np.float64)[:, None]
    end_arr = np.asarray(end, dtype=np.float64)[:, None]
    delta = start_arr - end_arr
    return end_arr + delta * (1.0 - _sigmoid(lam_arr * (x - tau_arr)))


def mean_sigmoid_loss(
    axis_values: np.ndarray,
    fit: TokenEventFit,
    *,
    mask: np.ndarray | None = None,
    chunk_size: int = 50000,
) -> np.ndarray:
    if mask is None:
        indices = np.arange(fit.n_tokens)
    else:
        indices = np.flatnonzero(mask)
    if indices.size == 0:
        return np.full(len(axis_values), np.nan, dtype=np.float64)
    total = np.zeros(len(axis_values), dtype=np.float64)
    count = 0
    for start_idx in range(0, indices.size, int(chunk_size)):
        idx = indices[start_idx : start_idx + int(chunk_size)]
        curves = sigmoid_loss(axis_values, fit.tau[idx], fit.lambda_[idx], fit.start[idx], fit.end[idx])
        total += np.sum(curves, axis=0)
        count += idx.size
    return total / max(count, 1)


def learning_pulse(offset_values: np.ndarray, lambda_: np.ndarray, delta: np.ndarray) -> np.ndarray:
    offset = np.asarray(offset_values, dtype=np.float64)[None, :]
    lam = np.asarray(lambda_, dtype=np.float64)[:, None]
    delta_arr = np.asarray(delta, dtype=np.float64)[:, None]
    z = np.clip(0.5 * lam * offset, -40.0, 40.0)
    sech2 = 1.0 / (np.cosh(z) ** 2)
    return delta_arr * lam * 0.25 * sech2
