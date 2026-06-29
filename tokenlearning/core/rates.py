from __future__ import annotations

import numpy as np


def loss_change_rate(axis_values: np.ndarray, mean_loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(axis_values, dtype=np.float64)
    y = np.asarray(mean_loss, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"axis_values and mean_loss must be matching 1D arrays, got {x.shape}, {y.shape}")
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("axis_values must be strictly increasing.")
    rate = -np.diff(y) / dx
    if np.all(x[:-1] > 0):
        rate_x = np.sqrt(x[:-1] * x[1:])
    else:
        rate_x = 0.5 * (x[:-1] + x[1:])
    return rate_x, rate
