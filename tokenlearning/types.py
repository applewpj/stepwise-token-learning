from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LossMatrix:
    axis_values: np.ndarray
    loss_matrix: np.ndarray
    axis_name: str = "axis"
    sample_id: np.ndarray | None = None
    token_pos: np.ndarray | None = None
    metadata: dict[str, np.ndarray] | None = None

    def validate(self) -> None:
        axis = np.asarray(self.axis_values)
        losses = np.asarray(self.loss_matrix)
        if axis.ndim != 1:
            raise ValueError(f"axis_values must be 1D, got {axis.shape}")
        if losses.ndim != 2:
            raise ValueError(f"loss_matrix must be 2D, got {losses.shape}")
        if losses.shape[1] != axis.shape[0]:
            raise ValueError(
                "loss_matrix second dimension must match axis_values: "
                f"{losses.shape[1]} vs {axis.shape[0]}"
            )
        if axis.shape[0] < 2:
            raise ValueError("Need at least two axis points.")
        if np.any(np.diff(axis.astype(np.float64)) <= 0):
            raise ValueError("axis_values must be strictly increasing.")
        if self.sample_id is not None and len(self.sample_id) != losses.shape[0]:
            raise ValueError("sample_id length must match number of tokens.")
        if self.token_pos is not None and len(self.token_pos) != losses.shape[0]:
            raise ValueError("token_pos length must match number of tokens.")
        if self.metadata is not None:
            for name, value in self.metadata.items():
                array = np.asarray(value)
                if array.ndim == 0:
                    raise ValueError(f"metadata field {name!r} must be at least 1D.")
                if array.shape[0] != losses.shape[0]:
                    raise ValueError(
                        f"metadata field {name!r} length must match number of tokens: "
                        f"{array.shape[0]} vs {losses.shape[0]}"
                    )


@dataclass
class TokenEventFit:
    tau: np.ndarray
    lambda_: np.ndarray
    delta: np.ndarray
    start: np.ndarray
    end: np.ndarray
    r2: np.ndarray
    valid_mask: np.ndarray

    def validate(self) -> None:
        n = len(self.tau)
        for name in ("lambda_", "delta", "start", "end", "r2", "valid_mask"):
            value = getattr(self, name)
            if len(value) != n:
                raise ValueError(f"{name} length mismatch: {len(value)} vs {n}")

    @property
    def n_tokens(self) -> int:
        return int(len(self.tau))

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "tau": np.asarray(self.tau, dtype=np.float32),
            "lambda_": np.asarray(self.lambda_, dtype=np.float32),
            "delta": np.asarray(self.delta, dtype=np.float32),
            "start": np.asarray(self.start, dtype=np.float32),
            "end": np.asarray(self.end, dtype=np.float32),
            "r2": np.asarray(self.r2, dtype=np.float32),
            "valid_mask": np.asarray(self.valid_mask, dtype=bool),
        }

    @classmethod
    def from_npz(cls, data: dict[str, np.ndarray]) -> "TokenEventFit":
        if "delta" in data:
            delta = np.asarray(data["delta"], dtype=np.float32)
        else:
            delta = np.asarray(data["start"], dtype=np.float32) - np.asarray(data["end"], dtype=np.float32)
        if "valid_mask" in data:
            valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        else:
            tau = np.asarray(data["tau"], dtype=np.float32)
            valid_mask = np.isfinite(tau)
        fit = cls(
            tau=np.asarray(data["tau"], dtype=np.float32),
            lambda_=np.asarray(data["lambda_"], dtype=np.float32),
            delta=delta,
            start=np.asarray(data["start"], dtype=np.float32),
            end=np.asarray(data["end"], dtype=np.float32),
            r2=np.asarray(data["r2"], dtype=np.float32),
            valid_mask=valid_mask,
        )
        fit.validate()
        return fit
