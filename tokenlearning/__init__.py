"""Public API for token-learning-spectrum."""

from tokenlearning.types import LossMatrix, TokenEventFit
from tokenlearning.io.loss_matrix import load_loss_npz, save_loss_npz
from tokenlearning.core.fitting import fit_token_events
from tokenlearning.core.sigmoid import sigmoid_loss, learning_pulse
from tokenlearning.core.spectrum import (
    compute_binned_learning_pulses,
    estimate_spectrum,
    filter_tokens,
    reconstruct_rate,
)

__all__ = [
    "LossMatrix",
    "TokenEventFit",
    "load_loss_npz",
    "save_loss_npz",
    "fit_token_events",
    "sigmoid_loss",
    "learning_pulse",
    "filter_tokens",
    "estimate_spectrum",
    "compute_binned_learning_pulses",
    "reconstruct_rate",
]
