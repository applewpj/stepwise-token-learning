# Analyze Your Own Loss Matrix

The analysis package is agnostic to the model, tokenizer, and source of the
losses. It requires the loss of the same contextualized token rows measured at
multiple ordered points along one scaling axis.

## NPZ Schema

Create a compressed NumPy archive with:

```text
axis_values: float array [K]
loss_matrix: float array [N, K]
axis_name: optional string array [1]
sample_id: optional array [N]
token_pos: optional array [N]
<metadata_name>: optional array [N]
```

`loss_matrix[i, j]` must refer to contextualized token `i` at axis point `j`.
Row alignment across checkpoints or scale points is essential: permuting rows
between columns invalidates token-level trajectories.

```python
import numpy as np

np.savez_compressed(
    "my_losses.npz",
    axis_values=np.asarray(axis_values, dtype=np.float64),
    loss_matrix=np.asarray(loss_matrix, dtype=np.float32),
    axis_name=np.asarray(["training step"]),
    sample_id=np.asarray(sample_ids),
    token_pos=np.asarray(token_positions),
)
```

Before fitting, check that:

- `axis_values` is one-dimensional and strictly increasing;
- `loss_matrix.shape[1] == len(axis_values)`;
- every token-level metadata array has length `N`;
- each row identifies the same contextualized token at every axis point;
- non-finite or missing loss values have been handled explicitly;
- token loss uses the same reduction and vocabulary convention at every point.

## Minimal Configuration

Save the following as `configs/my_analysis.yaml` and adjust the input path,
device, and fit settings:

```yaml
input:
  losses_npz: path/to/my_losses.npz
  axis_name: training step

output:
  dir: runs/my_analysis
  reuse_cache: true

fit:
  epochs: 2000
  lr: 0.03
  chunk_size: 8192
  device: cuda
  constrain_tau: true
  seed: 1234

filters:
  require_positive_delta: true
  min_abs_delta: 0.01
  min_r2: 0.0

spectrum:
  bins: 80
  binning: uniform
  curve: pos_mass

pulse:
  tau_bins: 40
  binning: uniform
  align_to_tau: false
  offset_grid_points: 401
  chunk_size: 50000

rate:
  normalize_by_axis: true

reconstruction:
  dense_points: 401
  chunk_size: 50000
```

Run:

```bash
tokenlearning-run --config configs/my_analysis.yaml
```

Use `device: cpu` for small matrices. For large matrices, tune `chunk_size`
according to available device memory.

## Python API

```python
from tokenlearning.core import (
    compute_binned_learning_pulses,
    estimate_spectrum,
    filter_tokens,
    fit_token_events,
)
from tokenlearning.io import load_loss_npz

data = load_loss_npz("path/to/my_losses.npz")
fit = fit_token_events(
    data.axis_values,
    data.loss_matrix,
    {"device": "cuda", "epochs": 2000, "chunk_size": 8192},
)
mask = filter_tokens(
    fit,
    {"require_positive_delta": True, "min_abs_delta": 0.01},
)
spectrum = estimate_spectrum(fit.tau[mask], bins=80)
pulses = compute_binned_learning_pulses(
    data.axis_values,
    fit,
    mask,
)
```

## Interpreting the Outputs

- `tau`: center of the fitted token-level transition along the selected axis.
- `lambda_`: transition sharpness in inverse axis units.
- `delta`: fitted loss decrease from the initial to final plateau.
- `r2`: per-token goodness of fit.
- `learning_time_spectrum.png`: distribution of selected learning events.
- `binned_learning_pulses.png`: average localized loss-decrease pulses grouped
  by learning time.
- `rate_reconstruction.png`: comparison between empirical, sigmoid-implied, and
  spectrum-dominated aggregate loss decrease rates.

The fitted event is a compact description of the dominant transition, not a
claim that every token trajectory is exactly sigmoid. Inspect filtering
sensitivity, poor-fit cases, axis-boundary events, and alternative bin counts
before drawing conclusions from a new dataset.

## Acquiring Losses

Two optional acquisition paths are included:

- [`loss_acquisition/synthetic_mano/`](../loss_acquisition/synthetic_mano/) for
  generating and scoring public synthetic arithmetic runs;
- [`loss_acquisition/llm_pretraining_sglang/`](../loss_acquisition/llm_pretraining_sglang/)
  for scoring user-supplied LLM checkpoints with SGLang.

Keep private model code, weights, tokenizer artifacts, and validation text
outside this repository.
