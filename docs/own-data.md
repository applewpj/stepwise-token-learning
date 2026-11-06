# Analyze Your Own Loss Matrix

The analysis package is agnostic to the model, tokenizer, and source of the
losses. It requires the loss of the same contextualized token rows measured at
multiple ordered points along one scaling axis: checkpoints of one run
(training steps), runs with different data budgets, or models of different
sizes.

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

## Configuration

Start from [`configs/template.yaml`](../configs/template.yaml), which has
generic settings, rather than from a paper config, which hard-codes
axis-specific values:

```bash
cp configs/template.yaml configs/mine.yaml      # set input.losses_npz and output.dir
tokenlearning run --config configs/mine.yaml --max-tokens 50000   # first look at a random subset
tokenlearning run --config configs/mine.yaml    # all rows
```

`fit.device: auto` uses CUDA when it is available. Flags such as `--device`,
`--epochs`, `--max-tokens`, and `--output-dir` override the file for one run,
and each run directory keeps the exact `config.yaml` it used. Settings worth
revisiting for a new matrix:

- `fit.epochs` and `fit.lr`: check that the per-token `r2` distribution has
  stopped improving;
- `fit.chunk_size`: rows fitted together. Larger chunks need more memory;
- `filters`: which events count, e.g. a minimum loss drop `min_abs_delta`;
- `spectrum.bins`: the histogram resolution of the learning-time spectrum;
- `reconstruction.tau_range_for_g_mean`: the learning-time range used to
  estimate the shared pulse. Exclude events at the edges of the axis, which are
  only partly observed.

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

- `tau`: learning time, the center of the fitted transition, in axis units.
- `lambda_`: transition sharpness in inverse axis units.
- `delta`: loss reduction from the initial to the final plateau.
- `r2`: per-token goodness of fit.
- `learning_time_spectrum.png`: the learning-time spectrum of the selected
  tokens, overlaid on the empirical loss derivative.
- `binned_learning_pulses.png`: mean learning pulses of tokens grouped by
  learning time.
- `rate_reconstruction.png`: the empirical, sigmoid-implied, and
  spectrum-dominated loss derivatives.

[`concepts.md`](concepts.md) maps every output to the paper's definitions. The
fitted event is a compact description of the dominant transition, not a claim
that every token trajectory is exactly sigmoid. Inspect filtering sensitivity,
poor-fit cases, axis-boundary events, and alternative bin counts before drawing
conclusions from a new dataset.

## Acquiring Losses

Two optional acquisition paths are included:

- [`loss_acquisition/synthetic_mano/`](../loss_acquisition/synthetic_mano/) for
  generating and scoring public synthetic arithmetic runs;
- [`loss_acquisition/llm_pretraining_sglang/`](../loss_acquisition/llm_pretraining_sglang/)
  for scoring user-supplied LLM checkpoints with SGLang.

Keep private model code, weights, tokenizer artifacts, and validation text
outside this repository.
