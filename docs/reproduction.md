# Reproduction Notes

The [README](../README.md#reproduce-the-paper) has the commands. This page
records what each paper config does, what `--quick` changes, how the reference
numbers were produced, and what the release does not cover.

## Settings of the paper configs

| | [`step_axis_1m`](../configs/step_axis_1m.yaml) | [`d_axis_1m`](../configs/d_axis_1m.yaml) | [`m_axis_1m`](../configs/m_axis_1m.yaml) | [`synthetic_mano_uniform_validation_10k`](../configs/synthetic_mano_uniform_validation_10k.yaml) |
| --- | --- | --- | --- | --- |
| Paper result | Figure 3 | Figure 4 | Figure 5 | Appendix, uniform Mano |
| Axis | training step, 15k–71k | data, 1.2B–34.7B tokens | model, 0.42–5.00 GFLOPs/token | training step, 5k–400k |
| Matrix | 1,000,000 × 54 | 1,000,000 × 50 | 1,000,000 × 60 | 10,000 × 80 |
| Fit | 10,000 epochs, lr 0.02 | 10,000 epochs, lr 0.05 | 10,000 epochs, lr 0.02 | 500 epochs, lr 0.1 |
| Rows per chunk | 500,000 | 1,000,000 | 1,000,000 | 2,048 |
| Event filter | drop if \|Δ\| < 0.3 and R² < 0.5 | drop if R² < 0 or λ < 0.2 | keep if loss SNR ≥ 0.2 | Δ ≥ 0.01 and 5k ≤ τ ≤ 405k |
| Spectrum bins | 40 | 40 | 40 | 301 |

The T, D, and M configs use the paper's "plot-compatible" fit: Adam on all
four sigmoid parameters of every token in a chunk at once, with a loss that
adds the error of the chunk's mean curve to the per-token squared error.
Because of that coupling, the chunk size belongs to the method. See
[determinism](#determinism-and-tolerances) below.

## What `--quick` changes

`--quick` fits a random subset of 50,000 rows (subset seed 0) for 2,000 epochs,
with every other setting taken from the paper config. Two details keep the
subset comparable to the full matrix:

- The empirical loss derivative is always computed from the mean loss over
  **all** one million rows, so the subset's spectrum is compared with the real
  curve rather than with a noisier subset mean.
- The paper configs normalize the reconstruction by the number of rows
  (`reconstruction.denominator: 1000000`). With a subset, the pipeline rescales
  this to the subset size, so the reconstruction stays on the same scale.

`--max-tokens N`, `--epochs E`, and `--seed S` give other subsets. They are
reported as `custom` runs and are not checked against the reference.

## Reference numbers

[`configs/reference_metrics.yaml`](../configs/reference_metrics.yaml) holds the
four numbers that `tokenlearning reproduce` checks:

| Key | Meaning | Tolerance |
| --- | --- | --- |
| `kept_fraction` | fraction of fitted tokens that pass the event filter | ± 0.02 |
| `tau_median_filtered` | median learning time of the kept tokens, in axis units | ± 3% |
| `mean_loss_r2` | R² of the sigmoid-implied mean loss against the measured mean loss (panels 3A, 4B, 5B) | ± 0.005 |
| `sigmoid_rate_reconstruction_corr` | Pearson correlation of the spectrum-dominated and sigmoid-implied loss derivatives (panels 3D, 4C, 5C) | ± 0.02 |

They were recorded on CPU (AMD EPYC 7K62, 16 vCPUs; Python 3.12, PyTorch
2.15, NumPy 2.5) with `tools/record_reference_metrics.py` from runs of the
unmodified configs.
`metrics.json` additionally reports `rate_reconstruction_corr`, the correlation
between the empirical and spectrum-dominated derivatives. It is not used as a
check, because the empirical derivative is a finite difference of a noisy
curve. Judge that agreement from `rate_reconstruction_loglog.png`, as in the
paper.

## Determinism and tolerances

On one machine, a rerun gives identical numbers. We also refitted the Figure 3
quick setting:

- **With 4 instead of 8 CPU threads:** all four numbers were identical, and the
  largest change in any token's learning time was 0.008 steps.
- **With half the chunk size (25,000 instead of 50,000 rows):** the median
  learning time moved by 2.2%, the mean-loss R² by 0.009, and the derivative
  correlation by 0.013. Individual learning times moved by a median of
  222 steps.

The tolerances are therefore loose enough for hardware and thread-count
differences, but a changed `chunk_size`, `epochs`, or `lr` can fail a check.
That is intended: those settings are part of the fitting method.

## Outputs

```text
runs/<id>/                      runs/<id>_quick/ with --quick
├── config.yaml                 the exact settings of this run
├── metrics.json                summary numbers, timings, subset, device
├── cache/
│   ├── fit_params.npz          tau, lambda_, delta, start, end, r2, valid_mask per token
│   ├── filtered_mask.npy       tokens that pass the event filter
│   └── analysis_arrays.npz     spectrum, pulses, and the three loss derivatives
└── figures/
    ├── mean_loss_raw_vs_sigmoid.png
    ├── binned_learning_pulses.png
    ├── learning_time_spectrum.png
    ├── rate_reconstruction.png
    ├── rate_reconstruction_loglog.png
    └── token_examples_by_tau_decile.png
```

The synthetic run adds `mean_loss_by_difficulty.png` and
`token_examples_by_difficulty.png`. [`concepts.md`](concepts.md) defines every
array and figure.

`cache/fit_params.npz` also stores a fingerprint of the fit settings. A rerun
reuses the fit only when the fingerprint matches, so changing `epochs`, `lr`,
`chunk_size`, or the subset always triggers a new fit.

## What the release does not cover

The T, D, and M matrices are sanitized analysis inputs: they contain
`axis_values`, `loss_matrix`, and `axis_name` only. Model code, weights,
tokenizers, validation text, training-data composition, and checkpoint paths
are not released. The release therefore reproduces the token-level analysis,
but not:

- the pre-training runs, or panels 4A and 5A that show their training curves;
- Figure 2, which compares three independent runs of the same model;
- the data-reshaping experiments of Figures 6 and 7, which ran inside the
  training stack. The learning times they use are the `tau` that this package
  estimates.

Rows of the T, D, and M matrices cannot be mapped back to token strings. The
synthetic Mano path is public end to end; see
[`loss_acquisition/synthetic_mano/`](../loss_acquisition/synthetic_mano/).
