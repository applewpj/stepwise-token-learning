# Analysis Configs

Each paper config runs the `tokenlearning` pipeline on one released matrix.
`tokenlearning reproduce <target>` uses them with the data path filled in, and
they also run directly:

```bash
tokenlearning run --config configs/<name>.yaml [--device cpu] [--max-tokens N] [--epochs E]
```

| Config | Paper result | `reproduce` target |
| --- | --- | --- |
| [`step_axis_1m.yaml`](step_axis_1m.yaml) | Figure 3, training steps *T* | `fig3` |
| [`d_axis_1m.yaml`](d_axis_1m.yaml) | Figure 4, data scale *D* | `fig4` |
| [`m_axis_1m.yaml`](m_axis_1m.yaml) | Figure 5, model scale *M* | `fig5` |
| [`synthetic_mano_uniform_validation_10k.yaml`](synthetic_mano_uniform_validation_10k.yaml) | Appendix, synthetic Mano | `synthetic` |
| [`template.yaml`](template.yaml) | starting point for your own matrix | |

[`reference_metrics.yaml`](reference_metrics.yaml) holds the numbers that
`tokenlearning reproduce` checks; [`docs/reproduction.md`](../docs/reproduction.md)
explains how they were recorded and how the configs differ.

## Config Sections

| Section | Controls |
| --- | --- |
| `input` | `losses_npz` path, the axis label used in figures, and optionally `max_tokens` and `subset_seed` for a random row subset |
| `output` | Run directory and whether to reuse a cached fit (`reuse_cache`) |
| `fit` | Per-token sigmoid fitting: `epochs`, `lr`, `chunk_size` (rows fitted together; part of the method, see the reproduction notes), `device` (`auto`, `cpu`, or `cuda`), initialization, and constraints |
| `filters` | Which fitted events enter the spectrum and pulses, e.g. `min_r2`, `min_abs_delta`, `require_positive_delta`, `tau_min`/`tau_max`, optional `loss_snr` |
| `token_examples`, `labels`, `label_examples` | Example-trajectory figures and label-grouped plots |
| `spectrum` | Histogram of learning times: `bins`, `binning`, optional `range` |
| `pulse` | Learning-time bins for the binned learning-pulse figure |
| `rate` | How the empirical loss derivative is computed (`normalize_by_axis`) and whether to add a log-log plot |
| `reconstruction` | Spectrum-dominated derivative: grid size, the token count used for normalization (`denominator`, rescaled automatically for a row subset), and `tau_range_for_g_mean`, the learning-time range used to estimate the shared pulse |

Command-line flags (`--device`, `--max-tokens`, `--epochs`, `--seed`,
`--output-dir`) override the file without editing it. For a new matrix, start
from [`template.yaml`](template.yaml) rather than a paper config: the paper
configs hard-code axis-specific values such as `tau_range_for_g_mean` and
`denominator`. [`docs/own-data.md`](../docs/own-data.md) walks through it.
