# Reproducing the Paper Results

This guide maps each public analysis artifact to the corresponding paper result.
The release starts from contextualized-token loss matrices; it does not reproduce
the private pretraining runs that generated the original checkpoints.

## Environment

```bash
git clone https://github.com/applewpj/token-learning-spectrum
cd token-learning-spectrum
python -m venv .venv
source .venv/bin/activate
pip install -e ".[hf]"
```

The 1M-token T/D/M fits are configured for CUDA. The 10K-sample synthetic Mano
example is configured for CPU and is the recommended installation smoke test.

## Public Artifacts

| Paper result | Asset ID | Matrix shape | Axis range | Size |
| --- | --- | ---: | --- | ---: |
| Figure 3, training-step axis | `step_axis_1m` | 1,000,000 × 54 | 15,000–71,000 steps | 196,658,576 bytes |
| Figure 4, data-scale axis | `d_axis_1m` | 1,000,000 × 50 | 1.196–34.662B tokens | 182,973,827 bytes |
| Figure 5, model-scale axis | `m_axis_1m` | 1,000,000 × 60 | 0.416–4.999 GFLOPs/token | 219,348,775 bytes |
| Synthetic Mano appendix analysis | `synthetic_mano_uniform_validation_10k` | 10,000 × 80 | 5,000–400,000 steps | 1,506,854 bytes |

The authoritative paths and SHA256 checksums are stored in
[`examples/manifest.yaml`](../examples/manifest.yaml). The download tool verifies
every file before analysis.

## Quick Verification

```bash
python tools/download_examples.py \
  --repo-id applewpj/token-learning-spectrum-examples \
  --name synthetic_mano_uniform_validation_10k

tokenlearning-run \
  --config configs/synthetic_mano_uniform_validation_10k.yaml
```

Expected directory:

```text
runs/synthetic_mano_uniform_validation_10k/
├── cache/
│   ├── analysis_arrays.npz
│   ├── filtered_mask.npy
│   └── fit_params.npz
├── figures/
│   ├── binned_learning_pulses.png
│   ├── learning_time_spectrum.png
│   ├── mean_loss_by_difficulty.png
│   ├── mean_loss_raw_vs_sigmoid.png
│   ├── rate_reconstruction.png
│   ├── token_examples_by_difficulty.png
│   └── token_examples_by_tau_decile.png
└── metrics.json
```

At minimum, verify that `metrics.json` reports:

- `n_tokens: 10000`;
- `axis_min: 5000.0` and `axis_max: 400000.0`;
- finite `rate_reconstruction_corr` and `rate_reconstruction_mae`;
- paths to all core figures.

## T/D/M Analyses

```bash
python tools/download_examples.py \
  --repo-id applewpj/token-learning-spectrum-examples \
  --all

tokenlearning-run --config configs/step_axis_1m.yaml
tokenlearning-run --config configs/d_axis_1m.yaml
tokenlearning-run --config configs/m_axis_1m.yaml
```

The outputs are written to `runs/step_axis_1m`, `runs/d_axis_1m`, and
`runs/m_axis_1m`. Each run produces the same output contract as the synthetic
example, except for label-conditioned Mano plots.

The default configs preserve the public release settings used for the paper
analysis:

- per-token sigmoid fitting;
- filtering of invalid or low-signal events;
- a 40-bin learning-time spectrum for T/D/M;
- binned learning pulses;
- empirical, sigmoid-implied, and spectrum-dominated loss-rate curves.

These analyses jointly fit one million token trajectories for 10,000 epochs and
are not intended for CPU execution. Runtime and peak memory vary substantially
with GPU model, CUDA/PyTorch build, and fit chunk size; no hardware-independent
runtime estimate is claimed.

## Reproduction Boundaries

The public T/D/M matrices are sanitized analysis inputs. They preserve
`axis_values`, `loss_matrix`, and `axis_name`, but omit private model code,
weights, tokenizers, raw validation text, training-data composition, experiment
registries, and checkpoint paths.

Consequently, this repository supports reproduction of the token-event fitting,
spectrum estimation, and loss-rate reconstruction from the released matrices.
It does not support exact regeneration of the private pretraining checkpoints.
