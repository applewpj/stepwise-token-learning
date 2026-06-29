# Token Learning Spectrum

Code and public loss matrices for decomposing smooth scaling laws into
token-level learning events, learning-time spectra, and spectrum-dominated loss
rate reconstructions.

This repository accompanies the paper:

> **Smooth Scaling Laws Hide Stepwise Token Learning**
> Pingjie Wang, Zechen Hu, Peiru Yang, Fu Guo, Debing Zhang. 2026.

## What This Repository Provides

Given a token-level loss matrix measured along a scaling axis `a`, the core
package fits each contextualized token with a localized learning event:

```text
ell_i(a) ~= ell_post_i + Delta_i * [1 - sigmoid(lambda_i * (a - tau_i))]
```

It then estimates the learning-time spectrum `p(tau)`, binned learning pulses
`g(t, tau)`, and the spectrum-dominated reconstruction:

```text
-dL/da ~= (p * g)(a)
```

The public release includes:

- `tokenlearning/`: loss loading, sigmoid fitting, filtering, spectrum
  estimation, reconstruction, and plotting.
- `configs/`: analysis configs for the synthetic Mano example and T/D/M paper
  axes.
- `examples/`: metadata for public `losses.npz` matrices hosted on
  HuggingFace.
- `loss_acquisition/`: optional public-safe paths for producing analyzable
  `losses.npz` files from synthetic Mano runs or SGLang checkpoint scoring.

The release does **not** include private model architectures, weights,
tokenizers, raw private validation text, training data composition, experiment
registries, or internal training infrastructure.

## Installation

```bash
git clone https://github.com/applewpj/token-learning-spectrum
cd token-learning-spectrum
pip install -e ".[dev,hf]"
```

For core analysis only, `pip install -e ".[dev]"` is enough. The `hf` extra is
only needed for downloading/uploading example assets.

## Download Example Loss Matrices

Large example `losses.npz` files are distributed through the public HuggingFace
Dataset repo `applewpj/token-learning-spectrum-examples` instead of GitHub.
Download them with:

```bash
python tools/download_examples.py \
  --repo-id applewpj/token-learning-spectrum-examples \
  --all
```

The expected assets are listed in `examples/manifest.yaml`.

To upload the current local example matrices to HuggingFace:

```bash
python tools/upload_examples_to_hf.py \
  --repo-id applewpj/token-learning-spectrum-examples \
  --dry-run

python tools/upload_examples_to_hf.py \
  --repo-id applewpj/token-learning-spectrum-examples
```

The upload script verifies file sizes and SHA256 checksums before uploading.

## Quickstart

Start with the CPU-friendly synthetic Mano example:

```bash
tokenlearning-run --config configs/synthetic_mano_uniform_validation_10k.yaml
```

The output appears under:

```text
runs/synthetic_mano_uniform_validation_10k/
  figures/
  cache/
  metrics.json
```

Core figures include:

- `mean_loss_raw_vs_sigmoid.png`
- `mean_loss_by_difficulty.png`
- `token_examples_by_tau_decile.png`
- `token_examples_by_difficulty.png`
- `learning_time_spectrum.png`
- `binned_learning_pulses.png`
- `rate_reconstruction.png`

## Reproducing Paper-Axis Figures

After downloading the HuggingFace example assets, run:

```bash
tokenlearning-run --config configs/step_axis_1m.yaml
tokenlearning-run --config configs/d_axis_1m.yaml
tokenlearning-run --config configs/m_axis_1m.yaml
```

| Paper axis | Config | Input asset | Main outputs |
| --- | --- | --- | --- |
| Step axis `T` | `configs/step_axis_1m.yaml` | `examples/step_axis_1m/losses.npz` | Mean loss, token examples, spectrum, binned pulses, rate reconstruction. |
| Data axis `D` | `configs/d_axis_1m.yaml` | `examples/d_axis_1m/losses.npz` | Data-scale frontier decomposition and reconstruction. |
| Model axis `M` | `configs/m_axis_1m.yaml` | `examples/m_axis_1m/losses.npz` | Model-scale frontier decomposition and reconstruction. |
| Synthetic Mano | `configs/synthetic_mano_uniform_validation_10k.yaml` | `examples/synthetic_mano_uniform_validation_10k/losses.npz` | Overall and difficulty-conditioned learning spectra. |

The T/D/M configs fit 1M-token matrices and are intended for CUDA. They default
to the same analysis settings used for the public figures; expect substantially
longer runtime on CPU.

## Input Format

The core input is a `.npz` file:

```text
axis_values: float array [K]
loss_matrix: float array [N, K]
axis_name: optional string array [1]
sample_id: optional array [N]
token_pos: optional array [N]
<metadata_name>: optional array [N]
```

`loss_matrix[i, j]` is the loss of the same contextualized token `i` at axis
point `j`. Extra arrays whose first dimension is `N` are loaded as token
metadata. For example, the synthetic Mano matrix includes `difficulty` and
`question`, which drive grouped loss plots and token-example subplot titles.

## Python API

```python
from tokenlearning.io import load_loss_npz
from tokenlearning.core import fit_token_events, filter_tokens
from tokenlearning.core import estimate_spectrum, compute_binned_learning_pulses

data = load_loss_npz("examples/synthetic_mano_uniform_validation_10k/losses.npz")
fit = fit_token_events(data.axis_values, data.loss_matrix, {"device": "cpu"})
mask = filter_tokens(fit, {"require_positive_delta": True, "min_r2": 0.2})
spectrum = estimate_spectrum(fit.tau[mask], bins=80)
pulses = compute_binned_learning_pulses(data.axis_values, fit, mask)
```

The public fit cache contains:

```text
tau
lambda_
delta
start
end
r2
valid_mask
axis_values
raw_loss_mean
fitted_loss_mean
```

It intentionally does not expose BIC-based fields.

## Producing Your Own `losses.npz`

The core package is loss-source agnostic. Acquisition code lives outside
`tokenlearning/`:

- `loss_acquisition/synthetic_mano/`: public synthetic Mano data generation,
  PhysicsLM-style training, checkpoint scoring, and loss export.
- `loss_acquisition/llm_pretraining_sglang/`: SGLang prompt-logprob scoring for
  real LLM pretraining checkpoints supplied by the user.

Synthetic Mano full pipeline:

```bash
python loss_acquisition/synthetic_mano/cli/generate_data.py \
  --config loss_acquisition/synthetic_mano/configs/mano_51m.yaml
python loss_acquisition/synthetic_mano/cli/train.py \
  --config loss_acquisition/synthetic_mano/configs/mano_51m.yaml
python loss_acquisition/synthetic_mano/cli/score_checkpoints.py \
  --config loss_acquisition/synthetic_mano/configs/mano_51m.yaml
```

SGLang public smoke test:

```bash
python loss_acquisition/llm_pretraining_sglang/cli/run_pipeline.py \
  --config loss_acquisition/llm_pretraining_sglang/configs/hf_smollm2_smoke.yaml
```

The SGLang smoke config uses a committed TinyStories validation JSONL and the
public `HuggingFaceTB/SmolLM2-135M` model. For private checkpoints, copy the
config and replace the model/checkpoint paths while keeping private code and
weights outside this repository.

## Repository Layout

```text
tokenlearning/                         Core analysis package.
configs/                               YAML analysis configs.
examples/                              HF asset manifest and example docs.
loss_acquisition/synthetic_mano/       Public synthetic Mano acquisition path.
loss_acquisition/llm_pretraining_sglang/ SGLang checkpoint-loss acquisition.
tools/                                 HF asset and release-check utilities.
tests/                                 Unit tests.
```

## Release Hygiene

Before pushing to GitHub:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python tools/check_release.py
```

Do not commit `runs/`, `__pycache__/`, `.pytest_cache/`, private checkpoints, or
large `examples/*/losses.npz` files. The large example matrices should live in
the HuggingFace Dataset asset repo.

## Citation

```bibtex
@article{wang2026smooth,
  title   = {Smooth Scaling Laws Hide Stepwise Token Learning},
  author  = {Wang, Pingjie and Hu, Zechen and Yang, Peiru and Guo, Fu and Zhang, Debing},
  year    = {2026},
  note    = {arXiv preprint},
}
```

## License

This repository is released under the MIT License. See `THIRD_PARTY_NOTICES.md`
for third-party assets and optional runtime dependencies.
