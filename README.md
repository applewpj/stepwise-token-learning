<div align="center">

<h1>Smooth Scaling Laws Hide Stepwise Token Learning</h1>

<p><b>NeurIPS 2026 &nbsp;·&nbsp; Spotlight</b></p>

<p>
  Pingjie Wang<sup>1,2,*,‡</sup>&ensp;
  Zechen Hu<sup>1,*,†</sup>&ensp;
  Peiru Yang<sup>1,3,*,‡</sup>&ensp;
  Fu Guo<sup>1</sup>&ensp;
  Debing Zhang<sup>1,†</sup>
  <br>
  <sup>1</sup>Dots Studio, Xiaohongshu Inc.&ensp;
  <sup>2</sup>Shanghai Jiao Tong University&ensp;
  <sup>3</sup>Tsinghua University
  <br>
  <sub><sup>*</sup>Equal contribution&ensp; <sup>†</sup>Corresponding authors&ensp; <sup>‡</sup>Work done during an internship at Dots Studio, Xiaohongshu Inc.</sub>
</p>

<p>
  <a href="https://openreview.net/forum?id=THZDVZArRH"><img src="https://img.shields.io/badge/Paper-OpenReview-8C1B13?style=flat-square" alt="Paper on OpenReview"></a>
  <a href="https://applewpj.github.io/stepwise-token-learning/"><img src="https://img.shields.io/badge/Project-Page-1F6FEB?style=flat-square" alt="Project page"></a>
  <a href="https://colab.research.google.com/github/applewpj/stepwise-token-learning/blob/main/notebooks/quickstart.ipynb"><img src="https://img.shields.io/badge/Colab-Quickstart-F9AB00?style=flat-square&amp;logo=googlecolab&amp;logoColor=white" alt="Open the quickstart in Colab"></a>
  <a href="https://huggingface.co/datasets/applewpj/token-learning-spectrum-examples"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Data-loss%20matrices-FFD21E?style=flat-square" alt="Loss matrices on Hugging Face"></a>
  <a href="https://github.com/applewpj/stepwise-token-learning/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/applewpj/stepwise-token-learning/ci.yml?branch=main&amp;style=flat-square&amp;label=CI" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2EA44F?style=flat-square" alt="MIT License"></a>
</p>

</div>

Code and data to reproduce the token-level analysis of the paper, and to run it
on your own checkpoints. In one sentence: each token's loss drops in a single
sigmoid-like step at its own learning time τ, the steps share one pulse shape,
and so the derivative of the smooth loss curve is set by the distribution of
learning times, the *learning-time spectrum* p(τ). The
[project page](https://applewpj.github.io/stepwise-token-learning/) walks
through the idea with interactive figures.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Figure 1 of the paper: token loss trajectories with sigmoid fits, and the learning-time spectrum against the loss derivative along training steps, data scale, and model scale.">
</p>

## Quickstart

```bash
git clone https://github.com/applewpj/stepwise-token-learning
cd stepwise-token-learning
pip install -e .            # Python >= 3.10; a CPU-only PyTorch build is fine

tokenlearning reproduce fig3 --quick
```

This downloads the training-step loss matrix from Hugging Face (197 MB,
checksum verified), fits a sigmoid learning event to 50,000 randomly chosen
tokens, draws the panels of Figure 3, and checks the numbers against ours, in
about a minute on CPU:

```text
== fig3: Figure 3, training steps T (1.5B dense model, 300B tokens) ==
data      examples/step_axis_1m/losses.npz  (1,000,000 tokens x 54 points, 196.7 MB)
setting   quick: random 50,000 of 1,000,000 tokens, 2,000 epochs, device cpu
output    runs/step_axis_1m_quick

result    29,953 of 50,000 fitted tokens pass the event filter
check     tokens kept by the event filter                     59.9%  reference 59.9% +/- 2.0%  ok
check     median learning time of kept tokens                22,924  reference 22,924 +/- 3%  ok
check     R^2, sigmoid-implied vs measured mean loss         0.9876  reference 0.9876 +/- 0.0050  ok
check     corr, spectrum-dominated vs sigmoid-implied L'      0.946  reference 0.946 +/- 0.020  ok
time      fit 58s, total 1m12s
figures   runs/step_axis_1m_quick/figures/
          A               mean_loss_raw_vs_sigmoid.png
          B               binned_learning_pulses.png
          C               learning_time_spectrum.png
          D               rate_reconstruction.png
          D, log-log      rate_reconstruction_loglog.png
paper     assets/t_axis.png
```

No local setup? The
[Colab quickstart](https://colab.research.google.com/github/applewpj/stepwise-token-learning/blob/main/notebooks/quickstart.ipynb)
runs the same steps in the browser.

## Reproduce the paper

| Command | Reproduces | Download |
| --- | --- | ---: |
| `tokenlearning reproduce fig3` | Figure 3, training steps T | 197 MB |
| `tokenlearning reproduce fig4` | Figure 4, data scale D | 183 MB |
| `tokenlearning reproduce fig5` | Figure 5, model scale M | 219 MB |
| `tokenlearning reproduce synthetic` | Appendix, synthetic Mano | 1.5 MB |
| `tokenlearning reproduce all` | all four | 600 MB |

Without flags, a command runs the paper setting: every token, 10,000 epochs.
That is meant for a GPU, which is used automatically when available
(`--device cpu` forces the CPU, at hours per figure). `--quick` draws the same
panels from a random 50,000 tokens in about a minute on CPU. Results go to
`runs/<id>/`, or `runs/<id>_quick/` with `--quick`.

Every run does the same four things:

1. Downloads the loss matrix to `examples/<id>/losses.npz`, or verifies the copy
   already there against the SHA256 in [`examples/manifest.yaml`](examples/manifest.yaml).
2. Fits one sigmoid per token with the settings in [`configs/<id>.yaml`](configs).
3. Writes `figures/`, the fitted parameters `cache/fit_params.npz`, `metrics.json`,
   and the exact `config.yaml` that was run.
4. Prints four summary numbers and compares them with
   [`configs/reference_metrics.yaml`](configs/reference_metrics.yaml).
   The command exits with status 1 if any of them is out of tolerance.

Running a command again reuses the fitted parameters when the fit settings are
unchanged, so restyling or re-filtering takes seconds. Changing any fit setting
triggers a new fit.

### Expected numbers

| Target | Setting | Tokens kept | Median τ of kept tokens | Mean-loss R² | corr(spectrum-dominated, sigmoid-implied) |
| --- | --- | ---: | ---: | ---: | ---: |
| `fig3` | `--quick` | 59.9% | 22,924 steps | 0.9876 | 0.946 |
| `fig3` | paper | 61.2% | 23,305 steps | 0.9991 | 0.978 |
| `fig4` | `--quick` | 52.5% | 2.05B tokens | 0.9968 | 0.998 |
| `fig4` | paper | 52.5% | 2.42B tokens | 0.9994 | 0.995 |
| `fig5` | `--quick` | 91.3% | 0.476 GFLOPs/token | 0.9952 | 0.990 |
| `fig5` | paper | 91.2% | 0.494 GFLOPs/token | 0.9960 | 0.990 |
| `synthetic` | paper | 63.5% | 162,650 steps | 0.9989 | 0.988 |

The last column is the central check of the paper. Replacing every token's own
learning pulse with one shared pulse, placed at the measured learning times
(the *spectrum-dominated* derivative), barely changes the loss derivative
implied by the per-token fits. The *empirical* derivative is a finite
difference between neighboring checkpoints and is noisy. That is especially
true along M, where adjacent models are only a few percent apart in FLOPs.
Compare it with the reconstruction on the log-log plot, as the paper does,
rather than through a single correlation.

### Which file is which panel

| Paper panel | File in `runs/<id>/figures/` |
| --- | --- |
| 3A, 4B, 5B: measured and sigmoid-implied mean loss | `mean_loss_raw_vs_sigmoid.png` |
| 3B: binned learning pulses | `binned_learning_pulses.png` |
| 3C, insets of 4C and 5C: learning-time spectrum | `learning_time_spectrum.png` |
| 3D, 4C, 5C: empirical, sigmoid-implied, and spectrum-dominated derivative | `rate_reconstruction.png`, `rate_reconstruction_loglog.png` |
| Sampled tokens with their fits (not a paper panel) | `token_examples_by_tau_decile.png` |

The released matrices do not cover panels 4A and 5A (training curves of the
model and data sweeps) or the data-reshaping experiments of Figures 6 and 7.
Those come from training runs whose code and data are not part of this
release. The learning times that the reshaping builds on are the `tau` field of
`fit_params.npz`.

## Run it on your own losses

All the analysis needs is the loss of the same tokens, in the same contexts,
measured at ordered points along one axis (checkpoints, data budgets, model
sizes, ...):

```python
import numpy as np

np.savez_compressed(
    "my_losses.npz",
    axis_values=axis_values,   # float [K], strictly increasing
    loss_matrix=loss_matrix,   # float [N, K]; row i is one token at every axis point
    axis_name=np.asarray(["training step"]),
)
```

Copy the template, point it at the file, and look at a subset before fitting
everything:

```bash
cp configs/template.yaml configs/mine.yaml        # set input.losses_npz and output.dir
tokenlearning run --config configs/mine.yaml --max-tokens 50000
tokenlearning run --config configs/mine.yaml      # all tokens
```

The same steps from Python:

```python
from tokenlearning.io import load_loss_npz
from tokenlearning.core import fit_token_events, filter_tokens, estimate_spectrum

data = load_loss_npz("my_losses.npz")
fit = fit_token_events(data.axis_values, data.loss_matrix, {"device": "cuda"})
mask = filter_tokens(fit, {"require_positive_delta": True, "min_abs_delta": 0.01})
spectrum = estimate_spectrum(fit.tau[mask], bins=80)   # learning-time spectrum p(tau)
```

[`docs/own-data.md`](docs/own-data.md) has the full schema, a config template,
and how to choose the filters. To score your own checkpoints,
[`loss_acquisition/`](loss_acquisition/README.md) has two paths: one scores LLM
checkpoints with SGLang, the other regenerates the synthetic Mano runs end to
end.

## What is fitted

For token $`i`$ measured along a scaling axis $`a`$ (training step, data, or
model size), the pipeline fits one learning event:

```math
\ell_i(a) \approx \ell_i^{\mathrm{post}} + \Delta_i \left[ 1 - \sigma\big( \lambda_i (a - \tau_i) \big) \right],
\qquad \sigma(z) = \frac{1}{1 + e^{-z}}
```

Here $`\tau_i`$ is when the token is learned, $`\lambda_i`$ how abruptly, and
$`\Delta_i`$ how much loss it removes. The negative derivative of each fit is a
localized pulse. If the pulses share one shape $`g`$, the derivative of the
mean loss is the learning-time spectrum smoothed by that shape:

```math
L'(a) := -\frac{\mathrm{d}L}{\mathrm{d}a} \approx \int p(\tau)\, g(a - \tau)\, \mathrm{d}\tau = (p * g)(a)
```

`cache/fit_params.npz` holds one row per fitted token:

| Field | Meaning |
| --- | --- |
| `tau` | learning time, in units of the axis |
| `lambda_` | sharpness of the transition |
| `start`, `end` | loss plateaus before and after the transition |
| `delta` | loss drop, `start - end` |
| `r2` | goodness of fit of the sigmoid to the token's trajectory |
| `valid_mask` | the fit is finite and usable |

[`docs/concepts.md`](docs/concepts.md) maps every output file and metric to
the paper.

## Troubleshooting

<details>
<summary><b>The download from Hugging Face is slow or blocked</b></summary>

Use a mirror, or download by hand; the checksum is verified either way.

```bash
HF_ENDPOINT=https://hf-mirror.com tokenlearning download fig3
```

To download manually, get `examples/<id>/losses.npz` from the
[dataset page](https://huggingface.co/datasets/applewpj/token-learning-spectrum-examples/tree/main/examples)
and put it at the same path in your clone. `tokenlearning download --list`
shows what is present.
</details>

<details>
<summary><b>I have no GPU</b></summary>

Use `--quick` for every figure, or run the paper setting with `--device cpu`
and let it run for a few hours. A subset spectrum is noisier than the full
one, so use the paper setting for quantitative comparisons. Fewer tokens with
`--max-tokens` or fewer epochs with `--epochs` also work; such custom runs are
not checked against the reference.
</details>

<details>
<summary><b>CUDA runs out of memory</b></summary>

Tokens are fitted in chunks of `fit.chunk_size` rows: 500,000 for `fig3` and
1,000,000 for `fig4` and `fig5`. On CPU the paper-setting runs needed 2 to 3 GB
of RAM. Lowering `chunk_size` saves memory but is not neutral: the fitting
objective includes each chunk's mean loss curve, so the chunk size is part of
the method. In our test, halving it moved the median learning time by 2% and
the mean-loss R² by 0.009, enough to fail a reference check. If memory is the
problem, prefer `--device cpu`.
</details>

<details>
<summary><b>My numbers differ slightly from the reference</b></summary>

Small differences come from floating-point order on different hardware and are
covered by the tolerances in
[`configs/reference_metrics.yaml`](configs/reference_metrics.yaml). If a check
fails, make sure you ran an unmodified config and the default subset seed, and
delete `runs/<id>` to rule out a stale cache from a modified config.
</details>

## Repository layout

```text
tokenlearning/      the package: loading, sigmoid fits, filters, spectra, reconstruction, plots, CLI
configs/            one YAML per paper analysis, and the reference numbers
examples/           manifest (paths, sizes, checksums) of the released loss matrices
notebooks/          Colab quickstart
loss_acquisition/   producing loss matrices: synthetic Mano runs, SGLang scoring of checkpoints
docs/               concepts, reproduction details, own data, and the project page
tools/              release checks and maintainer utilities
tests/              unit and end-to-end tests
```

More detail: [concepts](docs/concepts.md) ·
[reproduction notes](docs/reproduction.md) ·
[released data](examples/README.md) ·
[contributing](CONTRIBUTING.md).

## Citation

```bibtex
@inproceedings{wang2026smooth,
  title     = {Smooth Scaling Laws Hide Stepwise Token Learning},
  author    = {Wang, Pingjie and Hu, Zechen and Yang, Peiru and Guo, Fu and Zhang, Debing},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=THZDVZArRH}
}
```

## Acknowledgements and license

This work was supported by Dots Studio, Xiaohongshu Inc. and Shanghai Jiao Tong
University. The synthetic Mano pipeline is adapted from
[PhysicsLM4](https://github.com/facebookresearch/PhysicsLM4); the SGLang smoke
test uses [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) and
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). Code and the loss
matrices are released under the [MIT License](LICENSE). Questions and bug
reports are welcome on the [issue tracker](https://github.com/applewpj/stepwise-token-learning/issues).
