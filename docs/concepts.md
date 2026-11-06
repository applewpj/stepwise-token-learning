# Concepts

This page connects the vocabulary of the paper to the quantities the package
estimates and the files it writes. Section numbers refer to the paper.

## The Decomposition

Validation loss along a scaling axis $`a`$ (training steps $`T`$, data scale
$`D`$, or model scale $`M`$) is an average over contextualized token instances
$`\mathcal{I}`$:

```math
L(a) = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \ell_i(a),
\qquad
L'(a) := -\frac{\mathrm{d}L(a)}{\mathrm{d}a} = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \ell_i'(a)
```

A *contextualized token* is one target token at one position of one validation
sequence. Row `i` of a loss matrix is the loss of that instance at every axis
point, so the analysis needs rows that stay aligned across checkpoints or runs.

The question is which factor shapes $`L'(a)`$: global training dynamics,
power-law decay of individual tokens, or *when* different tokens are learned
(Section 3.1). The paper finds the last one.

## Learning Event

Each token trajectory is summarized by its dominant transition (Section 3.2):

```math
\ell_i(a) \approx \ell_i^{\mathrm{post}} + \Delta_i \left[ 1 - \sigma\big( \lambda_i (a - \tau_i) \big) \right],
\qquad
\Delta_i := \ell_i^{\mathrm{pre}} - \ell_i^{\mathrm{post}}
```

| Paper symbol | Meaning | Field in `cache/fit_params.npz` | Units |
| --- | --- | --- | --- |
| $`\tau_i`$ | Learning time: center of the transition, *when* the token is learned | `tau` | axis units |
| $`\lambda_i`$ | Sharpness: *how abruptly* it is learned; the width scales as $`1/\lambda_i`$ | `lambda_` | 1 / axis units |
| $`\Delta_i`$ | Loss reduction: *how much* is learned | `delta` | loss units |
| $`\ell_i^{\mathrm{pre}}`$, $`\ell_i^{\mathrm{post}}`$ | Plateaus before and after the event | `start`, `end` | loss units |
| | Goodness of fit of the sigmoid | `r2` | |
| | Finite parameters with $`\lambda_i > 0`$ | `valid_mask` | |

The cache also stores `axis_values`, the empirical mean loss `raw_loss_mean`,
and the sigmoid-implied mean loss `fitted_loss_mean`. The sigmoid is a compact
description of the dominant event, not a claim of exact logistic dynamics.
Tokens whose loss increases ($`\Delta_i < 0`$) are kept separate from learned
tokens throughout.

## Learning Pulse

The negative derivative of a learning event is a localized pulse centered at
$`\tau_i`$ that integrates to $`\Delta_i`$:

```math
\ell_i'(a) \approx \frac{\Delta_i \lambda_i}{4} \, \mathrm{sech}^2 \left( \frac{\lambda_i (a - \tau_i)}{2} \right)
```

Centering all pulses at zero and averaging gives the shared pulse
(Section 3.3):

```math
g(a) = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \ell_i'(a + \tau_i)
```

The figure `binned_learning_pulses.png` shows mean pulses for tokens grouped
into learning-time bins (`pulse_by_bin` in `cache/analysis_arrays.npz`).
Similar shapes across bins are what make the decomposition work.

## Learning-Time Spectrum

The learning-time spectrum is the distribution of learning times:

```math
p(\tau) = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \delta(\tau - \tau_i)
```

The package estimates it as a histogram over the filtered tokens
(`spectrum_*` arrays in `cache/analysis_arrays.npz`; figure
`learning_time_spectrum.png`, which overlays it on the empirical loss
derivative). On real language data the spectrum is heavy-tailed and
approximately power-law.

## Spectrum-Dominated Derivative

If pulses share the shape $`g`$, the aggregate loss derivative is a
superposition of shifted copies of $`g`$:

```math
L'(a) \approx \int p(\tau) \, g(a - \tau) \, \mathrm{d}\tau = (p * g)(a)
```

`rate_reconstruction.png` compares three curves, named as in the paper:

| Curve | Definition | Array in `cache/analysis_arrays.npz` |
| --- | --- | --- |
| Empirical loss derivative | Finite-difference derivative of the measured mean loss | `rate_axis`, `empirical_rate` |
| Sigmoid-implied derivative | Derivative of the mean of all fitted sigmoids | `sigmoid_rate` |
| Spectrum-dominated derivative | $`(p * g)(a)`$: one shared pulse placed at every learning time | `rate_axis_dense`, `reconstructed_rate_dense` |

`metrics.json` reports their agreement as `rate_reconstruction_corr` and
`rate_reconstruction_mae` (empirical vs. spectrum-dominated), and as
`sigmoid_rate_reconstruction_corr` and `sigmoid_rate_reconstruction_mae`
(sigmoid-implied vs. spectrum-dominated). Close agreement means the shape of
the loss derivative is set by *when* tokens are learned rather than by
token-specific pulse shapes. The empirical derivative is a finite difference
between neighboring axis points and inherits their noise, so compare it on the
log-log figure rather than through the correlation alone.

## Output Files

```text
runs/<name>/
├── config.yaml                 the exact settings of the run
├── metrics.json                summary numbers, timings, subset, device
├── cache/
│   ├── fit_params.npz          per-token learning-event parameters
│   ├── filtered_mask.npy       tokens used for the spectrum and pulses
│   └── analysis_arrays.npz     spectrum, pulses, and derivative curves
└── figures/
    ├── mean_loss_raw_vs_sigmoid.png
    ├── token_examples_by_tau_decile.png
    ├── learning_time_spectrum.png
    ├── binned_learning_pulses.png
    └── rate_reconstruction.png (and _loglog.png when enabled)
```

Besides the agreement metrics above, `metrics.json` reports the number of
fitted and kept tokens (`n_tokens`, `n_filtered_tokens`), the range and median
of the kept learning times (`tau_min_filtered`, `tau_max_filtered`,
`tau_median_filtered`), the mean R² of the kept fits (`mean_r2_filtered`), and
the R² of the sigmoid-implied against the measured mean loss (`mean_loss_r2`).

The public API keeps its original names (`reconstruct_rate`, `empirical_rate`,
and so on); in the paper and in the figures these quantities are called *loss
derivatives*.

## Reading Results on New Data

- Check `mean_r2_filtered` and the token-example figures before trusting a
  spectrum. Noisy trajectories, multiple transitions, and events at the edge of
  the measured range are common.
- Events with $`\tau`$ near the first or last axis point are only partially
  observed. `filters.tau_min`, `filters.tau_max`, and
  `reconstruction.tau_range_for_g_mean` exclude them from the spectrum or the
  shared pulse.
- Try a few spectrum bin counts and filtering thresholds; the conclusions
  should not hinge on one setting.
