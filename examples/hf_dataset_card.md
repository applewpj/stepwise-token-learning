---
license: mit
pretty_name: Token Learning Spectrum Examples
tags:
  - scaling-laws
  - language-modeling
  - token-loss
  - interpretability
---

# Token Learning Spectrum Examples

This dataset hosts public `losses.npz` matrices for reproducing the figures in
the Token Learning Spectrum code release.

Each `losses.npz` follows the schema documented in the GitHub repository:

```text
axis_values: float array [K]
loss_matrix: float array [N, K]
axis_name: string array [1]
sample_id, token_pos, and metadata arrays: optional arrays [N]
```

Files are listed in `manifest.yaml` with shapes, byte sizes, and SHA256
checksums. Download them with:

```bash
python tools/download_examples.py --repo-id applewpj/token-learning-spectrum-examples --all
```

The T/D/M-axis loss matrices are sanitized public analysis inputs. They do not
include private model architectures, weights, tokenizers, raw validation text,
training data composition, experiment registries, or checkpoint paths.

The synthetic Mano matrix is generated from the public synthetic arithmetic
pipeline adapted from PhysicsLM4.
