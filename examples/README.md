# Example Loss Matrices

This directory contains README files for public example loss matrices. The
large `losses.npz` files are distributed through a HuggingFace Dataset repo
rather than committed to GitHub.

Download all example assets after installing the optional HuggingFace dependency:

```bash
pip install -e ".[hf]"
python tools/download_examples.py --repo-id applewpj/token-learning-spectrum-examples --all
```

Download one example:

```bash
python tools/download_examples.py \
  --repo-id applewpj/token-learning-spectrum-examples \
  --name synthetic_mano_uniform_validation_10k
```

The expected files, shapes, sizes, and SHA256 checksums are listed in
`examples/manifest.yaml`.
