# Example Loss Matrices

The paper's analysis inputs are released as four `losses.npz` matrices in the
[Hugging Face dataset](https://huggingface.co/datasets/applewpj/stepwise-token-learning).
They are not committed to GitHub; this directory holds their manifest and the
dataset card.

| Asset ID | Paper result | Axis | Shape | Axis range | Size | Fields |
| --- | --- | --- | ---: | --- | ---: | --- |
| `step_axis_1m` | Figure 3 | training step | 1,000,000 × 54 | 15,000–71,000 | 196.7 MB | `axis_values`, `loss_matrix`, `axis_name` |
| `d_axis_1m` | Figure 4 | data scale (B tokens) | 1,000,000 × 50 | 1.196–34.662 | 183.0 MB | `axis_values`, `loss_matrix`, `axis_name` |
| `m_axis_1m` | Figure 5 | model scale (GFLOPs/token) | 1,000,000 × 60 | 0.416–4.999 | 219.3 MB | `axis_values`, `loss_matrix`, `axis_name` |
| `synthetic_mano_uniform_validation_10k` | Appendix | training step | 10,000 × 80 | 5,000–400,000 | 1.5 MB | the above plus `sample_id`, `token_pos`, `difficulty`, `seed`, `question` |

## Download

`tokenlearning reproduce` downloads what it needs. To fetch files yourself:

```bash
tokenlearning download --list           # what is released, and what you already have
tokenlearning download step_axis_1m     # one asset; figure names such as fig3 work too
tokenlearning download --all            # everything, about 600 MB
```

Files land at `examples/<asset-id>/losses.npz`, the paths the configs expect,
and are checked against the SHA256 recorded in [`manifest.yaml`](manifest.yaml).
Set `HF_ENDPOINT` to use a Hugging Face mirror. The older
`python tools/download_examples.py --name <asset-id>` still works.

To browse the data without downloading it, use the Dataset Viewer on the Hub or
load a Parquet preview (10,000 random rows per T/D/M matrix, all rows of the
synthetic matrix):

```python
from datasets import load_dataset

preview = load_dataset("applewpj/stepwise-token-learning", "step_axis_1m", split="preview")
```

## Provenance

- **T/D/M matrices.** Token losses from the paper's pre-training runs: a 1.5B
  dense model over 54 checkpoints (training steps), a 290M MoE data-scale
  frontier from a 50-run sweep, and a model-scale frontier from a 60-run sweep.
  Only aligned token losses and axis values are released: no model code,
  weights, tokenizers, validation text, training-data composition, run names,
  or checkpoint paths. Rows cannot be mapped back to token strings.
- **Synthetic Mano matrix.** Generated with the public synthetic arithmetic
  pipeline in [`loss_acquisition/synthetic_mano/`](../loss_acquisition/synthetic_mano/),
  adapted from PhysicsLM4. Its metadata (difficulty level, generator seed,
  and question string) is public.

## Files Here

- [`manifest.yaml`](manifest.yaml): asset IDs, Hub paths, configs, byte sizes,
  SHA256 checksums, shapes, and axis ranges. Used by the download and upload
  tools.
- [`hf_dataset_card.md`](hf_dataset_card.md): the dataset card published as
  the Hub repository's `README.md`.
