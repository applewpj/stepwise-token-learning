# Analysis Configs

Each YAML file runs the core `tokenlearning` analysis pipeline on one public
`losses.npz` matrix.

| Config | Paper axis | Input | Notes |
| --- | --- | --- | --- |
| `synthetic_mano_uniform_validation_10k.yaml` | Synthetic Mano appendix | `examples/synthetic_mano_uniform_validation_10k/losses.npz` | CPU-friendly first run. |
| `step_axis_1m.yaml` | T axis | `examples/step_axis_1m/losses.npz` | Large 1M-token fit; CUDA recommended. |
| `d_axis_1m.yaml` | D axis | `examples/d_axis_1m/losses.npz` | Large 1M-token fit; CUDA recommended. |
| `m_axis_1m.yaml` | M axis | `examples/m_axis_1m/losses.npz` | Large 1M-token fit; CUDA recommended. |

The T/D/M configs assume the example NPZ files have already been downloaded
from the HuggingFace Dataset asset repo:

```bash
python tools/download_examples.py --repo-id applewpj/token-learning-spectrum-examples --all
```
