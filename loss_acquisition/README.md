# Loss Acquisition

This directory contains public-safe examples for producing analyzable
`losses.npz` files. The core package under `tokenlearning/` only starts after a
multi-axis loss matrix exists.

Large public example `losses.npz` files are distributed through the HuggingFace
Dataset asset repo described by `examples/manifest.yaml`; this directory is for
generating or converting losses, not for storing large analysis assets.

## Synthetic Mano

See `loss_acquisition/synthetic_mano/README.md` for environment setup, file
layout, PhysicsLM citation, and full running instructions.

Run the 51.2M train / 10k validation pipeline by stage:

```bash
python loss_acquisition/synthetic_mano/cli/generate_data.py --config loss_acquisition/synthetic_mano/configs/mano_51m.yaml
python loss_acquisition/synthetic_mano/cli/train.py --config loss_acquisition/synthetic_mano/configs/mano_51m.yaml
python loss_acquisition/synthetic_mano/cli/score_checkpoints.py --config loss_acquisition/synthetic_mano/configs/mano_51m.yaml
```

Or run it end to end:

```bash
python loss_acquisition/synthetic_mano/cli/run_pipeline.py --config loss_acquisition/synthetic_mano/configs/mano_51m.yaml
```

The synthetic Mano path mirrors PhysicsLM's seed configs, trainer, checkpoint
format, and Mano per-sample loss evaluator. Set `physicslm_root` in the config or
`PHYSICSLM_ROOT` in the environment. Scoring writes `losses.npz`,
`metrics.mano.jsonl`, `checkpoint_loss_records.json`, and `summary.json`.

## LLM Pretraining SGLang Loss Acquisition

See `loss_acquisition/llm_pretraining_sglang/README.md` for the config-driven
pipeline for real LLM pretraining checkpoints. This path prepares a validation
set, scores each checkpoint with SGLang prompt log probabilities, and builds the
public `losses.npz` matrix used by `tokenlearning`.

Run the public HuggingFace smoke config:

```bash
python loss_acquisition/llm_pretraining_sglang/cli/prepare_data.py \
  --config loss_acquisition/llm_pretraining_sglang/configs/hf_smollm2_smoke.yaml
python loss_acquisition/llm_pretraining_sglang/cli/run_pipeline.py \
  --config loss_acquisition/llm_pretraining_sglang/configs/hf_smollm2_smoke.yaml
```

Or score user-provided checkpoints from a copied config:

```bash
python loss_acquisition/llm_pretraining_sglang/cli/score_checkpoint.py \
  --config path/to/your_llm_pretraining_sglang.yaml \
  --checkpoint step_0000
python loss_acquisition/llm_pretraining_sglang/cli/build_loss_npz.py \
  --run 0=runs/llm_pretraining_sglang/checkpoint_losses/step_0000 \
  --run 5000=runs/llm_pretraining_sglang/checkpoint_losses/step_5000 \
  --out runs/llm_pretraining_sglang/losses.npz \
  --axis-name "training step"
```

Keep private architecture implementations, checkpoint conversion code, weights,
tokenizers, and real-data composition outside this repository.
