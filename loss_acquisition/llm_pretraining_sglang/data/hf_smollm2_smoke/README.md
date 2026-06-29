# HF SmolLM2 Smoke Validation Data

`validation.jsonl` is a committed public smoke validation file generated from
the TinyStories validation split.

The smoke config uses it directly. To regenerate it from HuggingFace, delete the
file or set `data.overwrite: true`, then run:

```bash
python loss_acquisition/llm_pretraining_sglang/cli/prepare_data.py \
  --config loss_acquisition/llm_pretraining_sglang/configs/hf_smollm2_smoke.yaml
```
