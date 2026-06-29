# Prepared Validation Data

This directory stores small public validation JSONL files for
`llm_pretraining_sglang` acquisition smoke tests.

The public smoke config uses the committed file:

```text
loss_acquisition/llm_pretraining_sglang/data/hf_smollm2_smoke/validation.jsonl
```

Most generated `.jsonl` files under this directory are ignored by git, but this
smoke file is intentionally included so users can run the full smoke test
without downloading validation data first.

To recreate it from the public HuggingFace source:

```bash
python loss_acquisition/llm_pretraining_sglang/cli/prepare_data.py \
  --config loss_acquisition/llm_pretraining_sglang/configs/hf_smollm2_smoke.yaml
```
