# LLM Pretraining SGLang Loss Acquisition

This directory is the public loss-acquisition path for real LLM pretraining
checkpoints. It scores a fixed validation set with an SGLang-compatible runtime,
exports per-token negative log probabilities for each checkpoint, and combines
those checkpoint caches into the public `losses.npz` format consumed by
`tokenlearning`.

The directory intentionally does not contain private architecture code,
checkpoint conversion code, weights, tokenizers, datasets, data composition, or
experiment registries. Users provide their own model/checkpoint paths and
validation data.

## Layout

```text
loss_acquisition/llm_pretraining_sglang/
  README.md
  config.py                       # YAML loading, path/model reference helpers
  data.py                         # validation JSONL / HuggingFace data prep
  scoring.py                      # SGLang prompt logprob scoring
  export.py                       # checkpoint caches -> losses.npz
  pipeline.py                     # config-driven orchestration
  cli/
    prepare_data.py               # materialize validation JSONL
    score_checkpoint.py           # score one configured checkpoint
    build_loss_npz.py             # combine checkpoint loss caches
    run_pipeline.py               # run prepare, score, and combine
  configs/
    hf_smollm2_smoke.yaml         # public smoke-test config
  env/
    requirements-sglang.txt
```

Generated outputs are written under
`loss_acquisition/llm_pretraining_sglang/runs/`, which should stay out of git.
Prepared validation JSONL files are written under
`loss_acquisition/llm_pretraining_sglang/data/`; the public smoke config points
there explicitly and includes a tiny committed `validation.jsonl` so the full
smoke test can run without downloading validation data. Other generated JSONL
files should stay out of git.

## Environment

Install the core repo first:

```bash
pip install -e .
```

Then install an SGLang runtime in the environment where checkpoint scoring will
run:

```bash
pip install -r loss_acquisition/llm_pretraining_sglang/env/requirements-sglang.txt
```

SGLang model execution normally requires a CUDA-capable machine. If you use a
patched or local SGLang checkout for a custom public model implementation, pass
it in the YAML config with `sglang.sglang_python_path` and optional
`sglang.sgl_kernel_python_path`. Keep private implementations outside this repo.

## Run By Stage

Use the public smoke config as the command template:

```bash
CONFIG=loss_acquisition/llm_pretraining_sglang/configs/hf_smollm2_smoke.yaml
```

Prepare the validation JSONL. The config can read either an existing JSONL file
or a HuggingFace dataset split:

```bash
python loss_acquisition/llm_pretraining_sglang/cli/prepare_data.py --config "$CONFIG"
```

Score one checkpoint from `checkpoints`:

```bash
python loss_acquisition/llm_pretraining_sglang/cli/score_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint step_0
```

Score every checkpoint and build `losses.npz`:

```bash
python loss_acquisition/llm_pretraining_sglang/cli/run_pipeline.py --config "$CONFIG"
```

The output `losses.npz` can then be passed to any normal `tokenlearning`
analysis config.

## Manual Combination

If checkpoints are scored separately, combine the resulting directories:

```bash
python loss_acquisition/llm_pretraining_sglang/cli/build_loss_npz.py \
  --run 0=loss_acquisition/llm_pretraining_sglang/runs/example/checkpoint_losses/step_0 \
  --run 5000=loss_acquisition/llm_pretraining_sglang/runs/example/checkpoint_losses/step_5000 \
  --out loss_acquisition/llm_pretraining_sglang/runs/example/losses.npz \
  --axis-name "training step"
```

Each checkpoint cache must contain a `token_losses.npy` file with columns:

```text
shuffle_idx, sample_id, token_pos, loss
```

`build_loss_npz.py` verifies that checkpoint caches align on the same token
rows before writing the final matrix.

## Config Fields

`output.dir` controls the run directory.

`data` controls validation data:

- `source: jsonl` with `path` and `text_field`;
- `source: huggingface` with `dataset`, optional `subset`, `split`, and
  `text_field`;
- `limit` for small smoke runs;
- `prepared_jsonl` for the materialized prompt file.

`model` defines the default model/tokenizer reference. Each checkpoint entry can
override `model`, `path`, `tokenizer`, and `trust_remote_code`.

`checkpoints` maps scaling-axis values to checkpoints:

```yaml
checkpoints:
  - name: step_0000
    axis_value: 0
    path: /path/to/checkpoint_0000
  - name: step_5000
    axis_value: 5000
    path: /path/to/checkpoint_5000
```

`score` controls batch size, validation limit, sampling args, and `axis_name`.
For loss acquisition, generation length should usually remain
`max_new_tokens: 0`; the scorer only needs prompt token log probabilities.

`sglang` forwards runtime settings to `sgl.Engine`, including `tp_size`,
`dp_size`, `context_length`, `mem_fraction_static`, `disable_cuda_graph`, and
extra `engine_args`. Set `model_type` only for a runtime that needs a
`MODEL_TYPE` environment value.

`build` controls final `losses.npz` writing. Set `build.enabled: false` when
you only want per-checkpoint caches.

## Public Boundary

The only artifact that must enter the core analysis path is `losses.npz`.
Validation text, raw token-loss caches, manifests, checkpoints, tokenizer files,
custom model code, and internal metadata can remain local.

For comparable token trajectories, all checkpoints should score the same
validation examples with the same tokenizer and token order.
