# Synthetic Mano Loss Acquisition

This directory contains the public synthetic-data acquisition path for producing
`losses.npz` inputs from the Mano arithmetic task. It is intentionally outside
the `tokenlearning` core package: the core package only analyzes loss matrices,
while this directory handles data generation, PhysicsLM-style training, and
checkpoint scoring.

The Mano task, seed-driven data generation, and PhysicsLM/Lingua training flow
are adapted from the PhysicsLM4 code release. See the citation section below.

## Layout

```text
loss_acquisition/synthetic_mano/
  README.md
  _mano.py                         # deterministic Mano sample generation
  _pipeline.py                     # data/train/score orchestration
  cli/
    generate_data.py               # generate train/validation JSONL files
    train.py                       # launch the PhysicsLM trainer
    score_checkpoints.py           # export checkpoint losses to losses.npz
    run_pipeline.py                # run generation, training, and scoring
  configs/
    mano_51m.yaml                  # full 51.2M train / 10k validation recipe
    data/
      incremental_5120w.yaml       # train seed recipe
      validation_10k.yaml          # validation seed recipe
    physicslm/
      mano_5120w_baseline_bs128.yaml
  env/
    environment.yml
    requirements-physicslm-cu128.txt
```

Generated data, checkpoints, and loss files are written under
`loss_acquisition/synthetic_mano/runs/`, which is ignored by git.

## Environment

Data generation only needs the base `token-learning-spectrum` environment.
Training and checkpoint scoring require a CUDA environment that can run the
PhysicsLM/Lingua trainer.

From the repository root:

```bash
conda env create -f loss_acquisition/synthetic_mano/env/environment.yml
conda activate token-learning-spectrum-mano
pip install -e .
```

Then provide PhysicsLM4. Either clone it into the default location:

```bash
mkdir -p external
git clone https://github.com/facebookresearch/PhysicsLM4 external/PhysicsLM4
```

or point the pipeline at an existing checkout:

```bash
export PHYSICSLM_ROOT=/path/to/PhysicsLM4
```

For local runs without WandB logging:

```bash
export WANDB_MODE=disabled
```

The pinned environment uses Python 3.11 with `torch==2.10.0`,
`xformers==0.0.35`, and `triton==3.6.0`. If those wheels are unavailable on
your platform, install a matching CUDA `torch`/`xformers` pair before running
training or scoring.

## Run The Pipeline

Set a short variable for the full recipe:

```bash
CONFIG=loss_acquisition/synthetic_mano/configs/mano_51m.yaml
```

Generate the 51.2M training samples and 10k validation samples:

```bash
python loss_acquisition/synthetic_mano/cli/generate_data.py --config "$CONFIG"
```

This writes:

```text
loss_acquisition/synthetic_mano/runs/mano_51m/data/incremental_5120w/mano/mano.chunk.00.jsonl
loss_acquisition/synthetic_mano/runs/mano_51m/data/incremental_5120w/mano/mano.seeds.txt
loss_acquisition/synthetic_mano/runs/mano_51m/data/validation_10k/mano/mano.val.jsonl
loss_acquisition/synthetic_mano/runs/mano_51m/data/validation_10k/mano/mano.val.seeds.txt
```

Train the Mano baseline model:

```bash
python loss_acquisition/synthetic_mano/cli/train.py --config "$CONFIG"
```

If the data already exists, use:

```bash
python loss_acquisition/synthetic_mano/cli/train.py --config "$CONFIG" --skip-data
```

Score all saved checkpoints on the 10k validation set and export analyzable
losses:

```bash
python loss_acquisition/synthetic_mano/cli/score_checkpoints.py --config "$CONFIG"
```

The scorer writes these files under `loss_acquisition/synthetic_mano/runs/mano_51m/`:

```text
losses.npz
checkpoint_loss_records.json
metrics.mano.jsonl
summary.json
```

You can also run all three stages in order:

```bash
python loss_acquisition/synthetic_mano/cli/run_pipeline.py --config "$CONFIG"
```

After `losses.npz` exists, analyze it with a normal `tokenlearning` analysis
config. The acquisition code stops at producing losses; fitting, spectra, and
figures belong to the core `tokenlearning` pipeline.

## Configuration Notes

- `configs/mano_51m.yaml` is the top-level acquisition recipe.
- `configs/data/*.yaml` controls seed generation and difficulty ratios.
- `dataset_config.lens` controls the Mano difficulty distribution. The released
  config uses uniform difficulty over L1-L10.
- `configs/physicslm/*.yaml` is the PhysicsLM/Lingua training template.
- `data.overwrite: false` avoids replacing generated data by accident.
- `train.overrides` can be used to override PhysicsLM config fields without
  editing the template.

## Citation

If you use this synthetic Mano acquisition path, cite PhysicsLM:

```bibtex
@inproceedings{Allen2025-canon,
  author = {{Allen-Zhu}, Zeyuan},
  title = {{Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers}},
  year = {2025},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems},
  series = {NeurIPS~'25},
  note = {Full version available at \url{https://ssrn.com/abstract=5240330}}
}

@misc{Allen2025-resonate,
  title = {{Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality}},
  author = {{Allen-Zhu}, Zeyuan},
  year = {2025},
  url = {https://physics.allen-zhu.com/part-4-architecture-design/part-4-2},
  note = {Code released at \url{https://github.com/facebookresearch/PhysicsLM4}},
}
```
