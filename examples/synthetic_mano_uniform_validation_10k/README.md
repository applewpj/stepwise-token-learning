# Synthetic Mano Uniform Validation 10K

This example contains token-loss trajectories for a 10K uniform-difficulty
Mano validation set.

File:

```text
losses.npz
```

This file is distributed through the HuggingFace example asset repo. Download it
with:

```bash
python tools/download_examples.py \
  --repo-id applewpj/token-learning-spectrum-examples \
  --name synthetic_mano_uniform_validation_10k
```

Shape:

```text
loss_matrix: 10000 x 80
axis_values: 5000.0 -> 400000.0
axis_name: training step
```

Stored fields:

```text
axis_values
loss_matrix
axis_name
sample_id
token_pos
difficulty
seed
question
```

The `difficulty` field can be used as a label for grouped loss plots. The
`question` field can be used as a title field for sampled token-loss subplots.
