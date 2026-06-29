# Step Axis 1M

This example contains a public-safe token-loss matrix for the training-step
axis.

File:

```text
losses.npz
```

This file is distributed through the HuggingFace example asset repo. Download it
with:

```bash
python tools/download_examples.py \
  --repo-id applewpj/token-learning-spectrum-examples \
  --name step_axis_1m
```

Shape:

```text
loss_matrix: 1000000 x 54
axis_values: 15000.0 -> 71000.0
axis_name: training step
```

Stored fields:

```text
axis_values
loss_matrix
axis_name
```

No run names, raw text, checkpoint paths, model architecture details, weights,
or data-composition metadata are included.
