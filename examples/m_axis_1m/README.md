# M Axis 1M

This example contains a public-safe token-loss matrix for the model-scale axis.
It stores the raw 1M-token analysis-limit matrix before any prefit token
filtering.

File:

```text
losses.npz
```

This file is distributed through the HuggingFace example asset repo. Download it
with:

```bash
python tools/download_examples.py \
  --repo-id applewpj/token-learning-spectrum-examples \
  --name m_axis_1m
```

Shape:

```text
loss_matrix: 1000000 x 60
axis_values: 0.416082432 -> 4.998660096
axis_name: model scale (GFLOPs/token)
```

Stored fields:

```text
axis_values
loss_matrix
axis_name
```

No run names, raw text, checkpoint paths, model architecture details, weights,
or data-composition metadata are included.
