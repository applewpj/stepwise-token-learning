# Contributing

Contributions are welcome when they preserve the public/private boundary of the
release.

Before opening a pull request:

```bash
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python tools/check_release.py
```

Do not commit private model code, checkpoints, tokenizer artifacts, raw private
validation text, training data composition, internal experiment registries, or
local run outputs.

Large example `losses.npz` files should be published to the HuggingFace Dataset
asset repo and listed in `examples/manifest.yaml`, not committed to GitHub.
