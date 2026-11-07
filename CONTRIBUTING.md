# Contributing

Bug reports, questions, and pull requests are welcome. Please open an
[issue](https://github.com/applewpj/stepwise-token-learning/issues) before
starting a large change.

## Development Checks

```bash
pip install -e ".[dev,hf]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python tools/check_release.py --fail-on-large-github-files
```

CI runs the same checks on every push and pull request.

## Public/Private Boundary

Contributions must keep the release free of private material. Do not commit
model code, checkpoints, tokenizer artifacts, raw validation text, training-data
composition, internal experiment registries, local environments, or run outputs
(`runs/`, caches). `tools/check_release.py` scans text files for known private
patterns and fails on files over 100 MB.

Large `losses.npz` files belong in the
[Hugging Face dataset](https://huggingface.co/datasets/applewpj/token-learning-spectrum-examples),
not in this repository.

## Project Page

The project page is `docs/index.html`, served by GitHub Pages from the `docs/`
folder of `main`. Its live figures load JSON, so preview it over HTTP rather
than from a `file://` URL:

```bash
python -m http.server -d docs 8000      # then open http://localhost:8000
```

The figures read `docs/static/data/{t,d,m}.json` for the real data and
`synthetic.json` for Mano, about 100 KB each, exported from finished runs:

```bash
tokenlearning reproduce fig3 fig4 fig5 synthetic  # add --quick for the subset setting
python tools/build_site_data.py                   # all four files; --only T for one
```

Real axes use `runs/<id>` when it exists and `runs/<id>_quick` otherwise. The
grouped curves and the example tokens use only the interior of each axis
(`AXES` at the top of the script): learning times at the ends are poorly
determined, because tokens learned before the first point pile up there and
steps cut off by the last point are fitted from half a transition. `AXES` also
holds the irreducible loss E of each power-law fit, as in the paper's figures.

## README and Dataset Card Images

`assets/readme/` holds the banner, the link buttons, and the animated preview
of the mechanism toy that the README and the dataset card show. They are drawn
in the style of the project page:

```bash
tokenlearning reproduce fig3            # the banner plots tokens from the paper-setting fit
python tools/make_readme_assets.py      # all images; --only banner, buttons, or preview
```

The preview replays `docs/static/js/toy.js` through a Python port with the
page's seed and default sliders. If the toy changes, change the port with it.
The dataset card loads these images from `main` on GitHub, so push them before
the card.

## Publishing Example Data (Maintainers)

To add or update a released matrix:

1. Add its metadata to `examples/manifest.yaml` and verify shape, byte size,
   axis range, and SHA256.
2. Add the paper/config/output mapping to `docs/reproduction.md` and the asset
   description to `examples/README.md`.
3. Update the dataset card in `examples/hf_dataset_card.md`, including its
   `configs` block if a new viewer config is needed.
4. Export the Dataset Viewer previews:

   ```bash
   python tools/export_hf_previews.py
   ```

   This writes `examples/previews/*.parquet` (ignored by git): a seeded
   10,000-row sample of each large matrix, the full small matrices, and
   `axes.parquet`.
5. Preview the upload, then publish:

   ```bash
   python tools/upload_examples_to_hf.py \
     --repo-id applewpj/token-learning-spectrum-examples \
     --dry-run
   python tools/upload_examples_to_hf.py \
     --repo-id applewpj/token-learning-spectrum-examples
   ```

   The upload tool verifies sizes and checksums, then uploads the matrices,
   previews, manifest, license, and dataset card.
6. Run the release checks and CI, and tag an immutable dataset revision when
   possible.
