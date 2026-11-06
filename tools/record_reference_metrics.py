#!/usr/bin/env python3
"""Record finished runs as the reference values checked by `tokenlearning reproduce`.

    python tools/record_reference_metrics.py runs/step_axis_1m_quick runs/step_axis_1m

Run directories are named by `tokenlearning reproduce`: `runs/<asset>` for the paper
setting and `runs/<asset>_quick` for `--quick`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tokenlearning.reproduce import CHECKED_METRICS, REFERENCE_RELPATH, TARGETS, summary_values  # noqa: E402

HEADER = """\
# Reference values for `tokenlearning reproduce`, recorded from complete runs with
# tools/record_reference_metrics.py. `quick` is the --quick setting (random 50,000
# tokens, subset seed 0, 2,000 epochs); `full` is the paper setting in configs/.
# A rerun matches when every value is within the tolerance below.
"""

DEFAULT_TOLERANCE = {
    "kept_fraction": {"abs": 0.02},
    "tau_median_filtered": {"rel": 0.03},
    "mean_loss_r2": {"abs": 0.005},
    "sigmoid_rate_reconstruction_corr": {"abs": 0.02},
}


def _round(value: float) -> float:
    return float(f"{value:.5g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    path = REPO_ROOT / REFERENCE_RELPATH
    reference = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    reference = reference or {}
    reference.setdefault("tolerance", DEFAULT_TOLERANCE)
    assets = {target.asset for target in TARGETS.values()}

    for run_dir in args.run_dirs:
        name = run_dir.resolve().name
        mode = "quick" if name.endswith("_quick") else "full"
        asset = name.removesuffix("_quick")
        if asset not in assets:
            raise SystemExit(f"{run_dir}: expected a directory named <asset> or <asset>_quick, with asset in {sorted(assets)}")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        values = summary_values(metrics)
        entry = {key: _round(values[key]) for key, _, _ in CHECKED_METRICS if key in values}
        entry["n_tokens"] = int(metrics["n_tokens"])
        reference.setdefault(asset, {})[mode] = entry
        print(f"recorded {asset} ({mode}): {entry}")

    ordered = {"tolerance": reference.pop("tolerance")}
    ordered.update({asset: reference[asset] for asset in [t.asset for t in TARGETS.values()] if asset in reference})
    path.write_text(HEADER + "\n" + yaml.safe_dump(ordered, sort_keys=False), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
