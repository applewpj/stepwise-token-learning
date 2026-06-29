#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss_acquisition.llm_pretraining_sglang.export import build_loss_npz_from_runs


def parse_run_spec(text: str) -> tuple[float, Path]:
    if "=" not in text:
        raise ValueError(f"Run spec must be axis_value=/path/to/run_dir, got {text!r}")
    axis_text, path_text = text.split("=", 1)
    return float(axis_text), Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public losses.npz from SGLang checkpoint loss directories.")
    parser.add_argument("--run", action="append", required=True, help="axis_value=/path/to/checkpoint_loss_dir. Repeat in axis order.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--axis-name", default="training step")
    parser.add_argument("--loss-subdir", default="losses/all")
    parser.add_argument("--token-limit", type=int, default=None)
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    out = build_loss_npz_from_runs(
        [parse_run_spec(item) for item in args.run],
        args.out,
        axis_name=args.axis_name,
        loss_subdir=args.loss_subdir,
        token_limit=args.token_limit,
        dataset_limit=args.dataset_limit,
    )
    print(out)


if __name__ == "__main__":
    main()
