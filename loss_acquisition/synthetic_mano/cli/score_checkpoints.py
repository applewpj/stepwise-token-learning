#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, ACQUISITION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from _local_paths import add_local_paths

add_local_paths(REPO_ROOT)

from loss_acquisition.synthetic_mano._pipeline import score_mano_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mano checkpoints on validation data and export losses.npz.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = score_mano_checkpoints(args.config)
    if result.get("skipped"):
        print(f"skipped checkpoint scoring for {result['output_dir']}")
    else:
        print(f"wrote Mano losses to {result['losses_npz']}")


if __name__ == "__main__":
    main()
