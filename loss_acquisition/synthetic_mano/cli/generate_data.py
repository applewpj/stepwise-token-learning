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

from loss_acquisition.synthetic_mano._pipeline import generate_mano_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PhysicsLM-style Mano synthetic train/validation JSONL data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = generate_mano_data(args.config)
    for split in ("train", "validation"):
        if split in result:
            files = ", ".join(result[split]["files"])
            print(f"wrote {split} data: {files}")


if __name__ == "__main__":
    main()
