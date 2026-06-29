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

from loss_acquisition.synthetic_mano._pipeline import train_mano_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PhysicsLM Mano trainer and save step checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Prepare the PhysicsLM train config and launch training without regenerating data.",
    )
    args = parser.parse_args()
    result = train_mano_model(args.config, generate_data=False if args.skip_data else None)
    if result.get("skipped"):
        print(f"wrote PhysicsLM train config to {result['train_config']} (training skipped)")
    else:
        print(f"wrote checkpoints to {result['checkpoint_dir']}")


if __name__ == "__main__":
    main()
