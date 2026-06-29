#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss_acquisition.llm_pretraining_sglang.config import load_yaml
from loss_acquisition.llm_pretraining_sglang.data import prepare_validation_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare validation JSONL for SGLang checkpoint scoring.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    out = prepare_validation_data(load_yaml(config_path), base_dir=REPO_ROOT)
    print(out)


if __name__ == "__main__":
    main()
