#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss_acquisition.llm_pretraining_sglang.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SGLang LLM-pretraining checkpoint loss acquisition.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = run_pipeline(args.config, base_dir=REPO_ROOT)
    print(f"wrote SGLang loss acquisition outputs to {summary['output_dir']}")


if __name__ == "__main__":
    main()
