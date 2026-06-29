#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss_acquisition.llm_pretraining_sglang.config import load_yaml
from loss_acquisition.llm_pretraining_sglang.scoring import score_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one LLM pretraining checkpoint with SGLang.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint name from config.checkpoints. Defaults to the first entry.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    checkpoints = [dict(item) for item in config.get("checkpoints", [])]
    if not checkpoints:
        raise ValueError("Config must contain at least one checkpoint entry.")
    if args.checkpoint is None:
        checkpoint = checkpoints[0]
    else:
        matches = [item for item in checkpoints if str(item.get("name")) == args.checkpoint]
        if not matches:
            raise ValueError(f"No checkpoint named {args.checkpoint!r}")
        checkpoint = matches[0]
    out_dir = score_checkpoint(config, checkpoint, base_dir=REPO_ROOT)
    print(out_dir)


if __name__ == "__main__":
    main()
