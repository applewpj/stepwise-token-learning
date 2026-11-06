#!/usr/bin/env python3
"""Download released loss matrices. Same as `tokenlearning download`; kept for existing scripts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenlearning.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["download", *sys.argv[1:]])
