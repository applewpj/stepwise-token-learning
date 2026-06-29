from __future__ import annotations

import sys
from pathlib import Path


def add_local_paths(repo_root: Path) -> None:
    if (repo_root / "tokenlearning").is_dir() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
