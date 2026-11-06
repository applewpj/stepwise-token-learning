#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATTERNS = [
    "/" + "cpfs",
    "/" + "newcpfs",
    "wang" + "pingjie",
    "cyber" + "tron",
    "pa" + "seo",
    "scaling" + "_moe",
    "pre" + "train_",
]


def iter_text_files() -> list[Path]:
    skip_parts = {".git", ".pytest_cache", "__pycache__", "runs", ".venv", ".cache"}
    skip_names = {"memory.md"}
    skip_paths = {Path("tools/check_release.py")}
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT)
        if rel in skip_paths:
            continue
        if any(part in skip_parts for part in rel.parts):
            continue
        if rel.name in skip_names:
            continue
        if path.suffix in {".npz", ".npy", ".pyc", ".png", ".pdf", ".webp", ".jpg", ".jpeg", ".gif", ".parquet"}:
            continue
        files.append(path)
    return files


def iter_publishable_files() -> list[Path]:
    """Files a push would include: tracked, or untracked and not ignored (downloaded matrices are ignored)."""
    try:
        listing = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        return [path for path in REPO_ROOT.rglob("*") if path.is_file()]
    return [REPO_ROOT / name for name in listing.split("\0") if name and (REPO_ROOT / name).is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight release checks.")
    parser.add_argument("--fail-on-large-github-files", action="store_true")
    args = parser.parse_args()

    problems: list[str] = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PRIVATE_PATTERNS:
            if re.search(pattern, text):
                problems.append(f"private-pattern {pattern!r} in {path.relative_to(REPO_ROOT)}")
    for rel in [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "examples/manifest.yaml",
    ]:
        if not (REPO_ROOT / rel).exists():
            problems.append(f"missing required release file: {rel}")
    if args.fail_on_large_github_files:
        for path in iter_publishable_files():
            if path.stat().st_size > 100 * 1024 * 1024:
                problems.append(f"large file over 100MB: {path.relative_to(REPO_ROOT)}")
    if problems:
        for problem in problems:
            print(problem)
        raise SystemExit(1)
    print("release checks passed")


if __name__ == "__main__":
    main()
