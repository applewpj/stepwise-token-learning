#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "examples" / "manifest.yaml"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or "files" not in data:
        raise ValueError(f"Invalid examples manifest: {path}")
    return data


def selected_entries(manifest: dict[str, Any], names: list[str] | None, all_files: bool) -> list[dict[str, Any]]:
    entries = [dict(item) for item in manifest.get("files", [])]
    if all_files:
        return entries
    if not names:
        raise ValueError("Pass --all or at least one --name.")
    by_id = {str(item["id"]): item for item in entries}
    missing = [name for name in names if name not in by_id]
    if missing:
        raise ValueError(f"Unknown example id(s): {missing}. Available: {sorted(by_id)}")
    return [by_id[name] for name in names]


def download_entry(repo_id: str, entry: dict[str, Any], *, revision: str | None, force: bool) -> Path:
    from huggingface_hub import hf_hub_download

    local_path = REPO_ROOT / str(entry["local_path"])
    expected_sha = str(entry["sha256"])
    if local_path.exists() and not force:
        actual_sha = sha256_file(local_path)
        if actual_sha == expected_sha:
            print(f"ok existing {local_path}")
            return local_path
        raise ValueError(
            f"{local_path} exists but checksum differs. "
            "Pass --force to replace it."
        )

    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=str(entry["hf_path"]),
            revision=revision,
        )
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, local_path)
    actual_sha = sha256_file(local_path)
    if actual_sha != expected_sha:
        local_path.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {local_path}: {actual_sha} != {expected_sha}")
    print(f"downloaded {local_path}")
    return local_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Token Learning Spectrum example loss matrices from HuggingFace.")
    parser.add_argument("--repo-id", required=True, help="HuggingFace dataset repo id, e.g. applewpj/token-learning-spectrum-examples")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--name", action="append", default=None, help="Example id from examples/manifest.yaml. Repeatable.")
    parser.add_argument("--all", action="store_true", help="Download all example loss matrices.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true", help="Replace existing local files.")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    for entry in selected_entries(manifest, args.name, args.all):
        download_entry(args.repo_id, entry, revision=args.revision, force=bool(args.force))


if __name__ == "__main__":
    main()
