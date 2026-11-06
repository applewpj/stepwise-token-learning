#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "examples" / "manifest.yaml"
DEFAULT_DATASET_CARD = REPO_ROOT / "examples" / "hf_dataset_card.md"
DEFAULT_PREVIEWS_DIR = REPO_ROOT / "examples" / "previews"
LICENSE_FILE = REPO_ROOT / "LICENSE"


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


def verify_local_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [dict(item) for item in manifest.get("files", [])]
    for entry in entries:
        path = REPO_ROOT / str(entry["local_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing local example file: {path}")
        expected_size = int(entry["size_bytes"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"Size mismatch for {path}: {actual_size} != {expected_size}")
        expected_sha = str(entry["sha256"])
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"Checksum mismatch for {path}: {actual_sha} != {expected_sha}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload example losses.npz files to a HuggingFace Dataset repo.")
    parser.add_argument("--repo-id", required=True, help="HuggingFace dataset repo id, e.g. applewpj/token-learning-spectrum-examples")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-card", default=str(DEFAULT_DATASET_CARD))
    parser.add_argument(
        "--previews-dir",
        default=str(DEFAULT_PREVIEWS_DIR),
        help="Parquet previews written by tools/export_hf_previews.py; uploaded to previews/.",
    )
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    entries = verify_local_files(manifest)
    print(f"verified {len(entries)} local example files")
    previews = sorted(Path(args.previews_dir).glob("*.parquet"))
    if not previews:
        print(
            f"warning: no Parquet previews in {args.previews_dir}; the dataset card's viewer configs need them. "
            "Run tools/export_hf_previews.py first."
        )
    if args.dry_run:
        for entry in entries:
            print(f"would upload {entry['local_path']} -> {entry['hf_path']}")
        for path in previews:
            print(f"would upload {path} -> previews/{path.name}")
        print(f"would upload {manifest_path} -> manifest.yaml")
        print(f"would upload {LICENSE_FILE} -> LICENSE")
        print(f"would upload {args.dataset_card} -> README.md")
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=bool(args.private), exist_ok=True)
    for entry in entries:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(REPO_ROOT / str(entry["local_path"])),
            path_in_repo=str(entry["hf_path"]),
        )
        print(f"uploaded {entry['hf_path']}")
    for path in previews:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(path),
            path_in_repo=f"previews/{path.name}",
        )
        print(f"uploaded previews/{path.name}")
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(manifest_path),
        path_in_repo="manifest.yaml",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(LICENSE_FILE),
        path_in_repo="LICENSE",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(args.dataset_card),
        path_in_repo="README.md",
    )
    print(f"uploaded example assets to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
