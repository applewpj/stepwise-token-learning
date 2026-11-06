"""Locate the repository and download the released loss matrices."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REPO_ID = "applewpj/token-learning-spectrum-examples"
MANIFEST_RELPATH = Path("examples") / "manifest.yaml"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Return the clone root, i.e. the directory holding ``configs/`` and ``examples/manifest.yaml``."""
    here = Path(start or Path.cwd()).resolve()
    for path in (here, *here.parents, Path(__file__).resolve().parents[1]):
        if (path / MANIFEST_RELPATH).is_file() and (path / "configs").is_dir():
            return path
    raise FileNotFoundError(
        "Could not find examples/manifest.yaml. Run this command inside a clone of "
        "https://github.com/applewpj/stepwise-token-learning installed with `pip install -e .`."
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or "files" not in data:
        raise ValueError(f"Invalid examples manifest: {path}")
    return data


def manifest_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item) for item in manifest.get("files", [])}


def ensure_example(
    entry: dict[str, Any],
    repo_root: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Return the local path of a released matrix, downloading and verifying it if needed."""
    root = Path(repo_root)
    rel = Path(str(entry["local_path"]))
    local_path = root / rel
    expected_sha = str(entry["sha256"])

    if local_path.exists() and not force:
        if sha256_file(local_path) == expected_sha:
            print(f"ok        {rel} (sha256 verified)", flush=True)
            return local_path
        raise ValueError(
            f"{local_path} exists but does not match the checksum in {MANIFEST_RELPATH}. "
            "Delete it or pass --force to download it again."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Downloading needs huggingface_hub. Install it with: pip install huggingface_hub") from exc

    size_mb = float(entry.get("size_bytes", 0)) / 1e6
    print(f"download  {entry['hf_path']} from {repo_id} ({size_mb:,.1f} MB)", flush=True)
    hf_path = str(entry["hf_path"])
    if Path(hf_path) == rel:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=hf_path,
                revision=revision,
                local_dir=root,
                force_download=force,
            )
        )
    else:
        import shutil

        cached = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=hf_path, revision=revision)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, local_path)
        downloaded = local_path

    actual_sha = sha256_file(downloaded)
    if actual_sha != expected_sha:
        downloaded.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {rel}: {actual_sha} != {expected_sha}")
    print(f"ok        {rel} (sha256 verified)", flush=True)
    return downloaded
