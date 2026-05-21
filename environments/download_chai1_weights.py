#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path


CHAI_BASE_URL = "https://chaiassets.com/chai1-inference-depencencies/"
CHAI_ASSETS = [
    "conformers_v1.apkl",
    "models_v2/trunk.pt",
    "models_v2/token_embedder.pt",
    "models_v2/feature_embedding.pt",
    "models_v2/diffusion_module.pt",
    "models_v2/confidence_head.pt",
]
ESM_REPO_ID = "facebook/esm2_t36_3B_UR50D"


def download_url(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"present {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    print(f"download {url} -> {destination}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temporary.replace(destination)


def download_chai_assets(weights_dir: Path, force: bool = False) -> None:
    for asset in CHAI_ASSETS:
        download_url(CHAI_BASE_URL + asset, weights_dir / asset, force=force)


def download_esm_assets(weights_dir: Path, force: bool = False) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "ESM weight download requires huggingface_hub in the active environment. "
            "Install it with: python -m pip install huggingface_hub"
        ) from exc

    hf_home = weights_dir / "huggingface"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    print(f"download {ESM_REPO_ID} -> {os.environ['HF_HUB_CACHE']}")
    snapshot_download(
        repo_id=ESM_REPO_ID,
        cache_dir=os.environ["HF_HUB_CACHE"],
        local_files_only=False,
        force_download=force,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Chai-1 and ESM model weights without running inference.")
    parser.add_argument("--weights-dir", default="/scratch/xz2/chai-1_weights/")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist.")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir).expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    download_chai_assets(weights_dir, force=args.force)
    download_esm_assets(weights_dir, force=args.force)
    print(f"ready {weights_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
