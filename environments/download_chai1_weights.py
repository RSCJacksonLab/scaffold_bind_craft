#!/usr/bin/env python3
from __future__ import annotations

import argparse
import grp
import os
import stat
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
    "models_v2/bond_loss_input_proj.pt",
    "esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt",
]
ESM_DESTINATION = "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
DEFAULT_GROUP = "xz2"


def shared_permission_error(path: Path, group: str) -> PermissionError:
    return PermissionError(
        f"Could not make {path} writable by group {group}. If this is a shared "
        "weights directory owned by another user, ask the owner to run: "
        f"chgrp -R {group} {path} && chmod -R g+rwx {path} && "
        f"find {path} -type d -exec chmod g+s {{}} +"
    )


def apply_shared_permissions(path: Path, group: str | None) -> None:
    if group is None:
        return
    try:
        gid = grp.getgrnam(group).gr_gid
    except KeyError as exc:
        raise RuntimeError(f"Group does not exist: {group}") from exc

    paths = [path]
    if path.is_dir():
        for root, dirs, files in os.walk(path):
            root_path = Path(root)
            paths.extend(root_path / name for name in dirs)
            paths.extend(root_path / name for name in files)

    for item in paths:
        if not item.exists() or item.is_symlink():
            continue
        try:
            current = item.stat()
            if current.st_gid != gid:
                os.chown(item, -1, gid)
            mode = item.stat().st_mode | stat.S_IRWXG
            if item.is_dir():
                mode |= stat.S_ISGID
            os.chmod(item, mode)
        except PermissionError as exc:
            raise shared_permission_error(path, group) from exc


def download_url(
    url: str,
    destination: Path,
    force: bool = False,
    group: str | None = DEFAULT_GROUP,
) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        apply_shared_permissions(destination, group)
        print(f"present {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    apply_shared_permissions(destination.parent, group)
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
    apply_shared_permissions(temporary, group)
    temporary.replace(destination)
    apply_shared_permissions(destination, group)


def download_chai_assets(
    weights_dir: Path,
    force: bool = False,
    group: str | None = DEFAULT_GROUP,
) -> None:
    for asset in CHAI_ASSETS:
        destination = weights_dir / asset
        if asset.startswith("esm2/"):
            destination = weights_dir / ESM_DESTINATION
        download_url(CHAI_BASE_URL + asset, destination, force=force, group=group)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Chai-1 and ESM model weights without running inference.")
    parser.add_argument("--weights-dir", default="/scratch/xz2/chai-1_weights/")
    parser.add_argument("--group", default=os.environ.get("CHAI_WEIGHTS_GROUP", DEFAULT_GROUP), help="Group that should be able to read, write, and traverse staged weights. Use an empty value to skip chmod/chgrp.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist.")
    args = parser.parse_args()

    group = args.group or None
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    apply_shared_permissions(weights_dir, group)
    download_chai_assets(weights_dir, force=args.force, group=group)
    apply_shared_permissions(weights_dir, group)
    print(f"ready {weights_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
