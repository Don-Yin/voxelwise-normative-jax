"""resolve a frozen experiments/shared/artifacts payload and verify it against its manifest."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "experiments" / "shared" / "artifacts"
VOXEL_MATRIX = "ixi-grey-matter-voxel-matrix-v1"
PCNTOOLKIT_REFERENCE = "pcntoolkit-blr-reference-n1000-v1"

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    """sha256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(artifact_id: str, *names: str) -> dict[str, Path]:
    """paths of the named payload files, each checked against the manifest sha256."""
    root = ARTIFACTS / artifact_id
    manifest = json.loads((root / "manifest.json").read_text())
    listed = {item["path"]: item for item in manifest["files"]}
    paths = {}
    for name in names:
        if name not in listed:
            raise KeyError(f"{name} is not listed in {root / 'manifest.json'}")
        path = root / name
        expected = listed[name].get("sha256")
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise RuntimeError(f"{path} sha256 {actual} differs from manifest {expected}")
        paths[name] = path
    logger.info("resolved %s: %s", artifact_id, ", ".join(names))
    return paths
