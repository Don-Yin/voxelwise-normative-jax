"""Compare the stored voxels of the 57 rebuilt crops against their exact public originals."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import numpy as np
from nibabel.arrayproxy import ArrayProxy

from ..utils.integrity import expect, head_rows, write_json
from ..utils.paths import RESULTS, STAGED_INDEX
from .compare import load_image
from .staging import load_staged

logger = logging.getLogger(__name__)
def staged_digest() -> str:
    return expect(STAGED_INDEX, os.environ['IXI_STAGED_SHA256'])


def stored_voxels(path: Path) -> np.ndarray:
    """Read voxel samples without applying header transforms or intensity scaling."""
    proxy = load_image(path).dataobj
    if not isinstance(proxy, ArrayProxy):
        raise TypeError(f"{path}: expected an on-disk NIfTI voxel array")
    return np.asarray(proxy.get_unscaled())


def compare_crops(rebuilt_dir: Path) -> None:
    """Write one comparison row per original crop used to restore the published byte stream."""
    stage_sha = staged_digest()
    rows = head_rows()
    staged = load_staged(rows)
    recovered = [s for s in rows if staged[s].get("recovery") == "exact public source copy"]
    if len(recovered) != 57:
        raise ValueError(f"{STAGED_INDEX}: expected 57 recovered crops, got {len(recovered)}")
    checks = []
    for i, subject in enumerate(recovered, start=1):
        rebuilt = rebuilt_dir / subject / 'cropped.nii.gz'
        original = Path(staged[subject]["folder"]) / "cropped.nii.gz"
        expect(original, rows[subject]["cropped_sha256"])
        a, b = stored_voxels(rebuilt), stored_voxels(original)
        bad = int(np.count_nonzero(a != b)) if a.shape == b.shape else None
        checks.append({"subject": subject, "equal": bad == 0, "different_voxels": bad,
                       "rebuilt_shape": "x".join(map(str, a.shape)), "original_shape": "x".join(map(str, b.shape))})
        logger.info("[%d/%d] %s crop voxel comparison: %s", i, len(recovered), subject, bad)
    different = [r["subject"] for r in checks if not r["equal"]]
    table = RESULTS / "tables" / "recovered-crop-voxels.tsv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(checks)
    write_json(RESULTS / "json" / "recovered-crop-voxels.json", {
        'job': os.environ.get('SLURM_JOB_ID'),
        'staged_manifest_sha256': stage_sha, 'n_equal': len(checks) - len(different),
        "n_different": len(different), "different_subjects": different,
        "comparison": "unscaled stored NIfTI voxel samples; no header comparison",
    })
