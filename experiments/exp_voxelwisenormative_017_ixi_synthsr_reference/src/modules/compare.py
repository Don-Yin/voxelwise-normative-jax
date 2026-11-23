"""Verify base map hashes and compare public SynthSR controls across two runs."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
from statistics import median

import nibabel as nib
import numpy as np

from ..utils.integrity import expect, sha256, write_json
from ..utils.paths import EXPERIMENTS, RESULTS
from .reference import BASE, CONTROLS

logger = logging.getLogger(__name__)
MASK = EXPERIMENTS / "shared/artifacts/ixi-grey-matter-voxel-matrix-v1/mask.nii.gz"
MASK_SHA = os.getenv('IXI_MASK_SHA256', '3adc14b9c2681fc7a461e520c73933c1f50498b65ea6b13445fcb66b80529caf')
MAP_NAME = "gm-modulated-s8.nii.gz"
THRESHOLD = 0.99


def load_image(path: Path) -> nib.Nifti1Image:
    """Require a NIfTI-1 image rather than accepting another file type."""
    image = nib.load(path)
    if not isinstance(image, nib.Nifti1Image):
        raise ValueError(f"{path}: expected NIfTI-1, got {type(image).__name__}")
    return image


def map_values(path: Path, grid: nib.Nifti1Image) -> np.ndarray:
    """Check the grid and finite values before using an image in a pair."""
    image = load_image(path)
    a, b = image.affine, grid.affine
    if a is None or b is None or image.shape != grid.shape or not np.allclose(a, b, atol=1e-5, rtol=0.0):
        raise ValueError(f"{path}: grid differs from {MASK}")
    data = image.get_fdata(dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: map contains nonfinite values")
    return data


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, with a hard error for constant data."""
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    if not math.isfinite(r):
        raise ValueError("cannot compare maps with a constant vector")
    return r


def check_pair(cam: Path, mac: Path, mask: np.ndarray, grid: nib.Nifti1Image) -> dict[str, float]:
    """Compare on the frozen mask and record whole-grid agreement as a cross-check."""
    a, b = map_values(cam, grid), map_values(mac, grid)
    return {"r_frozen_mask": correlation(a[mask], b[mask]), "r_full_grid": correlation(a, b),
            "max_abs_diff_frozen_mask": float(np.max(np.abs(a[mask] - b[mask])))}


def compare(folder: Path, comparison: Path) -> None:
    """Require all expected receipts and publish evidence even when the comparison fails."""
    expect(MASK, MASK_SHA)
    grid = load_image(MASK)
    mask = np.asanyarray(grid.dataobj).astype(bool)
    if np.count_nonzero(mask) != 165_402:
        raise ValueError(f"{MASK}: expected 165402 mask voxels, found {mask.sum()}")
    if comparison.resolve() == (folder / 'sr').resolve():
        raise ValueError('comparison directory must be an independent run')
    checks = []
    for route, people in (("base", BASE), ("sr", CONTROLS)):
        if {p.name for p in (folder / route).iterdir()} != set(people):
            raise ValueError(f"{folder / route}: downloaded subject set does not equal {people}")
        for subject in people:
            unit = folder / route / subject
            receipt = json.loads((unit / "receipt.json").read_text())
            if receipt["subject"] != subject or receipt["route"] != route:
                raise ValueError(f"{unit}: receipt belongs to {receipt['subject']} {receipt['route']}")
            path = unit / MAP_NAME
            digest = expect(path, receipt["map_sha256"])
            result = {k: receipt[k] for k in ("subject", "route", "job", "seconds", "peak_process_rss_mib",
                                            "cpu_seconds_commands", "threads_synthsr", "map_sha256",
                                            "expected_map_sha256", "status")}
            if route == "base":
                result["passed"] = receipt["status"] == "complete" and digest == receipt["expected_map_sha256"]
            else:
                control = comparison / subject / MAP_NAME
                result.update(check_pair(path, control, mask, grid))
                result['comparison_map_sha256'] = sha256(control)
                result["synthsr_seconds"] = receipt["commands"][0]["seconds"]
                result["synthsr_peak_rss_mib"] = receipt["commands"][0]["peak_rss_mib"]
                result["passed"] = receipt["status"] == "complete" and result["r_frozen_mask"] >= THRESHOLD
            checks.append(result)
    sr = [r for r in checks if r["route"] == "sr"]
    threads, = {r["threads_synthsr"] for r in sr}
    slowest = max(r["seconds"] for r in sr)
    request_minutes = math.ceil(1.15 * slowest / 60)
    report = {"mask_voxels": int(mask.sum()), "threshold": THRESHOLD, "subjects": checks,
              "passed": all(r["passed"] for r in checks),
              "projection": {"n_people": 563, "cpus_per_person": threads,
                             "median_seconds": median(r["seconds"] for r in sr),
                             "slowest_seconds": slowest,
                             "observed_cpu_hours": 563 * median(r["cpu_seconds_commands"] for r in sr) / 3600,
                             "requested_minutes_from_slowest_plus_15pct": request_minutes,
                             "requested_cpu_hours": 563 * threads * request_minutes / 60}}
    write_json(RESULTS / "json" / "small-test.json", report)
    fields = list(dict.fromkeys(k for row in checks for k in row))
    table = RESULTS / "tables" / "small-test.tsv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(checks)
    logger.info("exact base matches %s/3; synthsr correlation ≥ %.2f %s/12; report: %s",
                sum(r["passed"] for r in checks if r["route"] == "base"), THRESHOLD,
                sum(r["passed"] for r in sr), table)
    if not report["passed"]:
        raise ValueError(f"small-scale checks failed: {[r['subject'] for r in checks if not r['passed']]}")

