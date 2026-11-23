"""Describe each masked grey-matter row against the mean of the other subjects."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.integrity import write_json


def measure(matrix: np.ndarray, people: list[str], recovered: set[str]) -> list[dict[str, Any]]:
    """Use float64 means and Pearson correlations, leaving the measured row out."""
    if len(people) != len(matrix) or len(people) < 2 or len(set(people)) != len(people):
        raise ValueError(f"quality control needs distinct matrix rows: {matrix.shape}, {len(people)} subjects")
    total = matrix.sum(axis=0, dtype=np.float64)
    rows = []
    for start in range(0, len(people), 32):
        x = matrix[start:start + 32].astype(np.float64)
        other = (total - x) / (len(people) - 1)
        means = x.mean(axis=1)
        x -= means[:, None]
        other -= other.mean(axis=1)[:, None]
        scale = np.sqrt(np.einsum("ij,ij->i", x, x) * np.einsum("ij,ij->i", other, other))
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.einsum("ij,ij->i", x, other) / scale
        for subject, value, mean in zip(people[start:start + 32], rs, means, strict=True):
            if not np.isfinite(value) or not np.isfinite(mean):
                raise ValueError(f"{subject}: undefined leave-one-out correlation {value}; mean {mean}")
            rows.append({"subject": subject, "loo_r": float(np.clip(value, -1.0, 1.0)),
                         "mean_gm": float(mean), "exact_public_crop_copy": subject in recovered})
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize both crop groups, the ten lowest correlations, and robust outlier flags."""
    groups = {}
    for is_copy, label in ((True, "exact_public_crop_copy"), (False, "cluster_rebuilt_crop")):
        values = [row["loo_r"] for row in rows if row["exact_public_crop_copy"] == is_copy]
        groups[label] = {"n": len(values), "median_loo_r": float(np.median(values)) if values else None,
                         "min_loo_r": min(values) if values else None}
    limits = {}
    for key in ("loo_r", "mean_gm"):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        center = np.median(values)
        width = 5.0 * 1.4826 * np.median(np.abs(values - center))
        limits[key] = {"lower": float(center - width), "upper": float(center + width)}
    flagged = [row["subject"] for row in rows if row["loo_r"] < limits["loo_r"]["lower"]
               or not limits["mean_gm"]["lower"] <= row["mean_gm"] <= limits["mean_gm"]["upper"]]
    return {"groups": groups, "lowest_10": sorted(rows, key=lambda row: (row["loo_r"], row["subject"]))[:10],
            "outlier_rule": "r below, or mean GM outside, median +/- 5 scaled median absolute deviations",
            "outlier_limits": limits, "flagged_subjects": flagged}


def save_qc(directory: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist row order, group summaries and the ranked low end before any promotion."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    write_json(directory / f"{name}.json", summary)
    return summary
