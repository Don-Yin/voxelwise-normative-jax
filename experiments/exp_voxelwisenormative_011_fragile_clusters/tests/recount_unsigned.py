"""Recount the single fit's clusters with the two signs labelled separately and as one |z| mask, with plain scipy.

    PYTHONPATH="$PWD" micromamba run -n voxelwise-normative-jax python \
        experiments/exp_voxelwisenormative_011_fragile_clusters/tests/recount_unsigned.py [--folds 0 1 2 3 4] [--threshold 1.96]

The cluster-share runs label positive (z > T) and negative (z < -T) regions separately. A recount from abs(z) > T as one
mask joins a positive and a negative region that are 26-neighbours into one cluster, so it finds one cluster fewer per such
contact. This script prints both counts by size class and every contact, so a difference between two independent counts can
be traced to it. It reads only the mask and each fold's z-summary; the flags are abs(original_z) > T in float64.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
from scipy import ndimage

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT), str(EXPERIMENT.parents[1])]
CLASSES = {"all": 1, "ge10": 10, "ge50": 50, "ge100": 100}
STRUCTURE = np.ones((3, 3, 3), dtype=bool)  # 26 neighbours
Z_SUMMARY = ".results/hugging-face/hugging-face-a100-z-summary-fold{fold}/arrays/hugging-face-a100-z-summary-fold{fold}/z-summary.npz"


def label(mask: np.ndarray, flags: np.ndarray) -> tuple[np.ndarray, int]:
    """26-neighbour components of the flagged voxels (one value per mask column), on the mask's grid."""
    grid = np.zeros(mask.shape, dtype=bool)
    grid[mask] = flags
    labelled = np.zeros(mask.shape, dtype=np.int32)
    n = ndimage.label(grid, structure=STRUCTURE, output=labelled)  # with an output array, scipy returns only the count
    return labelled, cast(int, n)


def sizes(labelled: np.ndarray, n: int) -> list[int]:
    return np.bincount(labelled.ravel(), minlength=n + 1)[1:].tolist()


def recount(mask: np.ndarray, z: np.ndarray, threshold: float) -> tuple[list[int], list[int], list[tuple[int, int, list[int], list[int]]]]:
    """Cluster sizes of every person labelled by sign and as one mask, and the contacts: (person, merged size, positive parts, negative parts)."""
    signed, unsigned, contacts = [], [], []
    for person, row in enumerate(z):
        positive, negative = row > threshold, row < -threshold
        pos_labels, n_pos = label(mask, positive)
        neg_labels, n_neg = label(mask, negative)
        all_labels, n_all = label(mask, positive | negative)
        signed += sizes(pos_labels, n_pos) + sizes(neg_labels, n_neg)
        unsigned += sizes(all_labels, n_all)
        if n_pos + n_neg != n_all:
            for part in range(1, n_all + 1):
                inside = all_labels == part
                pos_parts, neg_parts = np.unique(pos_labels[inside & (pos_labels > 0)]), np.unique(neg_labels[inside & (neg_labels > 0)])
                if len(pos_parts) and len(neg_parts):
                    contacts.append((person, int(inside.sum()), [int((pos_labels == k).sum()) for k in pos_parts], [int((neg_labels == k).sum()) for k in neg_parts]))
    return signed, unsigned, contacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--threshold", type=float, default=1.96)
    args = parser.parse_args()
    from src.modules.inputs import (
        read_mask,  # imported here: sys.path is extended above
    )

    from shared.voxelwisenormative.artifacts import VOXEL_MATRIX, resolve

    mask, _, _ = read_mask(resolve(VOXEL_MATRIX, "mask.nii.gz")["mask.nii.gz"])
    total = {"signed": np.zeros(len(CLASSES), dtype=int), "unsigned": np.zeros(len(CLASSES), dtype=int)}
    for fold in args.folds:
        with np.load(EXPERIMENT / Z_SUMMARY.format(fold=fold)) as archive:
            z = archive["original_z"]
        signed, unsigned, contacts = recount(mask, z, args.threshold)
        for name, found in (("signed", signed), ("unsigned", unsigned)):
            counts = np.array([int((np.asarray(found) >= k).sum()) for k in CLASSES.values()])
            total[name] += counts
            print(f"fold {fold}, {len(z)} people, abs(z) > {args.threshold:g}, 26 neighbours, {name:8s}: " + "  ".join(f"{c} {n:,}" for c, n in zip(CLASSES, counts, strict=True)))
        for person, merged, positive, negative in contacts:
            print(f"  contact: fold {fold} person {person}: positive parts {positive} and negative parts {negative} form one unsigned cluster of {merged} voxels")
    print("all folds given: " + "; ".join(f"{name} " + "  ".join(f"{c} {n:,}" for c, n in zip(CLASSES, counts, strict=True)) for name, counts in total.items()))


if __name__ == "__main__":
    main()
