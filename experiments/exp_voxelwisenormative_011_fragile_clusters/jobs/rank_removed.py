"""Rank the 113 held-out scans of fetched --reference-n runs by the original flags their 64 refits remove.

    micromamba run -n voxelwise-normative-jax python \
        experiments/exp_voxelwisenormative_011_fragile_clusters/jobs/rank_removed.py <fetched folder>...

Each folder holds one run's refit-flags.npz and result.json. Removed = flags of the one fit on all N references that
recur in fewer than 52 of 64 refits. Writes removed-ranking.tsv beside each pair; never overwrites one.

For a run without original flags, use --original-z with experiment 003's z_test.npy or the run's z-summary.npz:

    python jobs/rank_removed.py --original-z <scores> <fetched-folder>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    """Profile the ranking by default; any failed check stops with the offending folder named."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--original-z", type=Path, help="z of the one fit of a full-size run (z_test.npy or z-summary.npz); one folder")
    parser.add_argument("--no-profile", action="store_true", help="skip the local HTML profile")
    args = parser.parse_args()
    if args.original_z is not None and len(args.folders) != 1:
        parser.error("--original-z belongs to one run: pass exactly one folder")
    sys.path[:0] = [str(EXPERIMENT_DIR), str(EXPERIMENT_DIR.parents[1])]
    from src.utils.paths import RESULTS
    from src.utils.profiling import profiled

    with profiled(None if args.no_profile else RESULTS / "profile" / "rank-removed.html"):
        import numpy as np
        from src.modules.inputs import read_mask
        from src.modules.reference_size import rank_fetched

        from shared.voxelwisenormative.artifacts import VOXEL_MATRIX, resolve

        mask, planes, _ = read_mask(resolve(VOXEL_MATRIX, "mask.nii.gz")["mask.nii.gz"])
        on_planes = np.isin(np.argwhere(mask)[:, 2], planes)
        for folder in args.folders:
            rank_fetched(folder.resolve(), on_planes, args.original_z.resolve() if args.original_z else None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
