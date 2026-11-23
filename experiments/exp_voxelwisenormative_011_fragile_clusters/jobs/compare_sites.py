"""Compare one fold of the per-refit export from Slurm and Hugging Face: flag counts and cluster ranking, not bits.

    PYTHONPATH="$PWD" micromamba run -n voxelwise-normative-jax python \
        experiments/exp_voxelwisenormative_011_fragile_clusters/jobs/compare_sites.py \
        --slurm <slurm-run> --hugging-face <cloud-run> [--rows 0] [--single-fit-z PATH]

A run is a folder, or a name looked up under .results as fetch-<name>, <name> or hugging-face/<name>. Both runs must be
the same fold with the same draws and rule. Prints the flag totals, the recurrence counts and the fragile-cluster
ranking of both sites; --rows adds the per-person numbers the slides quote, and with --single-fit-z (exp_003's z_test.npy,
fold 0 only) the single fit's flags, how many were kept in all 64 refits, and the stable and fragile counts.
Reads only; --json writes one new file (never an existing one).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    """Profile the comparison by default; any failed check stops with the offending run named."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--slurm', required=True, help='the Slurm run (name or folder)')
    parser.add_argument("--hugging-face", required=True, help="the Hugging Face run (name or folder)")
    parser.add_argument("--rows", type=int, nargs="*", default=[], help="held-out rows (0-based) to report one by one")
    parser.add_argument("--single-fit-z", type=Path, help="exp_003's z_test.npy of the frozen split; fold 0 only")
    parser.add_argument("--json", type=Path, help="also write the numbers here; refuses to overwrite")
    parser.add_argument("--no-profile", action="store_true", help="skip the local HTML profile")
    args = parser.parse_args()
    sys.path[:0] = [str(EXPERIMENT_DIR), str(EXPERIMENT_DIR.parents[1])]
    from src.utils.paths import RESULTS
    from src.utils.profiling import profiled

    with profiled(None if args.no_profile else RESULTS / "profile" / "compare-sites.html"):
        import numpy as np
        from src.modules import site_comparison as sc
        from src.modules.clusters import ClusterGrid
        from src.modules.inputs import read_mask

        from shared.voxelwisenormative.artifacts import VOXEL_MATRIX, resolve

        slurm = sc.load('slurm', args.slurm, RESULTS)
        hugging_face = sc.load("hugging-face", args.hugging_face, RESULTS)
        mask, planes, _ = read_mask(resolve(VOXEL_MATRIX, "mask.nii.gz")["mask.nii.gz"])
        on_planes = np.isin(np.argwhere(mask)[:, 2], planes)
        report = sc.compare(slurm, hugging_face, ClusterGrid(mask, planes))
        z = np.load(args.single_fit_z, mmap_mode='r', allow_pickle=False) if args.single_fit_z else None
        table = sc.people(slurm, hugging_face, args.rows, z, on_planes) if args.rows else []
        for line in (*sc.render(report, 'slurm', 'hugging-face'), *sc.render_people(table, 'slurm', 'hugging-face')):
            print(line)
        if args.json:
            with args.json.open("x") as stream:
                json.dump({**report, "people": table}, stream, indent=2, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
