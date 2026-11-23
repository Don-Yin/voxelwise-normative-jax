"""What share of the clusters flagged in one held-out person's map is fragile across the 64 refits? Post-processing only.

    PYTHONPATH="$PWD" micromamba run -n voxelwise-normative-jax python \
        experiments/exp_voxelwisenormative_011_fragile_clusters/jobs/cluster_share.py --run-name fragile-cluster-share [--folds 0 1 2 3 4]
        [--threshold 3.1]

Reads the fetched runs .results/hugging-face/hugging-face-a100-z-summary-fold<k>/ (refit-flags.npz and z-summary.npz; digests
checked against each receipt); with a --threshold other than 1.96 the runs ...-a100-cutoff-flags-fold<k>/, whose cutoff-flags.npz
holds each refit's flags at 2.3, 2.5, 3.0, 3.1 and 3.5. It labels the clusters of every person's single-fit map at 26, 18 and 6
neighbours, measures each against the 64 refits, and writes .results/{arrays,json,tables}/<run-name>/ (never overwrites). No fit, no GPU, no IXI matrix.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    """Profile the labelling by default; any failed check stops with the offending run named."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True, help="new output folder name under .results/{arrays,json,tables}")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], choices=range(5), metavar="{0..4}")
    parser.add_argument("--threshold", type=float, default=1.96, help="abs(z) cutoff: 1.96, or one of the cutoffs of cutoff-flags.npz")
    parser.add_argument("--no-profile", action="store_true", help="skip the local HTML profile")
    args = parser.parse_args()
    sys.path[:0] = [str(EXPERIMENT_DIR), str(EXPERIMENT_DIR.parents[1])]
    from src.utils.paths import RESULTS
    from src.utils.profiling import profiled

    with profiled(None if args.no_profile else RESULTS / "profile" / f"cluster-share-{args.run_name}.html"):
        from src.modules.share_run import run

        from shared.voxelwisenormative.artifacts import VOXEL_MATRIX, resolve

        run(RESULTS, args.run_name, sorted(set(args.folds)), resolve(VOXEL_MATRIX, "mask.nii.gz")["mask.nii.gz"],
            args.threshold)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
