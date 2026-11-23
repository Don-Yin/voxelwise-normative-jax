"""CPU baseline: PCNtoolkit 1.4.0 warped BLR, one model per voxel, on real IXI grey matter.

Model (PCNtoolkit's documented BLR setup): covariates age (cubic B-spline, 5 knots) and sex;
site as a fixed-effect batch effect; SinhArcsinh warp; homoskedastic noise; default optimiser
(L-BFGS-B, finite-difference gradient, L2 penalty 0.1 on the hyperparameters).
Split: 80/20 by subject, stratified by site. Voxels: a seeded random subset of the mask.
Writes .results/<tag>/timing.json and arrays.npz (the reference a batched fit must reproduce).
"""

from __future__ import annotations

import argparse
import logging

from src.modules.pcntoolkit_baseline import run
from src.utils.paths import RESULTS
from src.utils.profiling import profiled, run_resources


def parse_args() -> argparse.Namespace:
    """parse command line arguments for this experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-voxels", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="L-BFGS-B finite-difference step; 0.1 is PCNtoolkit's default")
    parser.add_argument("--no-profile", action="store_true", help="skip the default pyinstrument html report")
    parser.add_argument("--resources", action="store_true", help="log wall, cpu, and peak rss for this run")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    with (
        run_resources(enabled=args.resources),
        profiled(None if args.no_profile else RESULTS / "profile" / "run.html"),
    ):
        run(args.n_voxels, args.seed, args.epsilon, RESULTS)
