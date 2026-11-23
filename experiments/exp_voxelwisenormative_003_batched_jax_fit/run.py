"""Batched warped BLR in JAX: every voxel's PCNtoolkit model fitted at once, checked against PCNtoolkit.

The fit itself lives in shared/voxelwisenormative/batched_blr.py.
Modes:
  check   fit the 1,000 reference voxels of pcntoolkit-blr-reference-n1000-v1 and compare z, objective
  full    fit every mask voxel with the reference split and design; time it and compare the same voxels
"""

from __future__ import annotations

import argparse
import logging

from src.modules.benchmark import run
from src.utils.paths import RESULTS
from src.utils.profiling import profiled, run_resources


def parse_args() -> argparse.Namespace:
    """parse command line arguments for this experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "full"))
    parser.add_argument("--n-iter", type=int, default=150, help="150 reaches converged PCNtoolkit on the check set")
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats after one warm-up")
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
        run(args.mode, args.n_iter, args.chunk, args.repeats, RESULTS)
