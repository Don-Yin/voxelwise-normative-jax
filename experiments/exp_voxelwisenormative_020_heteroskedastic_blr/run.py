"""Compare PCNtoolkit and JAX heteroskedastic warped BLR on CPU."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled, run_resources


def parse_args() -> argparse.Namespace:
    """Parse the frozen input location, benchmark width and profiling options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, help="directory containing the frozen artifact manifests")
    parser.add_argument("--voxels", type=int, default=1000, help="number of frozen response columns to compare")
    parser.add_argument("--repeats", type=int, default=3, help="timed JAX passes after the first compiled fit")
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="skip the default pyinstrument html report",
    )
    parser.add_argument(
        "--resources",
        action="store_true",
        help="log wall, cpu, and peak rss for this run",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the experiment against the package in this checkout, forcing the CPU backend."""
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    os.environ["JAX_PLATFORMS"] = "cpu"
    from src.modules.benchmark import benchmark

    benchmark(args.voxels, args.repeats, root=args.artifacts_dir, profiled=not args.no_profile)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    with (
        run_resources(enabled=args.resources),
        profiled(None if args.no_profile else RESULTS / "profile" / "run.html"),
    ):
        main(args)
