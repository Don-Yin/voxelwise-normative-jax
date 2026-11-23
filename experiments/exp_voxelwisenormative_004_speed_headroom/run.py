"""time full-mask warped Bayesian linear regression variants and check their z-scores."""

from __future__ import annotations

import argparse
import logging

from src.utils.paths import RESULTS, add_project_root
from src.utils.profiling import profiled


def parse_args() -> argparse.Namespace:
    """select the benchmark stage and requested device count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "full"))
    parser.add_argument("--devices", type=int, choices=(1, 4), default=1)
    parser.add_argument("--gpu", action="store_true", help="fail unless CUDA devices are available")
    parser.add_argument("--variant", action="append", help="run specific variant(s), without a chunk sweep")
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--single-job", help="job ID of the single-device sweep to replicate on four GPUs")
    parser.add_argument("--no-profile", action="store_true", help="skip the default pyinstrument html report")
    args = parser.parse_args()
    if args.repeats < 1 or args.chunk < 1:
        parser.error(f"repeats and chunk must be positive: {args.repeats=}, {args.chunk=}")
    if args.single_job and (args.devices != 4 or not args.single_job.isascii() or not args.single_job.isdigit()):
        parser.error(f"a single-job ID must be numeric and used with four GPUs: {args.single_job=}, {args.devices=}")
    return args


def main() -> None:
    """invoke the benchmark with default-on host profiling outside Slurm."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    add_project_root()
    from src.modules.benchmark import run

    with profiled(None if args.no_profile else RESULTS / "profile" / "run.html"):
        run(args, RESULTS)


if __name__ == "__main__":
    main()
