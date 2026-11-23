"""Calibrate, fit, join or audit the Slurm full-mask PCNtoolkit comparison."""

from __future__ import annotations

import argparse
import logging
import sys

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled, run_resources


def parse_args() -> argparse.Namespace:
    """Parse the stage, shard identity and explicit profiling opt-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-profile", action="store_true", help="skip inner profiling under Slurm")
    parser.add_argument("--resources", action="store_true", help="log local CPU, wall time and peak RAM")
    stage = parser.add_subparsers(dest="stage", required=True)
    calibrate = stage.add_parser("calibrate")
    calibrate.add_argument("--parity-job", help="reuse previously saved and verified 20-voxel pairs")
    fit = stage.add_parser("fit-shard")
    join = stage.add_parser("merge")
    audit = stage.add_parser("audit-outliers")
    bench = stage.add_parser("gradient-benchmark")
    stops = stage.add_parser("stopping-sweep")
    for sub in (fit, join, audit, bench, stops):
        sub.add_argument("--calibration", required=True)
        sub.add_argument("--array", required=True)
    for sub in (fit, join):
        sub.add_argument("--shards", required=True, type=int)
    fit.add_argument("--index", required=True, type=int)
    for sub in (audit, bench, stops):
        sub.add_argument("--merge", required=True)
    for sub in (bench, stops):
        sub.add_argument("--audit", required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Dispatch the specified stage without starting anything on import."""
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    from src.modules.campaign import calibrate, fit_shard

    if args.stage == "calibrate":
        calibrate(RESULTS, args.parity_job)
    elif args.stage == "fit-shard":
        fit_shard(RESULTS, args.calibration, args.array, args.index, args.shards)
    elif args.stage == "merge":
        from src.modules.compare import merge

        merge(RESULTS, args.calibration, args.array, args.shards)
    elif args.stage == "audit-outliers":
        from src.modules.audit_outliers import audit

        audit(RESULTS, args.array, args.merge, args.calibration)
    elif args.stage == "gradient-benchmark":
        from src.modules.gradient_benchmark import benchmark

        benchmark(RESULTS, args.array, args.merge, args.calibration, args.audit)
    else:
        from src.modules.stopping_sweep import sweep

        sweep(RESULTS, args.array, args.merge, args.calibration, args.audit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    with (
        run_resources(enabled=args.resources),
        profiled(None if args.no_profile else RESULTS / "profile" / "run.html"),
    ):
        main(args)
