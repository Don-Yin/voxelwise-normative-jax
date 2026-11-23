"""Batch K bootstrap refits per compiled GPU call: effective time, memory and utilisation, z checked against K=1."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled, run_resources
from src.utils.receipts import RUN_NAME


def run_name(value: str) -> str:
    """Only simple identifiers may enter experiment-owned result paths."""
    if not RUN_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError(f"run name must have 1–64 lowercase letters, digits or hyphens: {value!r}")
    return value


def parse_args() -> argparse.Namespace:
    """Parse one precision's sweep, or the join of a run's receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-profile", action="store_true", help="skip inner profiling under slurm-profile")
    parser.add_argument("--resources", action="store_true", help="log total time and peak host RSS")
    sub = parser.add_subparsers(dest="stage", required=True)
    sweep = sub.add_parser("sweep", help="K=1,2,...,64 refits per call for one precision")
    sweep.add_argument("--precision", choices=("float32", "float64"), required=True)
    sweep.add_argument("--run-name", type=run_name)
    sweep.add_argument("--chunk", type=int, help="voxels per compiled call; default 32768 (exp_010), 64 synthetic")
    sweep.add_argument("--synthetic", action="store_true", help="tiny CPU cohort; never opens IXI")
    collect = sub.add_parser("collect", help="join receipts, GPU samples and traces into tables")
    collect.add_argument("--run-name", type=run_name, required=True)
    args = parser.parse_args()
    if args.stage == "sweep":
        if args.run_name is None and args.synthetic:
            args.run_name = f"synthetic-{time.time_ns()}"
        if args.run_name is None or args.synthetic != args.run_name.startswith("synthetic-"):
            parser.error("synthetic runs need a synthetic- name; real runs need another explicit --run-name")
        if args.chunk is None:
            args.chunk = 64 if args.synthetic else 32_768
        if args.chunk < 1:
            parser.error(f"chunk must be positive, got {args.chunk}")
    return args


def work(args: argparse.Namespace) -> None:
    """Dispatch to src/ without importing JAX for collect."""
    if args.stage == "sweep":
        from src.modules.sweep import run

        run(RESULTS, args.run_name, args.precision, args.chunk, smoke=args.synthetic)
    else:
        from src.modules.collect import collect

        collect(RESULTS, args.run_name)


def main() -> None:
    """Provide repo imports and profile the actual work by default."""
    args = parse_args()
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with run_resources(enabled=args.resources), profiled(None if args.no_profile else RESULTS / "profile" / "run.html"):
        work(args)


if __name__ == "__main__":
    main()
