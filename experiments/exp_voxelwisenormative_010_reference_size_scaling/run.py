"""Measure voxelwise reference-size scaling, aggregate bootstrap recurrence and estimate fit cost."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled, run_resources


def run_name(value: str) -> str:
    """Only simple identifiers may enter experiment-owned result paths."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value):
        raise argparse.ArgumentTypeError(f"run name must have 1–64 lowercase letters, digits or hyphens: {value!r}")
    return value


def parse_args() -> argparse.Namespace:
    """Parse indexed tasks and result collection without importing the fit runtimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-profile", action="store_true", help="skip inner profiling under slurm-profile")
    parser.add_argument("--resources", action="store_true", help="log total time and peak host RSS")
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("cpu", "gpu", "bootstrap"):
        job = sub.add_parser(name)
        job.add_argument("--index", type=int, required=True)
        job.add_argument("--run-name", type=run_name, required=True)
    smoke = sub.add_parser("smoke", help="run a tiny synthetic CPU-only fit")
    smoke.add_argument("--engine", choices=("pcntoolkit", "jax", "bootstrap"), required=True)
    smoke.add_argument("--precision", choices=("float32", "float64"), default="float32")
    collect = sub.add_parser("collect", help="validate and combine fixed-grid aggregate receipts")
    collect.add_argument("--run-name", type=run_name, required=True)
    for kind in ("cpu", "gpu", "bootstrap"):
        collect.add_argument(f"--{kind}-run", action="append", type=run_name, required=True)
    cost = sub.add_parser("cost", help="estimate 1 fit + 64 refits for a requested N")
    cost.add_argument("--run-name", type=run_name, required=True)
    cost.add_argument("--n", type=int, required=True)
    return parser.parse_args()


def work(args: argparse.Namespace) -> None:
    """Dispatch a single process working at one configuration, or aggregate small receipts."""
    from shared.voxelwisenormative.reference_grid import BOOT_CASES, CPU_CASES, GPU_CASES, Cell, case_at

    if args.stage == "smoke":
        out = RESULTS / "json" / "smoke" / f"local-{time.time_ns()}" / f"{args.engine}.json"
        cell = Cell("synthetic", 25)
        if args.engine == "pcntoolkit":
            from src.modules.cpu import worker

            worker(cell, 4, out)
        elif args.engine == "jax":
            from src.modules.gpu import benchmark

            benchmark(cell, args.precision, out, smoke=True)
        else:
            from shared.voxelwisenormative.reference_bootstrap import worker

            worker(cell, 0, out, smoke=True)
        return
    folder = RESULTS / "json" / args.stage / args.run_name
    if args.stage == "cpu":
        from src.modules.cpu import worker

        cell, width = case_at(CPU_CASES, args.index)
        worker(cell, width, folder / f"{cell.label}-v{width:03d}.json")
    elif args.stage == "gpu":
        from src.modules.gpu import benchmark

        cell, precision = case_at(GPU_CASES, args.index)
        benchmark(cell, precision, folder / f"{cell.label}-{precision}.json")
    elif args.stage == "bootstrap":
        from shared.voxelwisenormative.reference_bootstrap import worker

        cell, chain = case_at(BOOT_CASES, args.index)
        worker(cell, chain, folder / f"{cell.label}-chain{chain}.json")
    else:
        from src.modules.collect import collect, save_cost

        if args.stage == "collect":
            collect(RESULTS, args.run_name, args.cpu_run, args.gpu_run, args.bootstrap_run)
        else:
            save_cost(RESULTS, args.run_name, args.n)


def main() -> None:
    """Provide repo imports and profile the actual work by default."""
    args = parse_args()
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with (run_resources(enabled=args.resources), profiled(None if args.no_profile else RESULTS / "profile" / "run.html")):
        work(args)


if __name__ == "__main__":
    main()
