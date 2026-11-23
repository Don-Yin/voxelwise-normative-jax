"""Rank unstable whole-brain flag clusters in the original IXI held-out split."""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled, run_resources


def cutoff_list(text: str) -> tuple[float, ...]:
    """`2.3,2.5,3.0` as floats; the rules for the values are validate_cutoffs's."""
    try:
        return tuple(float(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected numbers separated by commas, got {text!r}") from error


def parse_args() -> argparse.Namespace:
    """Validate a safe output tag and the explicitly selected runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--synthetic", action="store_true", help="tiny local fixture, no CUDA or IXI matrix")
    parser.add_argument("--per-refit", action="store_true",
                        help="only the 64 draws; keep every refit's flags for all 113 held-out scans")
    parser.add_argument("--fold", type=int, default=0, choices=range(5), metavar="{0..4}",
                        help="with --per-refit: cross-fit fold; 0 (default) is the frozen 113-scan split")
    parser.add_argument("--reference-n", type=int, metavar="N",
                        help="with --per-refit, fold 0: keep N of the references (age-stratified, fixed seed), "
                             "fit them once and draw the 64 samples from them; off by default")
    parser.add_argument("--z-summary", action="store_true",
                        help="with --per-refit: also fit once without resampling and keep that fit's z with every voxel's smallest "
                             "and largest |z| over the 64 refits (z-summary.npz, 3 float64 arrays), so any |z| threshold can be "
                             "applied afterwards; the other outputs do not change")
    parser.add_argument("--refit-flag-cutoffs", type=cutoff_list, default=(), metavar="T,T,...",
                        help="with --z-summary: also keep every refit's flags and the single fit's at these cutoffs, each above "
                             "1.96, packed and compressed (cutoff-flags.npz); the run stops unless their AND and OR agree with "
                             "the smallest and largest |z|")
    parser.add_argument("--resources", action="store_true", help="log wall time, CPU and peak RSS")
    parser.add_argument("--no-profile", action="store_true", help="skip local HTML when using slurm-profile")
    args = parser.parse_args()
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))  # the project root is not on sys.path before main()
    from shared.voxelwisenormative.batch_job import expected_environment

    try:
        environment = expected_environment()
    except ValueError as error:
        parser.error(str(error))
    if Path(sys.prefix).name != environment:
        parser.error(f"run in {environment}, not {sys.prefix}")
    if args.run_name is None and args.synthetic:
        args.run_name = datetime.now(UTC).strftime("synthetic-%Y%m%d-%H%M%S-%f")
    if not args.run_name or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_name):
        parser.error("run-name must use 1–64 lowercase letters, digits or hyphens")
    if args.synthetic != args.run_name.startswith("synthetic-"):
        parser.error("prefix synthetic runs with synthetic-; reserve all other names for CUDA jobs")
    if args.chunk < 1:
        parser.error("chunk must be positive")
    if args.fold and not args.per_refit:
        parser.error("--fold needs --per-refit")
    if args.z_summary and not args.per_refit:
        parser.error("--z-summary needs --per-refit")
    if args.refit_flag_cutoffs:
        from src.modules.z_summary import validate_cutoffs

        try:
            args.refit_flag_cutoffs = validate_cutoffs(args.refit_flag_cutoffs, args.z_summary)
        except ValueError as error:
            parser.error(f"--refit-flag-cutoffs: {error}")
    if args.reference_n is not None and (not args.per_refit or args.fold or args.reference_n < 3):
        parser.error(f"--reference-n needs --per-refit, fold 0 and N >= 3, got {args.reference_n}, fold {args.fold}")
    return args


def main() -> None:
    """Profile the actual work by default and propagate failures."""
    args = parse_args()
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    with (run_resources(enabled=args.resources),
          profiled(None if args.no_profile else RESULTS / "profile" / "run.html")):
        from src.modules.study import run, run_per_refit

        if args.per_refit:
            run_per_refit(RESULTS, args.run_name, args.chunk, use_fixture=args.synthetic, fold=args.fold,
                          reference_n=args.reference_n, z_summary=args.z_summary, cutoffs=args.refit_flag_cutoffs)
        else:
            run(RESULTS, args.run_name, args.chunk, use_fixture=args.synthetic)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
