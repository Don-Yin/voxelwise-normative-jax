"""Stage, verify, and process the public IXI SynthSR reference."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from src.utils.paths import RESULTS
from src.utils.profiling import profiled, run_resources


def parse_args() -> argparse.Namespace:
    """Select a staging, registration or local verification stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-profile", action="store_true", help="use the external Slurm profiler alone")
    parser.add_argument("--resources", action="store_true", help="log local wall/CPU time and peak RSS")
    commands = parser.add_subparsers(dest="mode", required=True)
    stage = commands.add_parser("stage", help="verify cropped heads against the original 563-person manifest")
    stage.add_argument("--workers", type=int, required=True)
    stage.add_argument("--repair-from", type=int, help="completed staging job with the original FSL crops")
    test = commands.add_parser("small-test", help="run one cluster task from the 3+12 public controls")
    test.add_argument("--task", type=int, required=True)
    test.add_argument("--threads", type=int, required=True)
    full = commands.add_parser("full", help="run one of the 563 public IXI reference people")
    full.add_argument("--task", type=int, required=True)
    full.add_argument("--threads", type=int, required=True)
    compare = commands.add_parser('compare', help='compare public IXI maps from independently prepared runs')
    compare.add_argument('--run-dir', type=Path, required=True, help='stage output with base and sr subdirectories')
    compare.add_argument('--comparison-dir', type=Path, required=True, help='independent SynthSR control maps by IXI ID')
    audit = commands.add_parser('audit-crops', help='compare the stored voxels of recovered public crops')
    audit.add_argument('--rebuilt-dir', type=Path, required=True)
    matrix = commands.add_parser("matrix", help="verify and assemble the public grey-matter maps")
    matrix.add_argument('--array-job', type=int, help='production array ID')
    matrix.add_argument('--pilot-job', type=int, help='validate the 12 small-test maps instead of the full array')
    matrix.add_argument("--workers", type=int, default=4)
    matrix.add_argument("--recovery-array", type=int)
    matrix.add_argument("--recovery-sha256")
    return parser.parse_args()


def main() -> None:
    """Dispatch only the requested stage and preserve failures."""
    args = parse_args()
    os.environ.setdefault('VOXELWISE_PROFILE_ROOT', str(RESULTS / 'profile'))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with (
        run_resources(enabled=args.resources),
        profiled(None if args.no_profile else RESULTS / "profile" / "run.html"),
    ):
        if args.mode == "stage":
            from src.modules.staging import stage

            stage(args.workers, args.repair_from)
        elif args.mode == "small-test":
            from src.modules.reference import small_test

            small_test(args.task, args.threads)
        elif args.mode == "full":
            from src.modules.reference import full_reference

            full_reference(args.task, args.threads)
        elif args.mode == "audit-crops":
            from src.modules.audit_crops import compare_crops

            compare_crops(args.rebuilt_dir)
        elif args.mode == "matrix":
            from src.modules.matrix import build_matrix

            build_matrix(args.array_job, args.workers, args.recovery_array, args.recovery_sha256, args.pilot_job)
        else:
            from src.modules.compare import compare

            compare(args.run_dir, args.comparison_dir)


if __name__ == "__main__":
    main()
