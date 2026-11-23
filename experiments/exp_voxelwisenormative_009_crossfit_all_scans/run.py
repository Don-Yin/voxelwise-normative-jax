"""Cross-fit the 563 IXI scans at all 165,402 grey-matter voxels; one score per held-out scan."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import UTC, datetime

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled, run_resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='verify inputs, design and folds locally; do not fit')
    mode.add_argument('--vbm-age', action='store_true', help='on Slurm only: fit the whole-cohort linear age contrast')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=32768)
    parser.add_argument('--focus-column', type=int, default=148058, help='also retain model for this mask column')
    parser.add_argument('--no-profile', action='store_true', help='skip the default pyinstrument HTML')
    parser.add_argument('--resources', action='store_true', help='log wall time, CPU and peak RSS')
    args = parser.parse_args()
    args.run_name = args.run_name or datetime.now(UTC).strftime('crossfit-%Y%m%d-%H%M%S-%f')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', args.run_name):
        parser.error('run-name must have 1–64 lowercase letters, digits or hyphens')
    if args.seed < 0 or args.chunk < 1 or not 0 <= args.focus_column < 165_402:
        parser.error('seed and focus column must be in range and chunk size must be positive')
    return args


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    with (run_resources(enabled=args.resources),
          profiled(None if args.no_profile else RESULTS / 'profile' / 'run.html')):
        if args.vbm_age:
            job = os.environ.get('SLURM_JOB_ID', '')
            if sys.platform != 'linux' or not re.fullmatch(r'[0-9]+', job):
                raise SystemExit('--vbm-age may only read the grey-matter matrix in a Linux Slurm job')
            from src.modules.vbm_age import fit_vbm

            fit_vbm(RESULTS / 'vbm' / job, min(args.chunk, 8192), args.focus_column)
        else:
            from src.modules.crossfit import run

            run(RESULTS, args.run_name, args.seed, args.chunk, args.plan, args.focus_column)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
