"""Find a matched pair of voxels with different recurrence across the fixed 64 draws."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.utils.profiling import profiled, run_resources

EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / ".results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, default=768)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--resources", action="store_true")
    args = parser.parse_args()
    if args.chunk <= 0:
        parser.error("--chunk must be positive")
    root = EXPERIMENT.parents[1]
    sys.path.insert(0, str(root))
    from src.modules.scan import scan

    job = os.environ.get("SLURM_JOB_ID", "local")
    with (run_resources(enabled=args.resources),
          profiled(None if args.no_profile else RESULTS / "profile" / "run.html")):
        scan(args.baseline, args.baseline_sha256, args.bootstrap, args.receipt,
             RESULTS / "json" / f"pairs-{job}.json", args.chunk)


if __name__ == "__main__":
    main()
