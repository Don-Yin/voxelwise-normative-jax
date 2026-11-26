"""Compare one and four A100s on the same 64 IXI bootstrap refits in one Hugging Face job."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled


def parse_args() -> argparse.Namespace:
    """Parse the receipt name and the optional synthetic CPU check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", help="unique directory name for this run's results")
    parser.add_argument("--synthetic", action="store_true", help="tiny check using four logical CPU devices; no IXI")
    parser.add_argument("--no-profile", action="store_true", help="skip the host profile, not the JAX chunk trace")
    args = parser.parse_args()
    if args.run_name is None and args.synthetic:
        args.run_name = f"synthetic-{time.time_ns()}"
    if args.run_name is None or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_name):
        parser.error(f"run name must have 1–64 lowercase letters, digits or hyphens: {args.run_name!r}")
    prefix = "synthetic-" if args.synthetic else "hugging-face-a100-"
    if not args.run_name.startswith(prefix):
        parser.error(f"run name must start with {prefix}: {args.run_name!r}")
    return args


def main() -> None:
    """Load repository code and profile the actual comparison by default."""
    args = parse_args()
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from src.modules.compare import run

    with profiled(None if args.no_profile else RESULTS / "profile" / "run.html"):
        run(RESULTS, args.run_name, smoke=args.synthetic)


if __name__ == "__main__":
    main()
