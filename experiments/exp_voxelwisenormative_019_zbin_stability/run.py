"""Measure bootstrap flag instability by original |z| for the frozen IXI split."""

from __future__ import annotations

import argparse
import logging
import re
import sys

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled


def main() -> None:
    """Parse the output name and profile the shared refit worker in process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="unique directory name for this run's results")
    parser.add_argument("--synthetic", action="store_true", help="tiny CPU smoke test; no IXI or bin estimates")
    parser.add_argument("--no-profile", action="store_true", help="skip the default pyinstrument HTML report")
    args = parser.parse_args()
    prefix = "synthetic-" if args.synthetic else "hugging-face-a100-"
    if not re.fullmatch(prefix + r"[a-z0-9][a-z0-9-]*", args.run_name) or len(args.run_name) > 64:
        parser.error(f"run name must start with {prefix} and use at most 64 lowercase letters, digits or hyphens: "
                     f"{args.run_name!r}")

    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with profiled(None if args.no_profile else RESULTS / "profile" / "run.html"):
        from src.modules.zbins import run

        run(RESULTS / "json" / args.run_name / "zbin-stability.json", smoke=args.synthetic)


if __name__ == "__main__":
    main()
