"""Export the fitted constant-noise model and check native PCNtoolkit predictions on IXI."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled, run_resources


def main() -> None:
    """Run the IXI round-trip check, profiling the actual fit and prediction by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, help="directory of frozen artifacts; defaults to this project")
    parser.add_argument("--no-profile", action="store_true", help="skip the default pyinstrument html report")
    parser.add_argument("--resources", action="store_true", help="log wall time, cpu and peak memory")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    with (run_resources(enabled=args.resources),
          profiled(None if args.no_profile else RESULTS / "profile" / "run.html")):
        from src.modules.round_trip import run

        run(RESULTS, args.artifacts_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
