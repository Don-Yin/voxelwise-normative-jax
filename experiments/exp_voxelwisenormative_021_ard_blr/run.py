"""Audit ARD optimization and time serial PCNtoolkit versus JAX on Mac CPU."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.utils.paths import EXPERIMENT_DIR, RESULTS
from src.utils.profiling import profiled

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """parse command line arguments for this experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="root of the verified IXI artifacts")
    parser.add_argument("--voxels", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3, help="number of warm JAX timing repetitions")
    parser.add_argument("--phase", choices=("both", "pcn", "jax"), default="both",
                        help="measure PCNtoolkit only, replay it for the JAX audit, or run both")
    parser.add_argument("--no-profile", action="store_true", help="skip the default pyinstrument html report")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Load the verified reference inputs and run the selected CPU benchmark phase."""
    sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("MPLCONFIGDIR", str(EXPERIMENT_DIR / ".intermediates" / "matplotlib"))
    os.environ.setdefault("PYTENSOR_FLAGS", f"base_compiledir={EXPERIMENT_DIR / '.intermediates' / 'pytensor'}")
    from src.modules.reference import reference

    from shared.voxelwisenormative.pcn_cpu_reference import load_ixi

    data = load_ixi(args.voxels, root=args.artifacts_dir)
    ref = reference(data, RESULTS / "tables" / f"pcntoolkit-{args.voxels}.npz", replay=args.phase == "jax")
    logger.info("serial pcntoolkit fit and score: %d voxels in %.3f s", data.voxels.size, ref.seconds)
    if args.phase != "pcn":
        from src.modules.benchmark import benchmark

        benchmark(data, ref, RESULTS, args.repeats, profiled=not args.no_profile)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    with profiled(None if args.no_profile else RESULTS / "profile" / "run.html"):
        main(args)
