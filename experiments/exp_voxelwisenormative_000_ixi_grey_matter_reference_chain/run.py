"""Build an IXI-only grey-matter reference using an external preprocessing chain."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS = EXPERIMENT_DIR / ".results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(EXPERIMENT_DIR.name)


def parse_args() -> argparse.Namespace:
    """parse command line arguments for this experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "prepare", "template", "register", "publish"))
    parser.add_argument("--derivatives", type=Path, help="new local directory for verified published derivatives")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="skip the default pyinstrument html report",
    )
    parser.add_argument(
        "--resources",
        action="store_true",
        help="log wall, cpu, and peak rss for this run",
    )
    args = parser.parse_args()
    if (args.mode == 'publish') != (args.derivatives is not None):
        parser.error('publish requires --derivatives; other stages do not use it')
    return args


def main(args: argparse.Namespace) -> None:
    """run the experiment and write every output under RESULTS."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    logger.info("stage %s, writing outputs to %s", args.mode, RESULTS)
    # Load the external chain only when running a stage.
    from src.modules import reference_chain

    reference_chain.run(args)


if __name__ == "__main__":
    args = parse_args()
    try:
        from src.utils.profiling import profiled, run_resources
    except ImportError:
        from contextlib import contextmanager

        @contextmanager
        def run_resources(*, enabled: bool = True, python_peak: bool = False):
            """stdlib-only fallback when the tier-1 helper is not present."""
            del enabled, python_peak
            yield

        @contextmanager
        def profiled(out: Path | None):
            """lazy pyinstrument fallback when the tier-1 helper is not present."""
            if out is None:
                yield
                return
            from pyinstrument import Profiler

            profiler = Profiler()
            profiler.start()
            try:
                yield
            finally:
                profiler.stop()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(profiler.output_html())

    with (
        run_resources(enabled=args.resources),
        profiled(None if args.no_profile else RESULTS / "profile" / "run.html"),
    ):
        main(args)
