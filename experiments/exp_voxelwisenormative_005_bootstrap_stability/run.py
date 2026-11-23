"""Refit all 165,402 voxels for one held-out IXI subject on a GPU."""

from __future__ import annotations

import argparse
import logging
import re
from contextlib import nullcontext
from datetime import UTC, datetime

from src.modules.experiment import run
from src.modules.live import live_view, no_view
from src.modules.verify import verify
from src.utils.paths import RESULTS
from src.utils.profiling import profiled, run_resources


def parse_args() -> argparse.Namespace:
    """Parse and validate run identity, GPU fitting and live display options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8, help="number of bootstrap reference samples")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--run-name", default=None, help="unique identifier under .results; defaults to UTC time")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="open the loopback viewer and pause after baseline")
    mode.add_argument("--verify", action="store_true", help="audit a saved --run-name without refitting")
    parser.add_argument("--port", type=int, default=8765, help="loopback viewer port (only with --live)")
    parser.add_argument("--no-profile", action="store_true", help="skip the default pyinstrument HTML report")
    parser.add_argument("--resources", action="store_true", help="log wall, CPU and peak RSS")
    args = parser.parse_args()
    if args.verify and not args.run_name:
        parser.error("--verify requires the saved --run-name")
    args.run_name = args.run_name or datetime.now(UTC).strftime("bootstrap-%Y%m%d-%H%M%S-%f")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_name):
        parser.error("--run-name must have 1–64 lowercase letters, digits or hyphens")
    if not (1 <= args.trials <= 512 and args.chunk > 0 and args.seed >= 0 and 0 <= args.port <= 65535):
        parser.error("expected 1–512 trials, a positive chunk, a nonnegative seed and a valid port")
    return args


def main(args: argparse.Namespace) -> None:
    """Run the measured job or the same GPU workload with real-time updates."""
    viewer = live_view(args.port, args.trials) if args.live else nullcontext(no_view)
    with viewer as publish:
        with (
            run_resources(enabled=args.resources),
            profiled(None if args.no_profile else RESULTS / "profile" / "run.html"),
        ):
            if args.verify:
                verify(RESULTS, args.run_name)
            else:
                run(RESULTS, args.run_name, args.trials, args.seed, args.chunk, publish)
        if args.live:
            input("Result saved. Press Enter to close the live viewer. ")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main(parse_args())
