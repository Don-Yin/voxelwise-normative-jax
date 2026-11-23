"""Time and peak memory against the number of voxels: batched fits on one A100, PCNtoolkit on a CPU.

Modes:
  gpu                one variant at one N on IXI grey matter (HPC only; slurm/scaling-a100.sh runs the grid)
  pcntoolkit-memory  PCNtoolkit peak resident memory on synthetic data of the IXI shape, one worker per N
  pcntoolkit-worker  one such worker (started by pcntoolkit-memory)
  collect            write both tables and promote the voxelwise-speed-memory-scaling-v1 artifact
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.paths import ARTIFACT, INTERMEDIATES, RESULTS
from src.utils.profiling import profiled


def parse_args() -> argparse.Namespace:
    """parse the mode and its options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("gpu", "pcntoolkit-memory", "pcntoolkit-worker", "collect"))
    parser.add_argument("--variant", choices=("early-f32", "baseline"))
    parser.add_argument("--n-voxels", type=int, nargs="+")
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu", action="store_true", help="fail unless a CUDA device is available")
    parser.add_argument("--out", type=Path, help="worker JSON path")
    parser.add_argument("--save-dir", type=Path, help="worker PCNtoolkit save_dir")
    parser.add_argument("--gpu-run", type=Path, help="collect: .results/json/gpu/<run> folder")
    parser.add_argument("--pcntoolkit-run", type=Path, help="collect: .results/json/pcntoolkit-synthetic/<run> folder")
    parser.add_argument("--no-profile", action="store_true", help="skip the default pyinstrument html report")
    args = parser.parse_args()
    needs = {"gpu": ("variant", "n_voxels"), "pcntoolkit-worker": ("n_voxels", "out", "save_dir"),
             "collect": ("gpu_run", "pcntoolkit_run")}.get(args.mode, ())
    missing = [name for name in needs if getattr(args, name) is None]
    if missing:
        parser.error(f"{args.mode} needs --{', --'.join(m.replace('_', '-') for m in missing)}")
    if args.mode in ("gpu", "pcntoolkit-worker") and len(args.n_voxels) != 1:
        parser.error(f"{args.mode} takes exactly one --n-voxels value")
    return args


def main(args: argparse.Namespace) -> None:
    """dispatch one mode; every output goes under .results/ or the promoted artifact."""
    if args.mode == "gpu":
        from src.modules.gpu_scaling import run

        run(args.variant, args.n_voxels[0], args.chunk, args.repeats, args.gpu, RESULTS)
    elif args.mode == "pcntoolkit-worker":
        from src.modules.pcntoolkit_memory import fit_one

        fit_one(args.n_voxels[0], args.out, args.save_dir)
    elif args.mode == "pcntoolkit-memory":
        from src.modules.pcntoolkit_memory import GRID, launch

        launch(tuple(args.n_voxels or GRID), RESULTS, INTERMEDIATES, Path(__file__).resolve(), not args.no_profile)
    else:
        from src.modules.collect import promote

        promote(RESULTS / args.gpu_run, RESULTS / args.pcntoolkit_run, RESULTS, ARTIFACT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arguments = parse_args()
    name = arguments.mode + (f"-n{arguments.n_voxels[0]}" if arguments.mode == "pcntoolkit-worker" else "")
    with profiled(None if arguments.no_profile else RESULTS / "profile" / f"{name}.html"):
        main(arguments)
