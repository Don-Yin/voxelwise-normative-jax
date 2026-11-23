"""Compare two runs' refit-flags.npz and z-summary.npz array by array, bit for bit.

    PYTHONPATH="$PWD" micromamba run -n voxelwise-normative-jax python \
        experiments/exp_voxelwisenormative_011_fragile_clusters/jobs/compare_arrays.py \
        [--baseline hugging-face-a100-z-summary-fold0] [--new hugging-face-a100-cutoff-flags-fold0] [--files refit-flags.npz z-summary.npz]

A run is a folder, or a name looked up as .results/arrays/<name>, .results/fetch-<name> or .results/<name>; the first folder
that holds every file wins. The defaults are the check of the --refit-flag-cutoffs change: the 23:14 z-summary fold 0 against
the cutoff-flags fold 0. For each array it prints "identical" or how many elements differ, where the first is, and for packed
flags the bits, for z arrays the largest difference and how many entries have a flag at 1.96 that moves. Exit status 0 when
every array is identical, 1 when one differs. Reads only.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def folder_of(run: str, names: list[str], results: Path) -> Path:
    """The folder named by `run` (a path, or a name under `results`) that holds every file in `names`."""
    candidates = [Path(run), results / f"fetch-{run}", results / "arrays" / run, results / run]
    for folder in candidates:
        if folder.is_dir() and all((folder / name).is_file() for name in names):
            return folder
    raise SystemExit(f"no folder with {', '.join(names)} for {run!r}; looked in " + ", ".join(str(c) for c in candidates))


def main(argv: list[str] | None = None) -> int:
    """Profile the comparison by default; return 0 when every array is identical, else 1."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="hugging-face-a100-z-summary-fold0", help="the earlier run (name or folder)")
    parser.add_argument("--new", default="hugging-face-a100-cutoff-flags-fold0", help="the run to check (name or folder)")
    parser.add_argument("--files", nargs="+", default=["refit-flags.npz", "z-summary.npz"], help="npz files to compare")
    parser.add_argument("--no-profile", action="store_true", help="skip the local HTML profile")
    args = parser.parse_args(argv)
    sys.path[:0] = [str(EXPERIMENT_DIR), str(EXPERIMENT_DIR.parents[1])]
    from src.utils.paths import RESULTS
    from src.utils.profiling import profiled

    with profiled(None if args.no_profile else RESULTS / "profile" / "compare-arrays.html"):
        from src.modules import array_comparison as ac

        baseline, new = folder_of(args.baseline, args.files, RESULTS), folder_of(args.new, args.files, RESULTS)
        rows = ac.compare_runs(baseline, new, args.files)
        for line in ac.render(rows, str(baseline), str(new)):
            print(line)
        return 0 if ac.identical(rows) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
