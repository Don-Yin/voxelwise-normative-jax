"""Regenerate one experiment result from the shipped aggregate tables."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from agreement import batched_fit, convergence, full_mask, synthsr_quality
from recurrence import fragile_clusters, refit_batching
from runtime import reference_scaling, variants, voxel_scaling
from style import settings

EXPERIMENTS = {
    "002": (convergence, 16.0),
    "003": (batched_fit, 16.0),
    "004": (variants, 16.0),
    "006": (voxel_scaling, 16.0),
    "007": (full_mask, 8.5),
    "010": (reference_scaling, 16.0),
    "011": (fragile_clusters, 16.0),
    "013": (refit_batching, 16.0),
    "017": (synthsr_quality, 8.5),
}


def build_figure(*, target_width_cm=None, experiment=None):
    """Build at the figure's intended width; the checker selects via environment."""
    experiment = experiment or os.environ.get("VOXELWISE_FIGURE_EXPERIMENT")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"choose one experiment from {', '.join(EXPERIMENTS)}")
    build, width = EXPERIMENTS[experiment]
    if target_width_cm is not None and target_width_cm != width:
        raise ValueError(f"experiment {experiment} is laid out for {width:g} cm")
    figure, metadata = build(width)
    return figure, {"experiment": experiment, **metadata}


def main():
    """Write only the release PNGs; leave caches in a process-local directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[2] / "assets")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="voxelwise-plots-") as runtime:
        os.environ.update(MPLBACKEND="Agg", MPLCONFIGDIR=f"{runtime}/mpl", XDG_CACHE_HOME=f"{runtime}/xdg")
        import matplotlib.pyplot as plt

        args.outdir.mkdir(parents=True, exist_ok=True)
        dpi = settings("plot-style.json")["savefig.dpi"]
        for experiment in (EXPERIMENTS if args.experiment == "all" else [args.experiment]):
            fig, _ = build_figure(experiment=experiment)
            output = args.outdir / f"exp_voxelwisenormative_{experiment}-result.png"
            fig.savefig(output, dpi=dpi, metadata={"Software": None})
            plt.close(fig)


if __name__ == "__main__":
    main()
