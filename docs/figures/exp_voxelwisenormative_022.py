"""Plot the difference caused by export separately from the difference between fits."""

from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from pathlib import Path

from python_plotting_convention import SINGLE_COLUMN_CM, publication_subplots  # pyright: ignore[reportMissingImports]
from result.style import BLUE, RED, categories

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "exp_voxelwisenormative_022_pcntoolkit_export"
REPORT = ROOT / "experiments" / EXPERIMENT / ".results" / "json" / "round-trip.json"


def build_figure(*, target_width_cm: float = SINGLE_COLUMN_CM):
    """Build a single-column paired-score comparison at its display size."""
    if target_width_cm != SINGLE_COLUMN_CM:
        raise ValueError(f"this figure is laid out for {SINGLE_COLUMN_CM} cm, got {target_width_cm}")
    raw = REPORT.read_bytes()
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot read {REPORT}: {exc}") from exc
    statistics = ("median_abs_z", "p99_abs_z", "max_abs_z")
    comparisons = (("round_trip", "Same fit after export", BLUE, "o"),
                   ("vs_frozen", "Independent PCNtoolkit fit", RED, "s"))
    fig, ax = publication_subplots(target_width_cm=target_width_cm, height_cm=6.5)
    for series, label, color, marker in comparisons:
        values = [report[series][stat] for stat in statistics]
        if not all(isfinite(value) and value > 0 for value in values):
            raise ValueError(f"log scale requires positive finite values for {series} in {REPORT}, got {values!r}")
        ax.plot(values, range(len(statistics)), label=label, color=color, marker=marker, lw=1, ms=5)
    ax.set(xscale="log", ylim=(-0.25, 2.25), xlabel="Absolute z difference (log scale)")
    ax.set_xticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3])
    ax.grid(axis="x", color="#d9dfe4", ls=":", lw=0.75)
    ax.set_axisbelow(True)
    ax.minorticks_off()
    ax.spines[["top", "right"]].set_visible(False)
    categories(ax, range(len(statistics)), ["Median", "99th pct.", "Maximum"], pad=55)
    fig.legend(loc="upper left", bbox_to_anchor=(0.055, 0.98), borderaxespad=0)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.73, bottom=0.18)
    return fig, {"scientific_job": "score agreement after reloading and between independent fits",
                 "paired_scores": report["n_z"], "input_sha256": {str(REPORT.relative_to(ROOT)): sha256(raw).hexdigest()}}
