"""Same-scale summary of CPU fit times and paired score agreement."""
from __future__ import annotations

import numpy as np
from python_plotting_convention import publication_subplots  # pyright: ignore[reportMissingImports]

from .style import BLUE, categories, panel


def build_comparison(times, abs_z, *, target_width_cm=16):
    """Plot (PCNtoolkit, first JAX, warm JAX) times and the empirical |Δz| distribution."""
    if target_width_cm != 16:
        raise ValueError(f"paired CPU comparison is laid out for 16 cm, got {target_width_cm}")
    seconds = np.asarray(times)
    if seconds.shape != (3,) or not np.isfinite(seconds).all() or np.min(seconds) <= 0:
        raise ValueError(f"expected three positive finite times: {times!r}")
    gaps = np.sort(np.ravel(abs_z))
    if not gaps.size or not np.isfinite(gaps).all() or gaps[0] < 0:
        raise ValueError(f"expected finite nonnegative z differences: {gaps.shape=}")

    fig, (timing, scores) = publication_subplots(1, 2, target_width_cm=target_width_cm, height_cm=5.75)
    fig.subplots_adjust(left=0.183, right=0.983, top=0.863, bottom=0.245, wspace=0.41)

    timing.set_xscale("log")
    timing.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    timing.set_xlim(min(0.55, 0.75 * seconds.min()), max(1450, 1.3 * seconds.max()))
    timing.set_ylim(-0.6, 2.6)
    timing.set_xlabel("Wall time (s)")
    panel(timing, "A  CPU fit and score", grid="x")
    categories(timing, [2, 1, 0], ["PCNtoolkit", "JAX (first)", "JAX (warm)"], pad=64)
    timing.plot(seconds, [2, 1, 0], linestyle="none", marker="o", color=BLUE, markersize=5)

    lo, hi = 1e-8, max(1.5, 1.2 * gaps[-1])
    percentile = np.arange(1, gaps.size + 1) * (100 / gaps.size)
    scores.set_xscale("log")
    scores.set_xticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    scores.set_xlim(lo, hi)
    scores.set_ylim(-2, 103)
    scores.set_yticks([0, 25, 50, 75, 100])
    scores.set_xlabel(r"Absolute $z$ difference")
    scores.set_ylabel("Scores (%)")
    panel(scores, "B  Score agreement")
    scores.step(np.r_[lo, np.maximum(gaps, lo), hi], np.r_[0, percentile, 100], where="post", color=BLUE)
    return fig, {"scientific_job": "paired CPU wall times and empirical distribution of absolute z gaps",
                 "score_count": gaps.size, "curve": "empirical cumulative fraction; log x-axis"}
