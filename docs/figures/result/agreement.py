"""Panels for optimiser convergence, paired scores and map-level quality."""

from style import BLUE, JAX_BASELINE, PACKAGE, canvas, key, panel, provenance, rows


def convergence(width):
    """Compare default and converged PCNtoolkit on the same sampled voxels."""
    table = "convergence.csv"
    bins = rows(table)
    specs = [("objective_gap", "objective gap", "default − converged (nats)", [0, 25, 50, 75, 100]),
             ("max_abs_z_shift", "shift in held-out z-scores", "largest |Δz| per voxel", [0, 1, 2, 3, 4, 5])]
    with canvas(1, 2, width=width, height=5.9, sharey=True) as (fig, axes):
        for ax, (quantity, title, label, ticks) in zip(axes, specs, strict=True):
            selected = [r for r in bins if r["quantity"] == quantity]
            lower = [float(r["lower"]) for r in selected]
            widths = [float(r["upper"]) - x for r, x in zip(selected, lower, strict=True)]
            ax.bar(lower, [int(r["voxels"]) for r in selected], width=widths,
                   align="edge", color=BLUE, edgecolor="white", lw=0.4)
            ax.set(xlim=(0, float(selected[-1]["upper"])), ylim=(0, 350), xticks=ticks, xlabel=label)
            panel(ax, title)
        axes[0].set_ylabel("voxels")
        fig.subplots_adjust(left=0.13, right=0.965, top=0.84, bottom=0.23, wspace=0.17)
    return fig, provenance("binned paired fit comparison", table, n_voxels=1000, n_held_out=113)


def batched_fit(width):
    """Separate whole-mask JAX timing from the smaller PCNtoolkit score comparison."""
    table = "batched-fit.csv"
    results = {r["category"]: float(r["value"]) for r in rows(table)}
    with canvas(1, 2, width=width, height=6.8) as (fig, (time, error)):
        time.plot(0, results["cpu"], ls="none", **JAX_BASELINE)
        time.plot(1, results["a100"], ls="none", **PACKAGE)
        time.set(xlim=(-0.45, 1.45), ylim=(-44, 650), yticks=[0, 200, 400, 600], ylabel="seconds")
        time.set_xticks([0, 1], ["JAX on CPU", "JAX on A100"])
        panel(time, "whole-mask fit and score")
        for i, statistic in enumerate(("median", "p99", "max")):
            error.plot(i, 1000 * results[statistic], ls="none", **PACKAGE)
        error.set(xlim=(-0.48, 2.48), ylim=(-0.1, 1.28), yticks=[0, 0.5, 1], ylabel="|Δz| (×10⁻³)")
        error.set_xticks([0, 1, 2], ["median", "99th pct.", "max"])
        panel(error, "sampled z gap vs PCNtoolkit")
        fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.16, wspace=0.40)
    return fig, provenance("same JAX fit on two platforms and sampled agreement with PCNtoolkit", table,
                           timing="best of three warm full-mask passes", paired_voxels=1000, held_out=113)


def full_mask(width):
    """Show two absolute-error statistics on separate, explicitly labelled linear scales."""
    table = "full-mask-agreement.csv"
    values = {r["statistic"]: float(r["absolute_z_difference"]) for r in rows(table)}
    specs = [("p999", "99.9th percentile", "|Δz| (×10⁻³)", 1000, [0, 0.5, 1]),
             ("maximum", "maximum", "|Δz|", 1, [0, 2, 4])]
    with canvas(2, 1, width=width, height=8.2) as (fig, axes):
        for ax, (statistic, title, label, factor, ticks) in zip(axes, specs, strict=True):
            ax.plot(factor * values[statistic], 0, ls="none", **PACKAGE)
            ax.set(xlim=(0, 1.2 * ticks[-1]), ylim=(-0.55, 0.55), xlabel=label, xticks=ticks, yticks=[])
            panel(ax, title, grid="x")
            ax.spines["left"].set_visible(False)
        key(fig, [axes[0].lines[0]], ["difference from PCNtoolkit"], y=0.995)
        fig.subplots_adjust(left=0.09, right=0.95, top=0.805, bottom=0.15, hspace=0.72)
    return fig, provenance("summary of paired whole-mask z-score differences", table, score_pairs=18_690_426)


def synthsr_quality(width):
    """Plot the group medians, not individual maps or extreme observations."""
    table = "synthsr-map-quality.csv"
    data = rows(table)
    with canvas(width=width, height=4.7) as (fig, ax):
        ax.plot([0, 1], [float(r["median_loo_pearson_r"]) for r in data], ls="none", **PACKAGE)
        ax.set(xlim=(-0.54, 1.54), ylim=(0.898, 0.904), ylabel="Pearson r")
        ax.set_yticks([0.898, 0.900, 0.902, 0.904])
        ax.set_xticks([0, 1], ["retained crops", "rebuilt crops"])
        panel(ax, "median map correlation")
        fig.subplots_adjust(left=0.29, right=0.97, top=0.785, bottom=0.192)
    return fig, provenance("median correlation with the leave-one-out mean grey-matter map", table)
