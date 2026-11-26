"""Pooled flag stability and the cost of batching resamples."""

from style import BLUE, JAX_BASELINE, PACKAGE, canvas, key, panel, provenance, read_json, rows
from zoom import attach_zoom


def fragile_clusters(width):
    """Show the share of clusters without an all-refit core by size bin."""
    table = "fragile-share-no-core.csv"
    data = rows(table)
    shares = [100 * int(r["clusters_without_core"]) / int(r["clusters"]) for r in data]
    with canvas(width=width, height=5.8) as (fig, ax):
        x = list(range(len(data)))
        ax.bar(x, shares, color=BLUE, width=0.65)
        ax.set_xticks(x, [r["size_to"] for r in data])
        ax.set(xlim=(-0.6, len(data) - 0.4), ylim=(0, 105), yticks=[0, 25, 50, 75, 100],
               xlabel="cluster size bin (upper bound in voxels)", ylabel="clusters without a core (%)")
        panel(ax, "clusters without a voxel flagged in all 64 refits")
        fig.subplots_adjust(left=0.145, right=0.975, top=0.845, bottom=0.23)
    return fig, provenance("pooled cluster share by size bin", table,
                           refits=64, connectivity=26, minimum_clusters_per_bin=30, threshold_abs_z=1.96)


def refit_batching(width):
    """Plot speed, live allocator peak and disagreement with sequential refits."""
    table = "refit-batching.csv"
    data = rows(table)
    with canvas(3, 1, width=width, height=12.2, sharex=True) as (fig, (time, memory, flips)):
        handles = []
        for precision, glyph, line in [("float64", {**JAX_BASELINE, "ms": 6.5}, "--"),
                                       ("float32", PACKAGE, "-")]:
            records = [r for r in data if r["precision"] == precision]
            batches = [int(r["batch_size"]) for r in records]
            style = {**glyph, "ls": line, "lw": 1.25}
            item, = time.plot(batches, [float(r["seconds_per_refit"]) for r in records], **style)
            memory.plot(batches, [int(r["allocator_peak_live_bytes"]) / 1e9 for r in records], **style)
            flips.plot(batches, [int(r["flag_disagreements"]) for r in records], **style)
            handles.append(item)
        time.set(ylim=(0.75, 2.85), yticks=[1, 1.5, 2, 2.5], ylabel="seconds")
        memory.set(ylim=(-4, 70), yticks=[0, 20, 40, 60], ylabel="GB")
        flips.set(ylim=(-500, 5000), yticks=[0, 2000, 4000], ylabel="count")
        flips.set(xlim=(-2, 68), xticks=[1, 8, 16, 32, 48, 64], xlabel="refits per batch")
        for ax, title in [(time, "elapsed time per refit"), (memory, "peak live device memory"),
                          (flips, "flags changed from serial refits")]:
            panel(ax, title)
        key(fig, handles, ["64-bit", "32-bit"])
        fig.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.11, hspace=0.42)
    return fig, provenance("refit batching, live memory and disagreement with batch size one", table,
                           refits=64, references=450, held_out=113, mask_voxels=165402,
                           memory="allocator peak live bytes after each pass, cumulative")


def stability_bins(width):
    """Show how pooled recurrence differs by original |z|, with a linear zoom near zero."""
    table = "zbin-stability.json"
    data = read_json(table)
    bins = data["baseline_abs_z_bins"]
    shares = [100 * row["unstable_flags"] / row["baseline_flags"] for row in bins]
    pooled = 100 * data["unstable_flags"] / data["baseline_flags"]
    threshold = data["z_abs_threshold"]
    labels = [f"{'(' if lo == threshold else '['}{lo:g}, {hi:g})" if hi is not None else f"≥{lo:g}"
              for lo, hi in ((r["abs_z_lower"], r["abs_z_upper_exclusive"]) for r in bins)]
    with canvas(width=width, height=5.8) as (fig, backdrop):
        backdrop.remove()
        whole = fig.add_axes((0.12, 0.225, 0.43, 0.536))
        near = fig.add_axes((0.631, 0.225, 0.280, 0.536))
        whole.plot(range(len(bins)), shares, ls="none", **PACKAGE)
        near.plot(range(1, len(bins)), shares[1:], ls="none", **PACKAGE)
        line = whole.axhline(pooled, ls="--", lw=1.1, color=JAX_BASELINE["color"])
        whole.set(xlim=(-0.53, 2.59), ylim=(-3.5, 38), yticks=[0, 10, 20, 30], ylabel="unstable flags (%)")
        whole.set_xticks(range(len(bins)), labels)
        near.set(xlim=(0.61, 2.42), ylim=(-0.004, 0.034), yticks=[0, 0.01, 0.02, 0.03])
        near.set_xticks(range(1, len(bins)), labels[1:])
        near.yaxis.tick_right()
        panel(whole, "pooled by original |z|")
        panel(near, "zoom on high |z| bins")
        key(fig, [line], [f"all original flags: {pooled:.1f}%"], y=0.995)
        fig.supxlabel("original |z| bin", y=0.045)
        zoom = attach_zoom(whole, near, whole_units_per_detail_unit=1)
    return fig, provenance("share of original flags with fewer than 52 of 64 recurrences", table,
                           held_out=data["n_test"], bins=bins, pooled_percent=pooled, zoom=zoom)
