"""Pooled flag stability and the cost of batching resamples."""

from style import BLUE, JAX_BASELINE, PACKAGE, canvas, key, panel, provenance, rows


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
