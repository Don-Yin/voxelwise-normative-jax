"""Comparable elapsed-time and device-memory panels for the IXI benchmarks."""

from hashlib import sha256
from pathlib import Path

from style import BLUE, RED, TEAL, JAX_BASELINE, PACKAGE, canvas, categories, key, panel, provenance, rows
from zoom import attach_zoom

PCNTOOLKIT_ICON = Path(__file__).resolve().parents[2] / "assets" / "pcntoolkit-icon.png"
PROJECTION_ICON_PT = 11


def _projection_icon(image, xy, *, xycoords="data"):
    """Use a dashed red frame to distinguish a projection from a measurement."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    return AnnotationBbox(
        OffsetImage(image, zoom=PROJECTION_ICON_PT / image.shape[0], interpolation="antialiased"),
        xy, xycoords=xycoords, zorder=5, fontsize=PROJECTION_ICON_PT, pad=0.11,
        bboxprops={"facecolor": "none", "edgecolor": RED, "linewidth": 0.9, "linestyle": (0, (2.8, 1.8))},
    )


class _ProjectionIconHandler:
    """Draw the same framed icon in the legend's handle coordinates."""

    def legend_artist(self, _legend, orig_handle, _fontsize, handlebox):
        xy = (handlebox.width / 2 - handlebox.xdescent, handlebox.height / 2 - handlebox.ydescent)
        mark = _projection_icon(orig_handle.offsetbox.get_data(), xy, xycoords=handlebox.get_transform())
        handlebox.add_artist(mark)
        return mark


def variants(width):
    """Keep all nine solver variants, separating errors inside and above the gate."""
    table = "speed-and-error.csv"
    data = rows(table)
    bound = 0.01
    glyphs = (JAX_BASELINE, PACKAGE, {"marker": "x", "color": TEAL, "ms": 5.5})
    with canvas(1, 3, width=width, height=6.2, sharey=True) as (fig, (time, within, above)):
        positions = list(reversed(range(len(data))))
        handles = {}
        for pos, row in zip(positions, data, strict=True):
            error = float(row["max_abs_z_error"])
            passes = error <= bound
            kind = 0 if row["method"].startswith("Original") else 1 if passes else 2
            handles[kind], = time.plot(float(row["seconds"]), pos, ls="none", **glyphs[kind])
            target, value = (within, error * 1000) if passes else (above, error)
            target.plot(value, pos, ls="none", **glyphs[kind])
        categories(time, positions, [r["method"].lower() for r in data], pad=105)
        time.set_ylim(-0.65, len(positions) - 0.35)
        for ax, title, label, limits, ticks in [
            (time, "fit and score", "seconds", (0, 8.1), [0, 2, 4, 6, 8]),
            (within, "within limit", "|Δz| (×10⁻³)", (0, 10.35), [0, 5, 10]),
            (above, "above limit", "|Δz|", (0.01, 2.25), [1, 2]),
        ]:
            ax.set(xlim=limits, xticks=ticks, xlabel=label)
            panel(ax, title, grid="x")
        for ax in (within, above):
            ax.tick_params(axis="y", left=False)
        key(fig, [handles[i] for i in range(3)], ["original JAX", "other passing fits", "failed fits"], x=0.62)
        fig.subplots_adjust(left=0.25, right=0.98, top=0.764, bottom=0.191, wspace=0.20)
    return fig, provenance("whole-mask timing and pinned-voxel error by JAX variant", table,
                           mask_voxels=165402, pinned_voxels=1000, pinned_error_limit=bound, warm_repeats=3)


def voxel_scaling(width):
    """Compare time and live device allocations as the fitted mask width grows."""
    table = "speed-memory.csv"
    data = rows(table)
    with canvas(1, 2, width=width, height=6.3, sharex=True) as (fig, (time, memory)):
        handles = []
        for method, glyph, line in [("baseline", JAX_BASELINE, "--"), ("early-f32", PACKAGE, "-")]:
            records = [r for r in data if r["method"] == method]
            x = [int(r["voxels"]) / 1000 for r in records]
            style = {**glyph, "ms": 4, "ls": line, "lw": 1.3}
            handle, = time.plot(x, [float(r["median_seconds"]) for r in records], **style)
            memory.plot(x, [int(r["peak_live_device_bytes"]) / 1e9 for r in records], **style)
            handles.append(handle)
        for ax in (time, memory):
            ax.set(xlim=(-6, 176), xticks=[0, 50, 100, 150])
        time.set(ylim=(-0.43, 7.8), yticks=[0, 2, 4, 6], ylabel="seconds")
        memory.set(ylim=(-0.04, 2.25), yticks=[0, 0.5, 1, 1.5, 2], ylabel="GB")
        panel(time, "fit and score")
        panel(memory, "peak live device memory")
        key(fig, handles, ["original 64-bit", "32-bit early stop"])
        fig.supxlabel("voxel count (thousands)", y=0.032)
        fig.subplots_adjust(left=0.11, right=0.98, top=0.75, bottom=0.21, wspace=0.34)
    return fig, provenance("single-A100 time and live device memory across voxel counts", table,
                           warm_repeats=3, memory="live allocator bytes, not reservation or CPU process memory")


def reference_scaling(width):
    """Compare two designs without joining them, and show IXI flag recurrence separately."""
    from matplotlib.image import imread

    icon = imread(PCNTOOLKIT_ICON)
    timing, flags = "speed.csv", "flag-stability.csv"
    inputs = {(r["dataset"], r["method"]): r for r in rows(timing)}
    cases = ("ixi", "synthetic")
    projection = [float(r["measured_seconds"]) * int(r["mask_voxels"]) / int(r["measured_voxels"])
                  for r in (inputs[(case, "pcntoolkit")] for case in cases)]
    measured = [float(inputs[(case, "jax_float64")]["measured_seconds"]) for case in cases]
    recurrence = next(r for r in rows(flags) if r["experiment"] == "010")
    failed = 100 * int(recurrence["count"]) / int(recurrence["original_flags"])
    kept = 100 - failed
    with canvas(width=width, height=10.4) as (fig, backdrop):
        backdrop.remove()
        # Absolute positions keep the zoom's gutter independent of the recurrence row.
        whole = fig.add_axes((0.125, 0.501, 0.375, 0.360))
        near = fig.add_axes((0.612, 0.501, 0.265, 0.360))
        bars = fig.add_axes((0.125, 0.225, 0.752, 0.16))
        # Dodge the two marks within each workload; leave the projection outside the zoom frame.
        offset = 0.19
        cpu = [whole.add_artist(_projection_icon(icon, (i - offset, s / 3600))) for i, s in enumerate(projection)]
        measured_x = [i + offset for i in range(len(cases))]
        gpu, = whole.plot(measured_x, [s / 3600 for s in measured], ls="none", zorder=3, **PACKAGE)
        near.plot(measured_x, measured, ls="none", **PACKAGE)
        for ax in (whole, near):
            ax.set_xticks([0, 1], ["IXI 450", "synthetic 50k"])
        whole.set(xlim=(-0.57, 1.57), ylim=(-37, 407), yticks=[0, 100, 200, 300, 400], ylabel="hours")
        near.set(xlim=(-0.07, 1.46), ylim=(-29, 370), yticks=[0, 100, 200, 300], ylabel="seconds")
        near.yaxis.tick_right()
        near.yaxis.set_label_position("right")
        panel(whole, "fit and score the full mask")
        panel(near, "zoom on A100")
        bars.barh([0], [kept], height=0.59, color=BLUE)
        bars.barh([0], [failed], height=0.59, left=kept, color=RED, hatch="////", edgecolor="white", lw=0.4)
        bars.set(xlim=(0, 100), ylim=(-0.49, 0.50), yticks=[], xticks=[0, 25, 50, 75, 100], xlabel="original flags (%)")
        panel(bars, "flag recurrence at 450 IXI references", grid="x")
        bars.spines["left"].set_visible(False)
        fig.legend([cpu[0], gpu], ["PCNtoolkit: one-core projection", "this package: A100 measurement"],
                   handler_map={cpu[0]: _ProjectionIconHandler()}, loc="upper center", bbox_to_anchor=(0.51, 0.998),
                   ncol=2, fontsize=9, handlelength=1.5, handletextpad=0.55,
                   columnspacing=1.6, borderpad=0.5, frameon=False)
        key(fig, bars.containers, [f"{kept:.1f}% in ≥52 of 64 refits", f"{failed:.1f}% in <52 of 64 refits"], y=0.081)
        zoom = attach_zoom(whole, near, whole_units_per_detail_unit=1 / 3600)
    return fig, provenance("measured and projected whole-mask timings and pooled refit recurrence", timing, flags,
                           cpu_projection_seconds=projection, device_median_seconds=measured, zoom=zoom,
                           recurrence_run="experiment 010", original_flags=int(recurrence["original_flags"]),
                           asset_sha256={"docs/assets/pcntoolkit-icon.png": sha256(PCNTOOLKIT_ICON.read_bytes()).hexdigest()})
