"""Matplotlib-only scale enlargement for adjacent, fixed-layout chart panels."""

from style import settings


def attach_zoom(whole, detail, *, whole_units_per_detail_unit):
    """Frame the detail's exact limits in the full chart and connect corresponding corners."""
    import numpy as np
    from matplotlib import patheffects
    from matplotlib.colors import to_rgb
    from matplotlib.image import BboxImage
    from matplotlib.patches import ConnectionPatch, Polygon
    from matplotlib.transforms import Bbox, TransformedBbox

    style = settings("zoom-style.json")
    colour, widths = style["colour"], style["widths"]
    fig = whole.figure

    def halo(width):
        return [patheffects.Stroke(linewidth=width + style["halo"], foreground="white"), patheffects.Normal()]

    x0, x1 = detail.get_xlim()
    y0, y1 = (value * whole_units_per_detail_unit for value in detail.get_ylim())
    indicator = whole.indicate_inset((x0, y0, x1 - x0, y1 - y0), edgecolor=colour,
                                     facecolor="none", alpha=1, linewidth=widths["box"], zorder=1.8)
    indicator.rectangle.set_path_effects(halo(widths["box"]))
    corners = ((x1, y0), (x1, y1))
    targets = ((0, 0), (0, 1))
    normalized = fig.transFigure.inverted()
    a, b = normalized.transform(whole.transData.transform(corners))
    d, c = normalized.transform(detail.transAxes.transform(targets))
    # The band is behind both panels; only the part in the gutter remains visible.
    bounds = Bbox.from_extents(a[0], min(a[1], d[1]), d[0], max(b[1], c[1]))
    band = BboxImage(TransformedBbox(bounds, fig.transFigure), zorder=-1, interpolation="bilinear")
    rgba = np.ones((2, 256, 4))
    rgba[..., :3] = to_rgb(colour)
    rgba[..., 3] = np.linspace(style["band_opacity"], style["band_fade"], rgba.shape[1])
    band.set_data(rgba)
    band.set_clip_path(Polygon([a, d, c, b], transform=fig.transFigure))
    fig.add_artist(band)
    for start, end in zip(corners, targets, strict=True):
        fig.add_artist(ConnectionPatch(start, end, "data", "axes fraction", axesA=whole, axesB=detail,
                                      color=colour, linewidth=widths["line"], capstyle="round",
                                      path_effects=halo(widths["line"]), zorder=4))
    for spine in detail.spines.values():
        spine.set(visible=True, edgecolor=colour, linewidth=widths["frame"], path_effects=halo(widths["frame"]))
    return {"x_bounds": [x0, x1], "y_bounds_in_full_panel_units": [y0, y1], "colour": colour}
