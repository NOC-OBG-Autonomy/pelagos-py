# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single source of truth for the look of every ``plot_diagnostics``: a 16:9
figure, one font/margin scale, one flag/category palette, one date format. Keeps
PNGs uniform and within what the dashboard's interactive (WebGL) renderer can
reproduce.

WebGL-safe rules:
  * Draw data with ``ax.plot(..., ls="", marker="o")`` (preferred, faster) or
    ``ax.scatter``; both round-trip. ``axhline``/``axvline`` are fine.
  * Colour by DISCRETE category (one call per category). Continuous colour +
    ``fig.colorbar`` adds a second axes the serialiser can't read and drops the
    WHOLE figure to PNG-only (as with MLD/salinity/bbp).
  * PNG-only (not reproducible), but still style them: hist, bar, fill_between,
    pcolormesh, contourf, imshow, fig.colorbar, twinx, cartopy/polar, LineCollection.
  * Multi-panel: build with plt.subplots; the interactive view stacks axes
    vertically in creation order with a shared x-axis, so use sharex=True.
"""

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from functools import partial

# Geometry: single panel is 16:9; multi-panel keeps width, grows height by ROW_H/row.
FIG_W, FIG_H, DPI = 10, 5.625, 130
ROW_H = 3.0  # inches per panel for multi-panel (nrows > 1) figures

FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 12, 11, 9, 8, 8

MARKER = 4            # markersize for plot() point series
ALPHA = 0.7
RASTER_ABOVE = 5000   # rasterize dense point layers above this many points
CAPTURE_MAX_POINTS = 100_000  # point cap for figures saved to disk (report capture)
MAX_POINTS = None  # set by report capture so dense series are thinned when drawn, not only when saved

# Colours (hex, one mapping used everywhere).
FLAG_COLOURS = {
    0: "#9aa5ad", 1: "#1f6fd6", 2: "#7fb2e5", 3: "#e8912b", 4: "#d6392f",
    5: "#9aa5ad", 6: "#9aa5ad", 7: "#9aa5ad", 8: "#17b6c4", 9: "#111111",
}
FLAGGED = "#b2bec3"
CATEGORY = ["#00b894", "#0984e3", "#d63031", "#fdcb6e", "#6c5ce7",
            "#e84393", "#00cec9", "#e17055"]

# QC flag meanings, used to label flag legend entries as "1 (good)".
FLAG_MEANINGS = {
    0: "no QC", 1: "good", 2: "prob good", 3: "prob bad", 4: "bad",
    5: "changed", 8: "interp", 9: "missing",
}


def flag_label(flag):
    """Legend label for a QC flag: '1 (good)' when the meaning is known, else '6'."""
    meaning = FLAG_MEANINGS.get(flag)
    return f"{flag} ({meaning})" if meaning else str(flag)


def thin_idx(n, cap=None):
    """Indices evenly thinning ``n`` points to ``cap`` (default MAX_POINTS); arange(n) if under the cap."""
    cap = cap or MAX_POINTS
    return np.linspace(0, n - 1, cap).astype(int) if cap and n > cap else np.arange(n)


def new_fig(nrows=1, ncols=1, sharex=False, sharey=False, height_ratios=None, width_ratios=None):
    """A standard figure + axes at the standard width/dpi. Axes always 2D: axes[r][c].

    Single panel is 16:9; multi-panel grows height by ``ROW_H`` per row.
    ``height_ratios`` / ``width_ratios`` (e.g. ``(3, 1)``) set unequal panel sizes via gridspec.
    """
    import matplotlib.pyplot as plt
    height = FIG_H if nrows == 1 else ROW_H * nrows
    gridspec_kw = {k: v for k, v in {"height_ratios": height_ratios, "width_ratios": width_ratios}.items()
                   if v is not None} or None
    return plt.subplots(nrows, ncols, figsize=(FIG_W, height), dpi=DPI,
                        sharex=sharex, sharey=sharey, squeeze=False,
                        gridspec_kw=gridspec_kw)


def style_axes(ax, *, title=None, xlabel=None, ylabel=None):
    """The common per-axes look."""
    if title:
        ax.set_title(title, fontsize=FS_TITLE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def axis_label(var, units=None):
    """'TEMP [degC]' when units are meaningful, else just the name."""
    u = (units or "").strip()
    return var if u in ("", "1", "unitless", "unknown", "none", "None") else f"{var} [{u}]"


def date_axis(ax, which="x", index=None):
    """Consistent date formatting on the chosen axis (x or y).

    ``index`` is the full TIME array whose position is N_MEASUREMENTS: with it, a
    time-series x axis also gets N_MEASUREMENTS along the top (of the top-most
    panel when the axes share x), so readers see both.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    loc = mdates.AutoDateLocator()
    axis.set_major_locator(loc)
    axis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    if index is None or which != "x":
        return
    t = np.asarray(index)
    pos = thin_idx(t.size)  # report capture: a thinned index maps just as well
    t = t[pos]
    if np.issubdtype(t.dtype, np.datetime64) or t.dtype == object:
        t = np.asarray(mdates.date2num(t), dtype=float)
    ok = np.isfinite(t)
    t, i = t[ok], pos[ok].astype(float)
    if t.size < 2:
        return
    order = np.argsort(t, kind="stable")
    ts, is_ = t[order], i[order]
    top = max(ax.get_shared_x_axes().get_siblings(ax), key=lambda a: a.get_position().y1)
    # partials (not lambdas) so the figure stays picklable for background saving
    sec = top.secondary_xaxis("top", functions=(partial(np.interp, xp=ts, fp=is_),
                                                partial(np.interp, xp=i, fp=t)))
    sec.set_xlabel("N_MEASUREMENTS", fontsize=FS_LABEL)
    sec.tick_params(labelsize=FS_TICK)
    top._pelagos_index = (ts, is_)  # picked up by the dashboard serialiser


def x_time(data):
    """What a whole-dataset series is plotted against: TIME if present, else N_MEASUREMENTS."""
    return data["TIME"].values if "TIME" in data else np.arange(data.sizes["N_MEASUREMENTS"])


def x_axis(ax, x):
    """Label the x axis for an ``x_time`` array: TIME below + N_MEASUREMENTS on top, or Index."""
    if np.issubdtype(np.asarray(x).dtype, np.datetime64):
        date_axis(ax, index=x)
        style_axes(ax, xlabel="TIME")
    else:
        style_axes(ax, xlabel="Index")


def points(ax, x, y, *, color, label=None, size=MARKER, alpha=ALPHA):
    """Standard point series: fast plot() markers, WebGL-safe, one legend entry."""
    x, y = np.asarray(x), np.asarray(y)
    if MAX_POINTS:
        idx = thin_idx(x.size)
        x, y = x[idx], y[idx]
    ax.plot(x, y, ls="", marker="o", markersize=size,
            markeredgewidth=0, color=color, alpha=alpha, label=label,
            rasterized=x.size > RASTER_ABOVE)


def flag_points(ax, x, y, flags):
    """y vs x coloured by QC flag: one series per present flag (0..9)."""
    flags, x, y = np.asarray(flags), np.asarray(x), np.asarray(y)
    # A flag with nothing drawable (e.g. 9 on NaN samples) gets no legend entry.
    drawable = np.isfinite(y) if np.issubdtype(y.dtype, np.number) else ~pd.isnull(y)
    for f in range(10):
        idx = np.flatnonzero((flags == f) & drawable)
        if MAX_POINTS:
            idx = idx[thin_idx(idx.size)]  # thin before gathering x/y
        if idx.size:
            points(ax, x[idx], y[idx], color=FLAG_COLOURS[f], label=flag_label(f))


def legend(ax, *, title=None):
    """Compact legend just outside the axes on the right."""
    ax.legend(title=title, fontsize=FS_LEGEND, title_fontsize=FS_LEGEND,
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.9, markerscale=2)


def finish(fig, suptitle=None):
    """Suptitle + tight layout, the same for every figure."""
    if suptitle:
        fig.suptitle(suptitle, fontsize=FS_SUPTITLE, fontweight="bold")
    fig.tight_layout()


def land_polygons(resolution="110m"):
    """Natural Earth land as one shapely geometry (cartopy caches the download)."""
    import cartopy.io.shapereader as sr
    import shapely

    path = sr.natural_earth(resolution=resolution, category="physical", name="land")
    return shapely.unary_union(list(sr.Reader(path).geometries()))


def map_extent(lon, lat, pad=0.3, min_span=0.05):
    """[lon0, lon1, lat0, lat1] around the finite positions; longitude widened by 1/cos(lat) so land keeps its shape."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[ok], lat[ok]
    lon_mid, lat_mid = (lon.min() + lon.max()) / 2, (lat.min() + lat.max()) / 2
    half = max(lon.max() - lon.min(), lat.max() - lat.min(), min_span) * (0.5 + pad)
    cos = max(np.cos(np.deg2rad(lat_mid)), 0.2)
    return [lon_mid - half / cos, lon_mid + half / cos, lat_mid - half, lat_mid + half]


def coastlines(ax, extent, *, color="0.35", linewidth=0.8, fill=None):
    """Coastlines clipped to extent as a single plot() call (WebGL-safe); fill=<colour> adds land fill (PNG-only)."""
    import shapely

    span = max(extent[1] - extent[0], extent[3] - extent[2])
    res = "10m" if span < 3 else "50m" if span < 20 else "110m"
    # Clip a little beyond the extent so the cut edges fall outside the axes.
    m = 0.05 * span
    land = shapely.clip_by_rect(land_polygons(res), extent[0] - m, extent[2] - m, extent[1] + m, extent[3] + m)
    xs, ys = [], []
    for poly in getattr(land, "geoms", [land]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        if fill:
            ax.fill(*poly.exterior.xy, color=fill, lw=0, zorder=0)
        for ring in (poly.exterior, *poly.interiors):
            x, y = ring.xy
            xs += [*x, np.nan]
            ys += [*y, np.nan]
    ax.plot(xs, ys, color=color, lw=linewidth, zorder=1)
    ax.grid(False)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / max(np.cos(np.deg2rad((extent[2] + extent[3]) / 2)), 0.2))
