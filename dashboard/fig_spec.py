"""Serialise a matplotlib figure for the dashboard's WebGL viewer.

Emits a JSON spec (layout, labels, limits, per-trace styling) plus a binary
blob holding every trace's full x/y as float32 -- and per-point RGBA where a
scatter is coloured by value -- so the browser draws the real data rather
than a thinned copy. A float64 copy is kept in an ``.npz`` so a clicked point
can report exact values. Any artist this does not understand (images,
pcolormesh, patches, non-rectilinear axes) makes the whole figure ``None`` and
the dashboard keeps the PNG: a plot is either faithful or it is the image.

Dates travel as float32 seconds relative to the figure's earliest timestamp
``t0`` (epoch ms, in the spec and blob header); the browser formats ticks from
that. Dashboard-only: nothing in pelagos_py imports this.
"""

import json

import matplotlib.dates as mdates
import numpy as np
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex

DASH = {"--": "dash", "-.": "dashdot", ":": "dot"}


def _hex(color, default="#1f77b4"):
    try:
        return to_hex(color)
    except Exception:  # noqa: BLE001 - a colour is cosmetic, never fatal
        return default


def _alpha(color, fallback):
    try:
        if not isinstance(color, str) and len(color) == 4:
            return float(color[3])
    except TypeError:
        pass
    return 1.0 if fallback is None else float(fallback)


def _is_datetime_axis(axis):
    return isinstance(axis.get_major_locator(), mdates.DateLocator) or isinstance(
        axis.get_major_formatter(), mdates.DateFormatter
    )


def _coord(raw):
    """A coordinate as float64 matplotlib day numbers for dates (and whether it was one).

    Steps hand ``plot`` either matplotlib date numbers (floats ~19000) or raw
    datetime64/datetime objects; both end up on the same scale here.
    """
    arr = np.asarray(raw)
    is_date = False
    if np.issubdtype(arr.dtype, np.datetime64):
        arr, is_date = mdates.date2num(arr), True
    elif arr.dtype == object and arr.size:
        try:
            arr, is_date = mdates.date2num(arr), True
        except (TypeError, ValueError):
            pass
    return np.asarray(arr, dtype=np.float64), is_date


def _epoch_ms(ordinal):
    return np.asarray(ordinal, dtype=np.float64) * 86400000.0


def _ref_line(line, ax):
    # axhline/axvline mix axes-space [0, 1] with one data coordinate; matplotlib
    # marks them by their blended transform. Emitted as a spanning line.
    try:
        is_h = line.get_transform() == ax.get_yaxis_transform()
        is_v = line.get_transform() == ax.get_xaxis_transform()
    except Exception:  # noqa: BLE001
        return None
    if not (is_h or is_v):
        return None
    val, is_date = _coord(line.get_ydata()[:1] if is_h else line.get_xdata()[:1])
    return {
        "axis": "y" if is_h else "x", "value": float(val[0]), "date": is_date,
        "color": _hex(line.get_color()),
        "dash": DASH.get(line.get_linestyle(), "solid"),
        "opacity": _alpha(line.get_color(), line.get_alpha()),
        "width": float(line.get_linewidth()),
    }


def _line_trace(line):
    raw_x, raw_y = line.get_data()
    x, x_date = _coord(raw_x)
    y, y_date = _coord(raw_y)
    if len(x) == 0 or not np.any(np.isfinite(y)):
        return None
    ls, marker = line.get_linestyle(), line.get_marker()
    has_line = ls not in ("None", " ", "", None)
    has_marker = marker not in ("None", " ", "", None)
    if not has_line and not has_marker:
        has_line = True
    color = line.get_color()
    return {
        "x": x, "y": y, "x_date": x_date, "y_date": y_date, "rgba": None,
        "mode": ("lines+markers" if has_line and has_marker else "lines" if has_line else "markers"),
        "label": line.get_label(),
        "color": _hex(color),
        "opacity": _alpha(color, line.get_alpha()),
        "width": float(line.get_linewidth()),
        "dash": DASH.get(ls, "solid"),
        "size": float(line.get_markersize()),
    }


def _scatter_trace(coll):
    offsets = np.asarray(coll.get_offsets())
    if offsets.size == 0:
        return None
    x, x_date = _coord(offsets[:, 0])
    y, y_date = _coord(offsets[:, 1])
    if not np.any(np.isfinite(y)):
        return None
    try:  # c=<array> colours only resolve at draw time, and this figure was never drawn
        coll.update_scalarmappable()
    except Exception:  # noqa: BLE001
        pass
    faces = coll.get_facecolors()
    rgba = None
    if len(faces) == 0:
        color, opacity = "#1f77b4", 1.0
    elif len(faces) == len(x) and len(faces) > 1:
        color, opacity = None, _alpha(faces[0], coll.get_alpha())
        rgba = np.clip(np.asarray(faces) * 255, 0, 255).astype(np.uint8)
    else:
        color, opacity = _hex(faces[0]), _alpha(faces[0], coll.get_alpha())
    sizes = coll.get_sizes()
    return {
        "x": x, "y": y, "x_date": x_date, "y_date": y_date, "rgba": rgba,
        "mode": "markers", "label": coll.get_label(), "color": color, "opacity": opacity,
        "width": 1.0, "dash": "solid",
        "size": float(np.sqrt(sizes[0])) if len(sizes) else 6.0,  # points^2 -> diameter
    }


def _unsupported(ax):
    if ax.images:
        return "image"
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            return type(coll).__name__
    if len(ax.patches) > 0:
        return "patches"
    if ax.name != "rectilinear":
        return f"projection:{ax.name}"
    return None


def _share_groups(axes, which):
    """``{axes index: group id}`` for axes sharing an x (or y) axis."""
    groups, out = [], {}
    for i, ax in enumerate(axes):
        try:
            grouper = ax.get_shared_x_axes() if which == "x" else ax.get_shared_y_axes()
            siblings = set(grouper.get_siblings(ax))
        except Exception:  # noqa: BLE001
            siblings = {ax}
        for gid, members in enumerate(groups):
            if members & siblings:
                members.add(ax)
                out[i] = gid
                break
        else:
            groups.append(set(siblings))
            out[i] = len(groups) - 1
    return out


def _grid_cell(ax):
    """Normalised ``[left, top, width, height]`` of the axes' gridspec cell, or None."""
    try:
        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        nrows, ncols = gs.get_geometry()
        hr = gs.get_height_ratios() or [1] * nrows
        wr = gs.get_width_ratios() or [1] * ncols
        htot, wtot = float(sum(hr)), float(sum(wr))
        r0, r1 = ss.rowspan.start, ss.rowspan.stop
        c0, c1 = ss.colspan.start, ss.colspan.stop
        return [
            round(sum(wr[:c0]) / wtot, 6), round(sum(hr[:r0]) / htot, 6),
            round(sum(wr[c0:c1]) / wtot, 6), round(sum(hr[r0:r1]) / htot, 6),
        ]
    except Exception:  # noqa: BLE001
        return None


def _collect(ax):
    traces, reflines = [], []
    for line in ax.get_lines():
        ref = _ref_line(line, ax)
        if ref is not None:
            reflines.append(ref)
            continue
        t = _line_trace(line)
        if t is not None:
            traces.append(t)
    for coll in ax.collections:
        t = _scatter_trace(coll)
        if t is not None:
            traces.append(t)
    return traces, reflines


def _top_axis(ax, x_date, t0):
    """Time -> N_MEASUREMENTS table (<=1000 rows) for the top axis, or None."""
    table = getattr(ax, "_pelagos_index", None)
    if table is None:
        return None
    t, i = table
    if len(t) > 1000:
        keep = np.linspace(0, len(t) - 1, 1000).astype(int)
        t, i = t[keep], i[keep]
    t = _rel(_epoch_ms(t) if x_date else t, x_date, t0)
    return {"label": "N_MEASUREMENTS", "t": [round(float(v), 3) for v in t], "i": [int(v) for v in i]}


def _rel(values, is_date, t0):
    """Float64 -> wire units: seconds since t0 for dates, else unchanged."""
    v = np.asarray(values, dtype=np.float64)
    return (v - t0) / 1000.0 if is_date else v


def _pack(header, arrays):
    hb = json.dumps(header).encode("utf-8")
    hb += b" " * (-len(hb) % 4)  # keep the float32 arrays 4-byte aligned for zero-copy views
    return b"".join([len(hb).to_bytes(4, "little"), hb, *(a.tobytes() for a in arrays)])


def serialise(fig):
    """``(spec, reason, blob, full)``: the JSON spec, why not (if None), the
    float32 binary the browser draws, and ``{"<panel>_<trace>_x": float64,...}``
    for exact click-lookups. Dates are epoch ms in ``full``."""
    # Secondary axes (the top N_MEASUREMENTS axis fig_spec.date_axis adds) hold
    # no data; their mapping is shipped as the parent panel's top_axis instead.
    axes = [ax for ax in fig.axes if ax.get_visible() and not isinstance(ax, SecondaryAxis)]
    if not axes:
        return None, "no axes", None, None
    for ax in axes:
        blocker = _unsupported(ax)
        if blocker is not None:
            return None, blocker, None, None

    collected = []
    for ax in axes:
        traces, reflines = _collect(ax)
        if not traces:
            return None, "empty axes", None, None
        x_date = _is_datetime_axis(ax.xaxis) or any(t["x_date"] for t in traces)
        y_date = _is_datetime_axis(ax.yaxis) or any(t["y_date"] for t in traces)
        for t in traces:  # matplotlib day numbers -> epoch ms on any date axis
            if x_date:
                t["x"] = _epoch_ms(t["x"])
            if y_date:
                t["y"] = _epoch_ms(t["y"])
        collected.append((ax, traces, reflines, x_date, y_date))

    # Dates are shipped relative to the figure's earliest timestamp so float32 keeps
    # sub-second resolution across a deployment; find it before converting anything.
    t0 = None
    for _ax, traces, _r, x_date, y_date in collected:
        for t in traces:
            for arr, is_date in ((t["x"], x_date), (t["y"], y_date)):
                if is_date and np.any(np.isfinite(arr)):
                    lo = float(np.nanmin(arr))
                    t0 = lo if t0 is None else min(t0, lo)
    t0 = 0.0 if t0 is None else t0

    share_x, share_y = _share_groups(axes, "x"), _share_groups(axes, "y")
    panels, header_traces, arrays, full = [], [], [], {}
    for i, (ax, traces, reflines, x_date, y_date) in enumerate(collected):
        specs = []
        for j, t in enumerate(traces):
            n = len(t["x"])
            full[f"{i}_{j}_x"], full[f"{i}_{j}_y"] = t["x"], t["y"]
            arrays.append(_rel(t["x"], x_date, t0).astype("<f4"))
            arrays.append(_rel(t["y"], y_date, t0).astype("<f4"))
            if t["rgba"] is not None:
                arrays.append(np.ascontiguousarray(t["rgba"]))
            header_traces.append({"panel": i, "trace": j, "n": n, "rgba": t["rgba"] is not None})
            specs.append({k: t[k] for k in ("mode", "label", "color", "opacity", "width", "dash", "size")} | {"n": n})
        for r in reflines:
            is_date = x_date if r["axis"] == "x" else y_date
            r["value"] = float(_rel(_epoch_ms(r["value"]) if is_date else r["value"], is_date, t0))
            del r["date"]
        legend = ax.get_legend()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        panels.append({
            "title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
            "xdate": x_date, "ydate": y_date,
            "xlim": [float(v) for v in _rel(_epoch_ms(xlim) if x_date else xlim, x_date, t0)],
            "ylim": [float(v) for v in _rel(_epoch_ms(ylim) if y_date else ylim, y_date, t0)],
            "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
            "legend": legend is not None,
            "legend_title": legend.get_title().get_text() if legend is not None else "",
            "cell": _grid_cell(ax), "share_x": share_x[i], "share_y": share_y[i],
            "traces": specs, "reflines": reflines,
            "top_axis": _top_axis(ax, x_date, t0),
        })

    width, height = fig.get_size_inches()
    spec = {
        "suptitle": fig._suptitle.get_text() if fig._suptitle is not None else "",
        "aspect": float(height) / float(width) if width else 0.75,
        "t0": t0,
        "points": int(sum(t["n"] for t in header_traces)),
        "panels": panels,
    }
    blob = _pack({"t0": t0, "traces": header_traces}, arrays)
    return spec, "", blob, full
