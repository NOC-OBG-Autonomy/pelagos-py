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

"""Manual QC: flag data inside (or outside) hand-drawn boxes on a two-variable plot."""

#### Mandatory imports ####
import numpy as np
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from pelagos_py.steps.quality_control.range_qc import QC_COMBINATRIX
from pelagos_py.utils import fig_spec


@register_qc
class manual_qc(BaseQC):
    """
    Flag samples by hand-drawn boxes: each box is a 2D range test on an
    ``x_variable``/``y_variable`` plot, and every sample inside (or outside) it
    gets the box's flag on the listed variables. The dashboard builds the boxes
    for you — pause on this test, cmd/ctrl-drag a box on the plot, pick a flag —
    but the config it writes is plain YAML, so the run is repeatable anywhere.

    Unlike other tests this one starts from the existing flags and *replaces*
    them: a box overrides whatever flag a sample had (bad -> good included),
    later boxes winning on overlap. Set ``override: false`` on a box to merge
    it by the Argo combinatrix instead (can raise but never lower a flag).
    Missing samples stay 9. A box whose ``x`` and ``y`` are single values is a
    point: it flags just the one sample nearest to it (cmd/ctrl-click in the
    dashboard). A box may name its own ``x_variable``/``y_variable`` (the plot
    it was drawn on; the dashboard's plot switcher does this), else it uses the
    test's. ``y_variable`` is always flagged, so its existing ``_QC`` is loaded
    and shown on the plot. Samples no box touches get flag 1 when
    ``flag_remaining_good`` is on (the default), else keep what they had.

    Target Variable: Any
    Flag Number: Any (user-defined, 0-9)
    Variables Flagged: Any (each box's ``variables``; defaults to ``y_variable``)

    EXAMPLE
    -------
    ::

        - name: "Apply QC"
          parameters:
            qc_settings:
              manual qc:
                x_variable: TIME              # default
                y_variable: PRES              # default; the dashboard dropdown changes it
                boxes:
                  - x: ["2024-05-02T10:00:00", "2024-05-02T14:30:00"]
                    y: [0, 15]
                    flag: 4
                    mode: inside              # flag samples inside the box
                    variables: [PRES, TEMP]   # optional, defaults to [y_variable]
                  - x: ["2024-05-01", "2024-05-20"]
                    y: [0, 1000]
                    flag: 3
                    mode: outside             # flag everything outside the box
                    override: false           # merge by combinatrix, never lower a flag
                  - x: ["2024-05-03T08:12:30"]
                    y: [42.5]
                    flag: 4                   # a point: the nearest sample only
                  - x: ["2024-05-04", "2024-05-05"]
                    y: [10, 12]
                    flag: 3
                    y_variable: TEMP          # drawn on the TEMP plot, flags TEMP
                flag_remaining_good: true     # untouched samples become 1 (good)
          diagnostics: true                   # the plot the dashboard pauses on
    """

    qc_name = "manual qc"
    dynamic = True
    overwrite_flags = True  # Apply QC replaces the columns this returns instead of merging them

    parameter_schema = {
        "x_variable": {
            "type": str,
            "default": "TIME",
            "description": "Variable on the plot's x axis; box x bounds are in its units "
                           "(ISO timestamps for TIME).",
        },
        "y_variable": {
            "type": str,
            "default": "PRES",
            "description": "Variable on the plot's y axis; box y bounds are in its units.",
        },
        "boxes": {
            "type": list,
            "default": [],
            "description": "List of {x: [lo, hi], y: [lo, hi], flag: 0-9, mode: inside|outside, "
                           "variables: [...], override: true} boxes; single-value x/y is a point "
                           "(nearest sample). Drawn in the dashboard, or written by hand.",
        },
        "colour_variable": {
            "type": str,
            "default": None,
            "description": "Colour the plot's points by this variable instead of by flag; flags "
                           "then show as a ring around non-good points.",
        },
        "profile_plot": {
            "type": bool,
            "default": False,
            "description": "With colour_variable: add a side panel of colour_variable vs y_variable "
                           "(a profile view), thinned to 100k points. The dashboard greys the "
                           "points outside the main plot's zoom.",
        },
        "flag_remaining_good": {
            "type": bool,
            "default": True,
            "description": "Give samples no box touches (still flag 0) flag 1 (good). Off leaves them 0.",
        },
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if self.boxes is None:
            self.boxes = []
        self.boxes = [self._check_box(b) for b in self.boxes]

        targets = [self.y_variable]
        axes = [self.x_variable, self.y_variable]
        for box in self.boxes:
            axes += [box["x_variable"], box["y_variable"]]
            for var in box["variables"]:
                if var not in targets:
                    targets.append(var)
        self.target_variables = targets
        if self.colour_variable:
            axes.append(self.colour_variable)
        self.required_variables = list(dict.fromkeys(axes + targets))
        self.qc_outputs = [f"{var}_QC" for var in targets]

    def _check_box(self, box):
        if not isinstance(box, dict):
            raise ValueError(f"[{self.qc_name}] each box must be a mapping, got {box!r}.")
        out = dict(box)
        for axis in ("x", "y"):
            bounds = out.get(axis)
            if not isinstance(bounds, (list, tuple)) or len(bounds) not in (1, 2):
                raise ValueError(f"[{self.qc_name}] box {axis!r} must be [lo, hi] or [value], got {bounds!r}.")
        if len(out["x"]) != len(out["y"]):
            raise ValueError(f"[{self.qc_name}] a point needs single x and y values, got {out['x']!r}, {out['y']!r}.")
        flag = out.get("flag")
        if isinstance(flag, bool) or not isinstance(flag, int) or not 0 <= flag <= 9:
            raise ValueError(f"[{self.qc_name}] box flag {flag!r} must be an Argo QC flag 0-9.")
        mode = str(out.get("mode", "inside")).strip().lower()
        if mode not in ("inside", "outside"):
            raise ValueError(f"[{self.qc_name}] box mode must be 'inside' or 'outside', got {mode!r}.")
        out["mode"] = mode
        out["x_variable"] = str(out.get("x_variable") or self.x_variable)
        out["y_variable"] = str(out.get("y_variable") or self.y_variable)
        variables = out.get("variables") or [out["y_variable"]]
        if isinstance(variables, str):
            variables = [variables]
        out["variables"] = list(variables)
        out["override"] = bool(out.get("override", True))
        return out

    def _axis_values(self, var):
        vals = self.data[var].values
        if np.issubdtype(vals.dtype, np.datetime64):
            return vals.astype("datetime64[ns]").astype("int64").astype(float), True
        return vals.astype(float), False

    @staticmethod
    def _bound(value, is_time):
        # Box bounds arrive as YAML scalars: ISO strings (or datetimes) on a time axis.
        if is_time:
            return float(pd.Timestamp(value).to_datetime64().astype("datetime64[ns]").astype("int64"))
        return float(value)

    def _box_mask(self, box, x, x_time, y, y_time):
        valid = np.isfinite(x) & np.isfinite(y)
        if len(box["x"]) == 1:
            inside = np.zeros_like(valid)
            if valid.any():
                # Nearest valid sample, distances normalised by each axis' data range.
                px, py = self._bound(box["x"][0], x_time), self._bound(box["y"][0], y_time)
                sx = np.ptp(x[valid]) or 1.0
                sy = np.ptp(y[valid]) or 1.0
                d = np.where(valid, ((x - px) / sx) ** 2 + ((y - py) / sy) ** 2, np.inf)
                inside[int(np.argmin(d))] = True
        else:
            x0, x1 = sorted(self._bound(v, x_time) for v in box["x"])
            y0, y1 = sorted(self._bound(v, y_time) for v in box["y"])
            inside = valid & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        return inside if box["mode"] == "inside" else (valid & ~inside)

    def return_qc(self):
        axes = {}
        for box in self.boxes:
            for var in (box["x_variable"], box["y_variable"]):
                if var not in axes:
                    axes[var] = self._axis_values(var)

        # Start from the flags as they stand (Apply QC's store when run there).
        existing = getattr(self, "existing_flags", None)
        qc_arrays = {}
        for var in self.target_variables:
            col = f"{var}_QC"
            if existing is not None and col in existing:
                qc = existing[col].values.astype(int).copy()
            elif col in self.data:
                qc = self.data[col].fillna(9).values.astype(int)
            else:
                qc = np.where(np.isfinite(self.data[var].values.astype(float)), 0, 9)
            qc_arrays[var] = qc
        for box in self.boxes:
            hit = self._box_mask(box, *axes[box["x_variable"]], *axes[box["y_variable"]])
            for var in box["variables"]:
                qc = qc_arrays[var]
                hit_var = hit & (qc != 9)
                qc[hit_var] = box["flag"] if box["override"] else QC_COMBINATRIX[qc[hit_var], box["flag"]]
        if self.flag_remaining_good:
            for qc in qc_arrays.values():
                qc[qc == 0] = 1

        self.flags = xr.Dataset(coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]})
        for var, qc in qc_arrays.items():
            self.flags[f"{var}_QC"] = (("N_MEASUREMENTS",), qc)
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        xv, yv = self.x_variable, self.y_variable
        x = self.data[xv].values
        y = self.data[yv].values
        x_time = np.issubdtype(x.dtype, np.datetime64)

        # Colour by the result (this test's flags include the existing ones), but
        # draw one series per (existing, result) pair with the existing flag in the
        # gid: the dashboard's live preview restarts from it. One legend entry per result.
        existing = getattr(self, "existing_flags", None)
        if existing is not None and f"{yv}_QC" in existing:
            before = existing[f"{yv}_QC"].values.astype(int)
        elif f"{yv}_QC" in self.data:
            before = self.data[f"{yv}_QC"].fillna(9).values.astype(int)
        else:
            before = np.where(np.isfinite(y.astype(float)), 0, 9)
        shown = self.flags[f"{yv}_QC"].values if self.flags is not None and f"{yv}_QC" in self.flags else before

        cv = self.colour_variable
        profile = bool(cv and self.profile_plot)
        fig, axes = fig_spec.new_fig(1, 2, sharey=True, width_ratios=(3, 1)) if profile else fig_spec.new_fig()
        ax = axes[0][0]
        labelled = set()
        for f in range(10):
            for b in range(10):
                m = (shown == f) & (before == b)
                if not m.any():
                    continue
                label = fig_spec.flag_label(f) if f not in labelled else "_"
                labelled.add(f)
                if cv:
                    # Flag as a ring under the coloured fill (drawn after, below); good
                    # points get an invisible ring so the live preview can recolour it.
                    ax.plot(x[m], y[m], ls="", marker="o", markersize=fig_spec.MARKER * 1.9,
                            markeredgewidth=0, color=fig_spec.FLAG_COLOURS[f],
                            alpha=0.0 if f == 1 else 1.0, label=label)
                    ax.lines[-1].set_gid(f"ring:{b}")
                else:
                    fig_spec.points(ax, x[m], y[m], color=fig_spec.FLAG_COLOURS[f], label=label)
                    ax.lines[-1].set_gid(f"flag:{b}")
        if cv:
            c = self.data[cv].values.astype(float)
            # No colorbar (it would drop the dashboard view to PNG); the range is in the title.
            ax.scatter(x, y, c=c, cmap="viridis", s=fig_spec.MARKER ** 2, linewidths=0, label="_fill")
        if profile:
            # Profile view: colour variable on x, same y. Every step-th sample (budget
            # 100k); the gid tells the dashboard the stride back into the main fill.
            step = max(1, int(np.ceil(len(y) / 100_000)))
            pax = axes[0][1]
            pax.scatter(c[::step], y[::step], c=c[::step], cmap="viridis", s=fig_spec.MARKER ** 2,
                        linewidths=0, label="_profile")
            pax.collections[-1].set_gid(f"profile:{step}")
            fig_spec.style_axes(pax, xlabel=fig_spec.axis_label(cv, self.data[cv].attrs.get("units")))
            pax.tick_params(labelleft=False)

        # Box outlines (points as rings), coloured by flag; underscore labels keep them out of the legend.
        for box in self.boxes:
            if (box["x_variable"], box["y_variable"]) != (xv, yv):
                continue
            try:
                bx = sorted(pd.Timestamp(v).to_datetime64() if x_time else float(v) for v in box["x"])
                by = sorted(float(v) for v in box["y"])
            except (TypeError, ValueError):
                continue
            colour = fig_spec.FLAG_COLOURS.get(box["flag"], "k")
            if len(bx) == 1:
                ax.plot(bx, by, "o", mfc="none", mec=colour, ms=9, mew=1.5, label="_point")
                continue
            (x0, x1), (y0, y1) = bx, by
            ax.plot(
                [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "--", lw=1.2, color=colour, label="_box",
            )

        fig_spec.style_axes(
            ax,
            xlabel=fig_spec.axis_label(xv, self.data[xv].attrs.get("units")) if not x_time else "Time",
            ylabel=fig_spec.axis_label(yv, self.data[yv].attrs.get("units")),
        )
        if x_time:
            fig_spec.date_axis(ax, which="x", index=x)
        if yv in ("PRES", "DEPTH"):
            ax.invert_yaxis()
        fig_spec.legend(ax, title="Flag")
        title = f"Manual QC — {yv} vs {xv}"
        if cv:
            lo, hi = np.nanmin(c), np.nanmax(c)
            title += f" · coloured by {cv} ({lo:.3g} – {hi:.3g}, viridis)"
        fig_spec.finish(fig, suptitle=title)
        plt.show(block=True)
