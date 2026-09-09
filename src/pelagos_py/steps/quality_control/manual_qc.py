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

    Boxes apply most-severe-flag-first, so the worse flag wins on overlap.
    Samples no box touches keep flag 0 here and so are left as they were when
    Apply QC merges the result (a manual flag can raise but never lower an
    existing flag — Argo merge rules).

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
          diagnostics: true                   # the plot the dashboard pauses on
    """

    qc_name = "manual qc"
    dynamic = True

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
                           "variables: [...]} boxes. Drawn in the dashboard, or written by hand.",
        },
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if self.boxes is None:
            self.boxes = []
        self.boxes = [self._check_box(b) for b in self.boxes]

        targets = []
        for box in self.boxes:
            for var in box["variables"]:
                if var not in targets:
                    targets.append(var)
        self.target_variables = targets
        self.required_variables = list(dict.fromkeys([self.x_variable, self.y_variable] + targets))
        self.qc_outputs = [f"{var}_QC" for var in targets]

    def _check_box(self, box):
        if not isinstance(box, dict):
            raise ValueError(f"[{self.qc_name}] each box must be a mapping, got {box!r}.")
        out = dict(box)
        for axis in ("x", "y"):
            bounds = out.get(axis)
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"[{self.qc_name}] box {axis!r} must be [lo, hi], got {bounds!r}.")
        flag = out.get("flag")
        if isinstance(flag, bool) or not isinstance(flag, int) or not 0 <= flag <= 9:
            raise ValueError(f"[{self.qc_name}] box flag {flag!r} must be an Argo QC flag 0-9.")
        mode = str(out.get("mode", "inside")).strip().lower()
        if mode not in ("inside", "outside"):
            raise ValueError(f"[{self.qc_name}] box mode must be 'inside' or 'outside', got {mode!r}.")
        out["mode"] = mode
        variables = out.get("variables") or [self.y_variable]
        if isinstance(variables, str):
            variables = [variables]
        out["variables"] = list(variables)
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
        x0, x1 = sorted(self._bound(v, x_time) for v in box["x"])
        y0, y1 = sorted(self._bound(v, y_time) for v in box["y"])
        valid = np.isfinite(x) & np.isfinite(y)
        inside = valid & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        return inside if box["mode"] == "inside" else (valid & ~inside)

    def return_qc(self):
        n = len(self.data["N_MEASUREMENTS"])
        x, x_time = self._axis_values(self.x_variable)
        y, y_time = self._axis_values(self.y_variable)

        qc_arrays = {var: np.zeros(n, dtype=int) for var in self.target_variables}
        for box in sorted(self.boxes, key=lambda b: b["flag"], reverse=True):
            hit = self._box_mask(box, x, x_time, y, y_time)
            for var in box["variables"]:
                qc = qc_arrays[var]
                qc[hit & (qc == 0)] = box["flag"]

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

        # Show what the merged result will look like: existing flags (if any) combined
        # with this test's, so untouched points keep their colour rather than reading as 0.
        n = len(y)
        shown = np.zeros(n, dtype=int)
        if f"{yv}_QC" in self.data:
            shown = self.data[f"{yv}_QC"].fillna(9).values.astype(int)
        if self.flags is not None and f"{yv}_QC" in self.flags:
            shown = QC_COMBINATRIX[shown, self.flags[f"{yv}_QC"].values]

        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        fig_spec.flag_points(ax, x, y, shown)

        # Box outlines, coloured by their flag; underscore labels keep them out of the legend.
        for box in self.boxes:
            try:
                x0, x1 = sorted(pd.Timestamp(v).to_datetime64() if x_time else float(v) for v in box["x"])
                y0, y1 = sorted(float(v) for v in box["y"])
            except (TypeError, ValueError):
                continue
            ax.plot(
                [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "--", lw=1.2,
                color=fig_spec.FLAG_COLOURS.get(box["flag"], "k"), label="_box",
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
        fig_spec.finish(fig, suptitle=f"Manual QC — {yv} vs {xv}")
        plt.show(block=True)
