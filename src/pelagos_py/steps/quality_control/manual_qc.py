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
from pelagos_py.utils.qc_handling import QC_COMBINATRIX

#### Custom imports ####
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from pelagos_py.utils import fig_spec


@register_qc
class manual_qc(BaseQC):
    """
    Flag samples inside (or outside) boxes drawn on an ``x_variable`` vs
    ``y_variable`` plot. The dashboard lets you draw the boxes, but the result
    is plain YAML, so the run can be repeated anywhere.

    ``variables`` lists the variables to QC: a name, or a list of names flagged
    together. Each box flags its own ``variables`` (default: its ``y_variable``).

    - A box replaces existing flags, bad -> good included, and later boxes win
      where they overlap. With ``override: false`` it merges by the Argo
      combinatrix instead, so a flag can only get worse.
    - A box with single ``x`` and ``y`` values is a point: it flags the nearest sample.
    - A box can name its own ``x_variable``/``y_variable``, else it uses the test's.
    - ``profiles`` (or ``cycles``) limits a box to those ``PROFILE_NUMBER``
      (or ``CYCLE``) values, so Find Profiles must have run.
    - Missing samples stay 9. Samples no box touches get 1 when
      ``flag_remaining_good`` is on (the default), else keep their flag.

    Target Variable: Any
    Flag Number: Any (user-defined, 0-9)
    Variables Flagged: ``variables``, plus each box's own

    EXAMPLE
    -------
    ::

        - name: "Apply QC"
          parameters:
            qc_settings:
              manual qc:
                variables: [CHLA, [TEMP, PSAL]]   # CHLA alone, TEMP and PSAL together
                x_variable: TIME              # axes for boxes that don't name their own
                y_variable: PRES
                boxes:
                  - {x: ["2024-05-02T10:00:00", "2024-05-02T14:30:00"], y: [0, 15], flag: 4, mode: inside, variables: [TEMP, PSAL]}
                  # everything outside, merged so no flag gets better
                  - {x: ["2024-05-01", "2024-05-20"], y: [0, 1000], flag: 3, mode: outside, override: false}
                  # a point: the nearest sample only
                  - {x: ["2024-05-03T08:12:30"], y: [42.5], flag: 4}
                  - x: ["2024-05-04", "2024-05-05"]
                    y: [10, 12]
                    flag: 3
                    y_variable: TEMP          # drawn on TEMP vs TIME, flags TEMP
                  # only within profile 42
                  - {x: [0.8, 3], y: [0, 20], flag: 4, x_variable: CHLA, y_variable: PRES, variables: [CHLA], profiles: [42]}
                flag_remaining_good: true     # untouched samples become 1 (good)
          diagnostics: true
    """

    qc_name = "manual qc"
    dynamic = True
    overwrite_flags = (
        True  # Apply QC replaces the columns this returns instead of merging them
    )

    parameter_schema = {
        "variables": {
            "type": list,
            "default": [],
            "description": "Variables to QC, one entry per dashboard plot: a name or a list of names "
            "flagged together, e.g. [CHLA, [TEMP, PSAL]].",
        },
        "x_variable": {
            "type": str,
            "default": "TIME",
            "description": "x axis for boxes that don't name their own; box x bounds are in its "
            "units (ISO timestamps for TIME).",
        },
        "y_variable": {
            "type": str,
            "default": "PRES",
            "description": "y axis for boxes that don't name their own; box y bounds are in its units.",
        },
        "boxes": {
            "type": list,
            "default": [],
            "description": "List of {x: [lo, hi], y: [lo, hi], flag: 0-9, mode: inside|outside, "
            "variables: [...], override: true, profiles|cycles: [...]} boxes; single-value x/y "
            "is a point (nearest sample). Drawn in the dashboard, or written by hand.",
        },
        "flag_remaining_good": {
            "type": bool,
            "default": True,
            "description": "Give samples no box touches (still flag 0) flag 1 (good). Off leaves them 0.",
        },
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.boxes = [self._check_box(b) for b in self.boxes]

        self.groups = [[v] if isinstance(v, str) else list(v) for v in self.variables]
        targets = list(dict.fromkeys(v for group in self.groups for v in group))
        axes = [self.x_variable, self.y_variable]
        for box in self.boxes:
            axes += [box["x_variable"], box["y_variable"]]
            if box["scope"]:
                axes.append(box["scope"][0])
            for var in box["variables"]:
                if var not in targets:
                    targets.append(var)
        self.target_variables = targets
        self.required_variables = list(dict.fromkeys(axes + targets))
        self.qc_outputs = [f"{var}_QC" for var in targets]

    def _check_box(self, box):
        if not isinstance(box, dict):
            raise ValueError(
                f"[{self.qc_name}] each box must be a mapping, got {box!r}."
            )
        out = dict(box)
        for axis in ("x", "y"):
            bounds = out.get(axis)
            if not isinstance(bounds, (list, tuple)) or len(bounds) not in (1, 2):
                raise ValueError(
                    f"[{self.qc_name}] box {axis!r} must be [lo, hi] or [value], got {bounds!r}."
                )
        if len(out["x"]) != len(out["y"]):
            raise ValueError(
                f"[{self.qc_name}] a point needs single x and y values, got {out['x']!r}, {out['y']!r}."
            )
        flag = out.get("flag")
        if isinstance(flag, bool) or not isinstance(flag, int) or not 0 <= flag <= 9:
            raise ValueError(
                f"[{self.qc_name}] box flag {flag!r} must be an Argo QC flag 0-9."
            )
        mode = str(out.get("mode", "inside")).strip().lower()
        if mode not in ("inside", "outside"):
            raise ValueError(
                f"[{self.qc_name}] box mode must be 'inside' or 'outside', got {mode!r}."
            )
        out["mode"] = mode
        out["x_variable"] = str(out.get("x_variable") or self.x_variable)
        out["y_variable"] = str(out.get("y_variable") or self.y_variable)
        variables = out.get("variables") or [out["y_variable"]]
        if isinstance(variables, str):
            variables = [variables]
        out["variables"] = list(variables)
        out["override"] = bool(out.get("override", True))
        if "profiles" in out and "cycles" in out:
            raise ValueError(
                f"[{self.qc_name}] a box can limit to profiles or cycles, not both."
            )
        out["scope"] = None
        for key, var in (("profiles", "PROFILE_NUMBER"), ("cycles", "CYCLE")):
            if key in out:
                ids = out[key] if isinstance(out[key], (list, tuple)) else [out[key]]
                out["scope"] = (var, [float(i) for i in ids])
        return out

    def _axis_values(self, var):
        vals = self.data[var].values
        if np.issubdtype(vals.dtype, np.datetime64):
            # NaT would cast to a finite int64, so mask it to NaN like a missing number
            nanos = vals.astype("datetime64[ns]").astype("int64").astype(float)
            nanos[np.isnat(vals)] = np.nan
            return nanos, True
        return vals.astype(float), False

    @staticmethod
    def _bound(value, is_time):
        # on a time axis, bounds are ISO strings or datetimes
        if is_time:
            return float(
                pd.Timestamp(value)
                .to_datetime64()
                .astype("datetime64[ns]")
                .astype("int64")
            )
        return float(value)

    def _box_mask(self, box, x, x_time, y, y_time):
        valid = np.isfinite(x) & np.isfinite(y)
        if box["scope"]:
            var, ids = box["scope"]
            valid &= np.isin(self.data[var].values, ids)
        if len(box["x"]) == 1:
            inside = np.zeros_like(valid)
            if valid.any():
                # Nearest valid sample, distances normalised by each axis' data range.
                px, py = (
                    self._bound(box["x"][0], x_time),
                    self._bound(box["y"][0], y_time),
                )
                sx = np.ptp(x[valid]) or 1.0
                sy = np.ptp(y[valid]) or 1.0
                d = np.where(valid, ((x - px) / sx) ** 2 + ((y - py) / sy) ** 2, np.inf)
                inside[int(np.argmin(d))] = True
        else:
            x0, x1 = sorted(self._bound(v, x_time) for v in box["x"])
            y0, y1 = sorted(self._bound(v, y_time) for v in box["y"])
            inside = valid & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        return inside if box["mode"] == "inside" else (valid & ~inside)

    def _start_flags(self, var):
        # Flags as they stand: Apply QC's store when run there, else the data's _QC, else 0/9
        col = f"{var}_QC"
        if self.existing_flags is not None and col in self.existing_flags:
            return self.existing_flags[col].values.astype(int)
        if col in self.data:
            return self.data[col].fillna(9).values.astype(int)
        return np.where(np.isfinite(self.data[var].values.astype(float)), 0, 9)

    def return_qc(self):
        axes = {}
        for box in self.boxes:
            for var in (box["x_variable"], box["y_variable"]):
                if var not in axes:
                    axes[var] = self._axis_values(var)

        qc_arrays = {var: self._start_flags(var) for var in self.target_variables}
        # Kept for the plot: Apply QC writes the result into existing_flags straight after
        self.start_flags = {var: qc.copy() for var, qc in qc_arrays.items()}
        for box in self.boxes:
            hit = self._box_mask(
                box, *axes[box["x_variable"]], *axes[box["y_variable"]]
            )
            for var in box["variables"]:
                qc = qc_arrays[var]
                hit_var = hit & (qc != 9)
                qc[hit_var] = (
                    box["flag"]
                    if box["override"]
                    else QC_COMBINATRIX[qc[hit_var], box["flag"]]
                )
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
        # One panel per QC'd group, coloured by the flags of its first variable where it has data.
        groups = self.groups or [[v] for v in self.target_variables] or [[yv]]
        fig, axes = fig_spec.new_fig(len(groups), 1, sharex=True)
        for ax, group in zip((row[0] for row in axes), groups):
            var = group[0]
            has = np.isfinite(self.data[var].values.astype(float))
            if f"{var}_QC" in self.flags:
                fig_spec.flag_points(
                    ax, x[has], y[has], self.flags[f"{var}_QC"].values[has]
                )
            else:
                fig_spec.points(ax, x[has], y[has], color="#9aa5ad")
            for box in self.boxes:
                if (box["x_variable"], box["y_variable"]) != (xv, yv) or box[
                    "variables"
                ] != group:
                    continue
                self._draw_box(ax, box, x_time)
            fig_spec.style_axes(
                ax,
                title=" + ".join(group),
                xlabel=fig_spec.axis_label(xv, self.data[xv].attrs.get("units"))
                if not x_time
                else "Time",
                ylabel=fig_spec.axis_label(yv, self.data[yv].attrs.get("units")),
            )
            if x_time:
                fig_spec.date_axis(ax, which="x", index=x)
            if yv in ("PRES", "DEPTH"):
                ax.invert_yaxis()
            fig_spec.legend(ax, title="Flag")
        fig_spec.finish(fig, suptitle=f"Manual QC — {yv} vs {xv}")
        plt.show(block=True)

    @staticmethod
    def _draw_box(ax, box, x_time):
        # a point is drawn as a ring; '_' labels keep boxes out of the legend
        try:
            bx = sorted(
                pd.Timestamp(v).to_datetime64() if x_time else float(v)
                for v in box["x"]
            )
            by = sorted(float(v) for v in box["y"])
        except (TypeError, ValueError):
            return
        colour = fig_spec.FLAG_COLOURS.get(box["flag"], "k")
        if len(bx) == 1:
            ax.plot(bx, by, "o", mfc="none", mec=colour, ms=9, mew=1.5, label="_point")
            return
        (x0, x1), (y0, y1) = bx, by
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            "--",
            lw=1.2,
            color=colour,
            label="_box",
        )
