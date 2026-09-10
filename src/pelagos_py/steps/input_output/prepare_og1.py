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

"""Normalise a freshly loaded file onto the OG1 names/units the rest of the pipeline assumes."""

from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.processing_utils import cndc_scale_factor

import numpy as np

# canonical name -> source names tried in order when the canonical one is absent.
RENAMES = {
    "LATITUDE": ["LATITUDE_GPS", "ALATPT01"],
    "LONGITUDE": ["LONGITUDE_GPS", "ALONPT01"],
    "MOLAR_DOXY": ["DOXY"],
    "TEMP_DOXY": ["TEMPDOXY"],
}

# Conductivity never exceeds ~7 S/m at sea, while even brackish water (Baltic ~1 S/m)
# reads >10 mS/cm, so the median alone says which unit the values are really in.
CNDC_MSCM_ABOVE = 8.0

# Some data centres ship raw beta under BBP700 before converting it. Delete this
# block (and the bbp700_is_beta parameter) once every file carries a real BBP700.
BETA_NAME, BBP_NAME = "BETA_BACKSCATTERING700", "BBP700"


@register_step
class PrepareOG1(BaseStep):
    """
    Normalise a loaded file onto the variable names and units the pipeline expects.

    Runs straight after ``Load OG1`` and applies only the fixes a file needs, logging
    each one, so one config serves files with slightly different conventions:

    - Missing ``LATITUDE``/``LONGITUDE`` are renamed from ``LATITUDE_GPS`` or the BODC
      codes ``ALATPT01``/``ALONPT01``; ``DOXY`` -> ``MOLAR_DOXY``, ``TEMPDOXY`` ->
      ``TEMP_DOXY`` likewise (QC companions follow).
    - ``CNDC`` labelled S/m but holding mS/cm values (median > 20) is scaled x0.1 to
      S/m and relabelled; a mislabelled mS/cm attribute on S/m values is relabelled.
      Genuine mS/cm data is left alone.
    - With ``bbp700_is_beta`` (default), a ``BBP700`` with no ``BETA_BACKSCATTERING700``
      alongside is treated as raw beta and renamed, so ``BBP from Beta`` converts it.

    Examples
    --------
    .. code-block:: yaml

        steps:
          - name: Load OG1
            parameters:
              file_path: "/path/to/dataset.nc"
          - name: Prepare OG1
            parameters:
              bbp700_is_beta: true
    """

    step_name = "Prepare OG1"
    required_variables = []
    provided_variables = []

    parameter_schema = {
        "bbp700_is_beta": {
            "type": bool,
            "default": True,
            "description": "Treat a BBP700 with no BETA_BACKSCATTERING700 alongside as raw "
                           "beta and rename it, so 'BBP from Beta' converts it.",
        },
    }

    @classmethod
    def renames_for(cls, names, parameters=None):
        # {source: canonical} this step would apply to a file holding `names` --
        # shared with the config validator so it can predict the step's outputs.
        out = {}
        for canonical, sources in RENAMES.items():
            if canonical in names:
                continue
            src = next((s for s in sources if s in names), None)
            if src:
                out[src] = canonical
        bbp_is_beta = (parameters or {}).get("bbp700_is_beta", True)
        if bbp_is_beta and BBP_NAME in names and BETA_NAME not in names:
            out[BBP_NAME] = BETA_NAME
        return out

    def run(self):
        self.check_data()
        self.data = self.context["data"]

        # All-NaN placeholders don't count as present (e.g. an empty LATITUDE_GPS).
        real = {
            v for v in self.data.data_vars
            if self.data[v].dtype.kind != "f" or bool(np.isfinite(self.data[v].values).any())
        }
        for src, dst in self.renames_for(real, {"bbp700_is_beta": self.bbp700_is_beta}).items():
            mapping = {src: dst}
            if f"{src}_QC" in self.data and f"{dst}_QC" not in self.data:
                mapping[f"{src}_QC"] = f"{dst}_QC"
            self.data = self.data.rename(mapping)
            self.data[dst].attrs["comment"] = (
                f"{self.data[dst].attrs.get('comment', '')} Renamed from {src}.".strip()
            )
            self.log(f"Renamed '{src}' -> '{dst}'.")

        if "CNDC" in self.data:
            self._fix_cndc()

        self.context["data"] = self.data
        return self.context

    def _fix_cndc(self):
        cndc = self.data["CNDC"]
        vals = cndc.values.astype(float)
        if not np.isfinite(vals).any():
            return
        median = float(np.nanmedian(vals))
        values_mscm = median > CNDC_MSCM_ABOVE
        units = cndc.attrs.get("units")
        labelled_mscm = cndc_scale_factor(units) == 1.0
        if values_mscm and not labelled_mscm:
            self.data["CNDC"] = cndc.copy(data=vals * 0.1)
            self.data["CNDC"].attrs["units"] = "S/m"
            self.log_warn(
                f"CNDC labelled '{units}' but its median ({median:.3g}) is mS/cm; "
                "scaled x0.1 to S/m."
            )
        elif labelled_mscm and not values_mscm:
            self.data["CNDC"].attrs["units"] = "S/m"
            self.log_warn(
                f"CNDC labelled '{units}' but its median ({median:.3g}) is S/m; "
                "relabelled as S/m."
            )
