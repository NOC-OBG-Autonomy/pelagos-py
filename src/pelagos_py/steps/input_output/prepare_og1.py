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

"""A hacky fix to normalise OG1 files onto the names/units the rest of the pipeline assumes.

Planned to be temporary until OG1 files are more consistent, and may need updating as
OG1 becomes more established:
https://oceangliderscommunity.github.io/OG-format-user-manual/OG_Format-v1.0.0.html
"""

from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.processing_utils import cndc_scale_factor

import numpy as np

# Expected name -> source names tried in order when the expected one is absent.
RENAMES = {
    "LATITUDE": ["LATITUDE_GPS", "ALATPT01"],
    "LONGITUDE": ["LONGITUDE_GPS", "ALONPT01"],
    "MOLAR_DOXY": ["DOXY"],
    "TEMP_DOXY": ["TEMPDOXY"],
}

# Temporary fix for CNDC mislabelled with the wrong units, usually S/m vs mS/cm (off by x10).
# Conductivity should never exceed ~7 S/m at sea but reads >10 in mS/cm, so the median
# alone fixes most mislabelled files. Ideally removed once files are correct.
CNDC_MSCM_ABOVE = 8.0

# Some older BODC files name raw beta as BBP700 before converting it, so this renames it
# when BETA_BACKSCATTERING700 isn't in the file. Should also be removed once files are correct.
BETA_NAME, BBP_NAME = "BETA_BACKSCATTERING700", "BBP700"


@register_step
class PrepareOG1(BaseStep):
    """
    Normalise a loaded file into OG1 with the names and units the pipeline expects.

    An optional and hopefully temporary step for data that doesn't properly comply with
    the OG1 format. Should be run straight after ``Load OG1``; applies only the fixes a
    file needs, logging each one (CNDC fixes also raise a warning):

    - Missing ``LATITUDE``/``LONGITUDE`` are renamed from ``LATITUDE_GPS`` or
      ``ALATPT01``/``ALONPT01`` if found instead; ``DOXY`` -> ``MOLAR_DOXY``,
      ``TEMPDOXY`` -> ``TEMP_DOXY``. QC flag variables are renamed alongside.
    - ``CNDC`` labelled S/m but holding mS/cm values (median > 8) is scaled x0.1 to
      S/m and relabelled.
    - When ``BBP700`` is found with no ``BETA_BACKSCATTERING700`` alongside, it can be
      treated as raw beta and renamed (``bbp700_is_beta``).

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
              renames:
                DOWNWELLING_PAR: PAR
    """

    step_name = "Prepare OG1"
    required_variables = []
    provided_variables = []

    parameter_schema = {
        "bbp700_is_beta": {
            "type": bool,
            "default": True,
            "description": "Treat a BBP700 with no BETA_BACKSCATTERING700 alongside as raw "
            "beta and rename it, so later steps can use it.",
        },
        "renames": {
            "type": dict,
            "default": {},
            "description": "Expected name -> the file's name for it (e.g. "
            "{DOWNWELLING_PAR: PAR}); renamed only when the expected one is absent.",
        },
    }

    @staticmethod
    def renames_for(names, renames=None, bbp700_is_beta=True):
        # {source: expected} this step would apply to a file holding `names` --
        # shared with the config builder so it can predict the step's outputs.
        planned = {}
        all_renames = {**RENAMES, **{k: [v] for k, v in (renames or {}).items()}}
        for expected, sources in all_renames.items():
            if expected in names:
                continue
            src = next((s for s in sources if s in names), None)
            if src:
                planned[src] = expected
        if bbp700_is_beta and BBP_NAME in names and BETA_NAME not in names:
            planned[BBP_NAME] = BETA_NAME
        return planned

    def run(self):
        self.check_data()
        self.data = self.context["data"]

        # All-NaN placeholders don't count as present (e.g. an empty LATITUDE_GPS).
        present = {
            v
            for v in self.data.data_vars
            if self.data[v].dtype.kind != "f"
            or bool(np.isfinite(self.data[v].values).any())
        }
        for src, dst in self.renames_for(
            present, self.renames, self.bbp700_is_beta
        ).items():
            # `dst` can only exist here as an all-NaN placeholder, which would block the rename.
            self.data = self.data.drop_vars([dst, f"{dst}_QC"], errors="ignore")
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
