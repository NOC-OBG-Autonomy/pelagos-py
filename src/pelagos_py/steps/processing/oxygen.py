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

"""Pipeline steps for processing dissolved-oxygen optode data (uncalibrated phase and optode temperature)."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag

#### Custom imports ####
import numpy as np


def check_config(self, expected_params):
    """Runtime checks beyond the parameter schema.

    Parameter *presence* and defaults are handled by ``parameter_schema``; this
    additionally (a) catches method-dependent parameters left unset (``None``) and
    (b) verifies that any ``*_name`` parameter points at a variable that actually
    exists in the dataset.
    """
    for param in expected_params:
        if getattr(self, param, None) is None:
            raise KeyError(f"[{self.step_name}] '{param}' is missing from config")
        if "_name" in param:
            if getattr(self, param) not in self.data.data_vars:
                raise KeyError(
                    f"[{self.step_name}] {getattr(self, param)} could not be found in the data"
                )


@register_step
class DeriveUncalibratedPhase(BaseStep, QCHandlingMixin):

    step_name = "Derive Uncalibrated Phase"

    parameter_schema = {
        "blue_phase_name": {
            "type": str,
            "required": True,
            "description": "Name of the blue-phase variable in the dataset.",
        },
        "red_phase_name": {
            "type": str,
            "default": None,
            "description": "Optional red-phase variable; subtracted from blue phase when given.",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Derive Uncalibrated Phase"
              parameters:
                #  <MANDATORY>
                blue_phase_name: "BPHASE_DOXY"
                # <OPTIONAL>
                red_phase_name: "RPHASE_DOXY"
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check blue_phase_name is present
        check_config(self, ("blue_phase_name",))

        # Check if the output already exists
        if "UNCAL_PHASE_DOXY" in self.data.data_vars:
            self.log_warn("UNCAL_PHASE_DOXY already exists in the data. Overwriting...")

        # Calculate Uncalibrated phase and specify what QC will be derived from
        qc_parents = [f"{self.blue_phase_name}_QC"]
        if self.red_phase_name is not None:
            check_config(self, ("red_phase_name",))
            self.data["UNCAL_PHASE_DOXY"] = (
                self.data[self.blue_phase_name] - self.data[self.red_phase_name]
            )
            qc_parents.append(f"{self.red_phase_name}_QC")
        else:
            self.data["UNCAL_PHASE_DOXY"] = self.data[self.blue_phase_name]

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc({"UNCAL_PHASE_DOXY_QC": qc_parents})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass


@register_step
class DeriveOptodeTemperature(BaseStep, QCHandlingMixin):

    step_name = "Derive Optode Temperature"

    parameter_schema = {
        "temp_voltage_name": {
            "type": str,
            "required": True,
            "description": "Name of the optode temperature-voltage variable in the dataset.",
        },
        "calib_coefficients": {
            "type": list,
            "required": True,
            "description": "Polynomial calibration coefficients (at least two).",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Derive Optode Temperature"
              parameters:
                temp_voltage_name: "TEMP_VOLTAGE_DOXY"
                calib_coefficients: [0, 1, 0, 0, 0, 0]
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the optode temperature voltage and calibration coefficients are present
        check_config(self, ("temp_voltage_name", "calib_coefficients"))

        # Check there are at least two coefficients for the polynomial. Fill in missing values.
        if len(self.calib_coefficients) < 2:
            raise ValueError(
                f"[{self.step_name}] At least two calibration coefficients are required."
            )
        coeffs = [0] * 6
        for i in range(len(self.calib_coefficients)):
            coeffs[i] = self.calib_coefficients[i]

        # Check if the output already exists
        if "TEMP_DOXY" in self.data.data_vars:
            self.log_warn("TEMP_DOXY already exists in the data. Overwriting...")

        # Calculate temp_doxy
        temp_doxy = 0
        for i, coeff in enumerate(coeffs):
            temp_doxy += coeff[i] * self.data[self.temp_voltage_name] ** i
        self.data["TEMP_DOXY"] = temp_doxy

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc({"TEMP_DOXY_QC": [f"{self.temp_voltage_name}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass


@register_step
class PhasePressureCorrection(BaseStep, QCHandlingMixin):

    step_name = "Phase Pressure Correction"

    parameter_schema = {
        "optode_pressure_name": {
            "type": str,
            "required": True,
            "description": "Name of the pressure variable used for the correction.",
        },
        "correction_coefficient": {
            "type": float,
            "required": True,
            "description": "Pressure correction coefficient.",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Phase Pressure Correction"
              parameters:
                optode_pressure_name: "PRES"
                correction_coefficient: 0.1
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the optode pressure and correction coefficient are present and that UNCAL_PHASE_DOXY is in the data
        check_config(self, ("optode_pressure_name", "correction_coefficient"))
        if "UNCAL_PHASE_DOXY" not in self.data.data_vars:
            raise KeyError(
                f"[{self.step_name}] UNCAL_PHASE_DOXY required but is missing from the data"
            )

        # Apply the correction
        self.data["UNCAL_PHASE_DOXY_PCORR"] = (
            self.data["UNCAL_PHASE_DOXY"]
            + 0.001 * self.correction_coefficient * self.data[self.optode_pressure_name]
        )

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {
                "UNCAL_PHASE_DOXY_PCORR_QC": [
                    f"{self.optode_pressure_name}_QC",
                    "UNCAL_PHASE_DOXY_QC",
                ]
            }
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass


@register_step
class DeriveCalibratedPhase(BaseStep, QCHandlingMixin):

    step_name = "Derive Calibrated Phase"

    parameter_schema = {
        "uncalibrated_phase_name": {
            "type": str,
            "required": True,
            "description": "Name of the uncalibrated-phase variable in the dataset.",
        },
        "calib_coefficients": {
            "type": list,
            "required": True,
            "description": "Polynomial calibration coefficients (at least two).",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Derive Calibrated Phase"
              parameters:
                uncalibrated_phase_name: "UNCAL_PHASE_DOXY"
                calib_coefficients: [0, 1, 0, 0]
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the config satisfies requirements
        check_config(self, ("uncalibrated_phase_name", "calib_coefficients"))

        # Check there are at least two coefficients for the polynomial. Fill in missing values.
        if len(self.calib_coefficients) < 2:
            raise ValueError(
                f"[{self.step_name}] At least two calibration coefficients are required."
            )
        coeffs = [0] * 4
        for i in range(len(self.calib_coefficients)):
            coeffs[i] = self.calib_coefficients[i]

        # Check if the output already exists
        if "CAL_PHASE_DOXY" in self.data.data_vars:
            self.log_warn("CAL_PHASE_DOXY already exists in the data. Overwriting...")

        # Calculate cal_phase_doxy
        cal_phase_doxy = 0
        for i, coeff in enumerate(coeffs):
            cal_phase_doxy += coeff * self.data[self.uncalibrated_phase_name] ** i
        self.data["CAL_PHASE_DOXY"] = cal_phase_doxy

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc({"CAL_PHASE_DOXY_QC": [f"{self.uncalibrated_phase_name}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass


@register_step
class DeriveOxygenConcentration(BaseStep, QCHandlingMixin):

    step_name = "Derive Oxygen Concentration"

    parameter_schema = {
        "method": {
            "type": str,
            "required": True,
            "options": ["poly", "SVU"],
            "description": "Conversion method: 'poly' or 'SVU'.",
        },
        "temperature_name": {
            "type": str,
            "required": True,
            "description": "Name of the temperature variable in the dataset.",
        },
        "calib_coefficient_matrix": {
            "type": list,
            "required": True,
            "description": "Calibration coefficient matrix ((5, 4) for poly, (2, 4) for SVU).",
        },
        "temperature_independent_coefficients": {
            "type": list,
            "default": None,
            "description": "[F1, F2] coefficients required by the 'SVU' method.",
        },
    }

    def func_poly(self):
        # Check the calibration matrix has the right shape
        if np.shape(self.calib_coefficient_matrix) != (5, 4):
            raise ValueError(
                f"[{self.step_name}] Calib coefficient matrix must be of shape (5, 4) for method 'poly'."
            )

        # Build the internal coefficient matrix
        coeffs_matrix = np.full((5, 4), 0)
        for i, row in enumerate(self.calib_coefficient_matrix):
            coeffs_matrix[i, :] = row

        # Apply the conversion
        poly_temp = np.array(
            [self.data[self.temperature_name].values ** i for i in range(4)]
        )[np.newaxis, :, :]
        molar_doxy = (
            (poly_temp * coeffs_matrix[:, :, np.newaxis]).sum(axis=1)
            * np.array([self.data["CAL_PHASE_DOXY"].values ** i for i in range(5)])
        ).sum(axis=0)

        return molar_doxy

    def func_SVU(self):
        # Check the calibration matrix has the right shape
        if np.shape(self.calib_coefficient_matrix) != (2, 4):
            raise ValueError(
                f"[{self.step_name}] Calib coefficient matrix must be of shape (2, 4) for method 'poly'."
            )

        # Build the internal coefficient matrix
        coeffs_matrix = np.full((2, 4), 0)
        for i, row in enumerate(self.calib_coefficient_matrix):
            coeffs_matrix[i, :] = row

        F1, F2 = self.temperature_independent_coefficients

        # Apply the conversion
        poly_temp = np.array(
            [self.data[self.temperature_name].values ** i for i in range(4)]
        )[np.newaxis, :, :]
        coeffs = (poly_temp * coeffs_matrix[:, :, np.newaxis]).sum(axis=1)

        # Apply Stern–Volmer equation
        molar_doxy = (
            F1 / (coeffs[0, :] * self.data["CAL_PHASE_DOXY"] + F2) - 1.0
        ) * coeffs[1, :]

        return molar_doxy

    def run(self):
        """
        Example
        -------
        ::

            - name: "Derive Oxygen Concentration"
              parameters:
                # <MANDATORY>
                method: "poly"
                # <METHOD DEPENDENT>
                # The following params are for "poly" method
                temperature_name: "TEMP"
                calib_coefficient_matrix: [
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]
                ]
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        methods = {
            "poly": (self.func_poly, ("temperature_name", "calib_coefficient_matrix")),
            "SVU": (
                self.func_SVU,
                (
                    "temperature_name",
                    "calib_coefficient_matrix",
                    "temperature_independent_coefficients",
                ),
            ),
        }

        # Check the specified method
        check_config(self, ("method",))
        if self.method not in methods.keys():
            raise ValueError(f"[{self.step_name}] Unknown method '{self.method}'")

        # Unpack the method args and functions
        func, args = methods[self.method]

        # Check the config satisfies requirements
        check_config(self, args)

        # Check if the output already exists
        if "MOLAR_DOXY" in self.data.data_vars:
            self.log_warn("MOLAR_DOXY already exists in the data. Overwriting...")

        self.data["MOLAR_DOXY"] = (("N_MEASUREMENTS",), func())

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {"MOLAR_DOXY_QC": ["CAL_PHASE_DOXY_QC", f"{self.temperature_name}_QC"]}
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass


@register_step
class MolarDOXYSalinityCorrection(BaseStep, QCHandlingMixin):

    step_name = "Molar DOXY Salinity Correction"

    parameter_schema = {
        "salinity_name": {
            "type": str,
            "required": True,
            "description": "Name of the salinity variable in the dataset.",
        },
        "temperature_name": {
            "type": str,
            "required": True,
            "description": "Name of the temperature variable in the dataset.",
        },
        "reference_salinity": {
            "type": float,
            "default": 0,
            "description": "Reference salinity the correction is computed relative to.",
        },
    }

    def oxy_solubility_salinity_correction(self):
        # Get data
        T = self.data[self.temperature_name]
        S = self.data[self.salinity_name]

        # Coefficients (Garcia & Gordon 1992 – Benson & Krause refit)
        B0 = -6.24523e-3
        B1 = -7.37614e-3
        B2 = -1.03410e-2
        B3 = -8.17083e-3
        C0 = -4.88682e-7

        # Scaled temperature term Ts
        Ts = np.log((298.15 - T) / (273.15 + T))

        # SCorr computation
        salinity_correction_factor = np.exp(
            (S - self.reference_salinity) * (B0 + B1 * Ts + B2 * (Ts**2) + B3 * (Ts**3))
            + C0 * ((S**2) - (self.reference_salinity**2))
        )

        return salinity_correction_factor

    def water_vapour_partial_pressure(self, reference_salinity=None):
        # Get data
        T = self.data[self.temperature_name]
        if reference_salinity is None:
            S = self.data[self.salinity_name]
        else:
            S = reference_salinity

        # Convert degrees C to Kelvin
        T = T + 273.15

        # Constants from polynomial equation 10 in Weiss&Price, 1980.
        A = 24.4543
        B = -67.4509
        C = -4.8489
        D = -0.000544

        # Equation 10 in Weiss&Price, 1980
        vapour_partial_pressure = 1013.25 * np.exp(
            A + B * (100 / T) + C * np.log(T / 100) + D * S
        )

        return vapour_partial_pressure

    def run(self):
        """
        Example
        -------
        ::

            - name: "Molar DOXY Salinity Correction"
              parameters:
                # <MANDATORY>
                salinity_name: "PRAC_SALINITY"
                temperature_name: "TEMP"
                # <OPTIONAL>
                reference_salinity: 0
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the requred variable names are specified
        check_config(self, ("salinity_name", "temperature_name"))
        if "MOLAR_DOXY" not in self.data.data_vars:
            raise KeyError(
                f"[{self.step_name}] MOLAR_DOXY required but is missing from the data"
            )

        # Calculate factor with partial pressure of water vapour, following Weiss & PRice (1980)
        A = 1013.25 - self.water_vapour_partial_pressure(
            reference_salinity=self.reference_salinity
        )
        B = 1013.25 - self.water_vapour_partial_pressure()

        S_Corr = self.oxy_solubility_salinity_correction()

        MOLAR_DOXY_PSAL = (A / B) * S_Corr * self.data["MOLAR_DOXY"]

        # Apply the correction
        self.data["MOLAR_DOXY_PSAL"] = MOLAR_DOXY_PSAL

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {
                "MOLAR_DOXY_PSAL_QC": [
                    f"{self.salinity_name}_QC",
                    f"{self.temperature_name}_QC",
                    "MOLAR_DOXY_QC",
                ]
            }
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass


@register_step
class MolarDOXYPressureCorrection(BaseStep, QCHandlingMixin):

    step_name = "Molar DOXY Pressure Correction"

    parameter_schema = {
        "pressure_name": {
            "type": str,
            "required": True,
            "description": "Name of the pressure variable in the dataset.",
        },
        "temperature_name": {
            "type": str,
            "required": True,
            "description": "Name of the temperature variable in the dataset.",
        },
        "molar_doxy_name": {
            "type": str,
            "required": True,
            "description": "Name of the molar oxygen variable to correct.",
        },
        "uncalibrated_phase_correction_applied": {
            "type": bool,
            "required": True,
            "description": "Whether the uncalibrated-phase pressure correction was already applied (selects coefficients).",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Molar DOXY Pressure Correction"
              parameters:
                # <MANDATORY>
                pressure_name: "PRES"
                temperature_name: "TEMP"
                molar_doxy_name: "MOLAR_DOXY_PSAL"
                uncalibrated_phase_correction_applied: true
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the required variable names are supplied
        check_config(
            self,
            (
                "pressure_name",
                "temperature_name",
                "molar_doxy_name",
                "uncalibrated_phase_correction_applied",
            ),
        )

        # Set the correction coefficients
        if self.uncalibrated_phase_correction_applied:
            C1, C2 = 0.00022, 0.0419
        else:
            C1, C2 = 0.00025, 0.0328

        MOLAR_DOXY_PSAL_PRES = self.data[self.molar_doxy_name] * (
            1.0
            + (
                (C1 * self.data[self.temperature_name] + C2)
                * self.data[self.pressure_name]
            )
            / 1000
        )

        # Apply the correction
        self.data["MOLAR_DOXY_PSAL_PRES"] = MOLAR_DOXY_PSAL_PRES

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {
                "MOLAR_DOXY_PSAL_PRES_QC": [
                    f"{self.pressure_name}_QC",
                    f"{self.temperature_name}_QC",
                    f"{self.molar_doxy_name}_QC",
                ]
            }
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass
