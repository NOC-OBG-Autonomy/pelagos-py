# This file is part of the NOC Autonomy Toolbox.
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

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import xarray as xr
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

def check_chl_variables(self, allowed_requests):
    user_request = self.apply_to
    if user_request not in self.data.data_vars:
        raise KeyError(f"The variable {user_request} does not exist in the data.")
    if user_request not in allowed_requests:
        raise KeyError(f"The variable {user_request} is not permitted for [{self.step_name}]")

    if f"{user_request}_ADJUSTED" in self.data.data_vars:
        self.log(f"User requested processing on {user_request} but {user_request}_ADJUSTED already exists. Using {user_request}_ADJUSTED...")
        user_request = f"{user_request}_ADJUSTED"

    output_as = user_request + ("_ADJUSTED" if "_ADJUSTED" not in user_request else "")

    self.log(f"Processing {user_request}...")
    return user_request, output_as

@register_step
class chla_deep_correction(BaseStep, QCHandlingMixin):

    step_name = "Chla Deep Correction"
    required_variables = ["TIME", "PROFILE_NUMBER", "DEPTH"]
    provided_variables = []

    def run(self):
        """
        Example
        -------

        - name: "Chla Deep Correction"
          parameters:
            apply_to: "CHLA"
            dark_value: null
            depth_threshold: 200
        diagnostics: true

        Returns
        -------

        """
        self.filter_qc()

        # Check this step is being applied to a valid variable
        self.apply_to, self.output_as = check_chl_variables(
            self,
            ["CHLA",
             "CHLA_ADJUSTED"
             "CHLA_FLUORESCENCE",
             "CHLA_FLUORESCENCE_ADJUSTED"]
        )

        self.compute_dark_value()
        self.apply_dark_correction()

        self.reconstruct_data()
        self.update_qc()

        # Generate new QC if a non-adjusted variable was used in processing (this causes an _ADJUSTED variable to be added)"
        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def compute_dark_value(self):
        """
        Compute dark value for chlorophyll-a correction.

        The dark value represents the sensor's baseline reading in the absence of
        chlorophyll fluorescence. Computed as the median of minimum CHLA values from
        deep profiles (>= depth_threshold).

        Parameters
        ----------
        ds : xarray.Dataset
            Glider dataset with variables: CHLA, DEPTH (or PRES), PROFILE_NUMBER
        depth_threshold : float, optional
            Minimum depth [m] for dark value calculation (default: 200)
        n_profiles : int, optional
            Number of deep profiles to use (default: 5)
        config_path : str or Path, optional
            Path to config file to check for existing dark value

        Returns
        -------
        dark_value : float
            Computed dark value
        profile_data : dict
            Dictionary containing profile information used in calculation
            Keys are profile numbers, values are dicts with 'depth', 'chla', 'min_value', 'min_depth'
        """

        # Check config file for existing dark value
        if self.dark_value:
            self.log(f"Using dark value from config: {self.dark_value}")
            return self.dark_value
        self.log(f"Computing dark value from profiles reaching >= {self.depth_threshold}m")

        # Get DEPTH and CHLA data TODO: Refactor below for user input variables
        missing_vars = {"TIME", "PROFILE_NUMBER", "DEPTH", self.apply_to} - set(self.data.data_vars)
        if missing_vars:
            raise KeyError(f"[Chla Deep Correction] {missing_vars} could not be found in the data.")

        # Convert to pandas dataframe and interpolate the DEPTH data
        interp_data = self.data[["TIME", "PROFILE_NUMBER", "DEPTH", self.apply_to]].to_pandas()
        interp_data["DEPTH"] = interp_data.set_index("TIME")["DEPTH"].interpolate().reset_index(drop=True)
        interp_data = interp_data.dropna(subset=[self.apply_to, "PROFILE_NUMBER"])

        # Subset the data to only deep measurements
        interp_data = interp_data[
            interp_data["DEPTH"] < self.depth_threshold
        ]

        # Remove profiles that do not have CHLA data below the threshold depth
        deep_profiles = interp_data.groupby("PROFILE_NUMBER").agg({self.apply_to: "count"}).reset_index()
        deep_profiles = deep_profiles[deep_profiles[self.apply_to] > 0]["PROFILE_NUMBER"].to_numpy()
        if len(deep_profiles) == 0:
            raise ValueError(
                "[Chla Deep Correction] No deep profiles could be identified. "
                "Try adjusting the 'depth_threshold' parameter."
            )
        interp_data = interp_data[interp_data["PROFILE_NUMBER"].isin(deep_profiles)]

        # Extract the profile number, depth and chla data for all chla minima per profile
        self.chla_deep_minima = interp_data.loc[
            interp_data.groupby("PROFILE_NUMBER")[self.apply_to].idxmin(),
            ["TIME", "PROFILE_NUMBER", "DEPTH", self.apply_to]
        ]

        # Compute median of minimum values
        self.dark_value = np.nanmedian(self.chla_deep_minima[self.apply_to])
        self.log(
            f"\nComputed dark value: {self.dark_value:.6f} "
            f"(median of {len(self.chla_deep_minima)} profile minimums)\n"
            f"Min values range: {np.min(self.chla_deep_minima[self.apply_to]):.6f} "
            f"to {np.max(self.chla_deep_minima[self.apply_to]):.6f}"
        )

    def apply_dark_correction(self):
        """
        Apply dark value correction to CHLA data.
        """

        # Create adjusted chlorophyll variable
        self.data[self.output_as] = xr.DataArray(
            self.data[self.apply_to] - self.dark_value,
            dims=self.data[self.apply_to].dims,
            coords=self.data[self.apply_to].coords,
        )

        # Copy and update attributes
        if hasattr(self.data[self.apply_to], 'attrs'):
            self.data[self.output_as].attrs = self.data[self.apply_to].attrs.copy()
        self.data[self.output_as].attrs["comment"] = (
            f"{self.apply_to} with dark value correction (dark_value={self.dark_value:.6f})"
        )
        self.data[self.output_as].attrs["dark_value"] = self.dark_value

    def generate_diagnostics(self):

        mpl.use("tkagg")

        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

        ax.plot(
            self.chla_deep_minima[self.apply_to],
            self.chla_deep_minima["DEPTH"],
            ls="",
            marker="o",
            c="b"
        )

        ax.axhline(self.depth_threshold, ls="--", c="k", label="Depth Threshold")
        ax.axvline(self.dark_value, ls="--", c="r", label="Dark Value")
        ax.legend(loc="upper right")

        ax.set(
            xlabel=f"{self.apply_to}",
            ylabel="DEPTH",
            title="Deep Minima Values",
        )

        fig.tight_layout()
        plt.show(block=True)

@register_step
class chla_quenching_correction(BaseStep, QCHandlingMixin):

    step_name = "Chla Quenching Correction"
    required_variables = ["PROFILE_NUMBER", "TIME", "DEPTH", "LATITUDE", "LONGITUDE"]
    provided_variables = []

    def run(self):
        """
        Example
        -------

        - name: "Chla Quenching Correction"
          parameters:
            method: "Argo"
            apply_to: "CHLA"
            mld_settings: {
              "threshold_on": "TEMP",
              "reference_depth": 10,
              "threshold": 0.2
              }
            plot_profiles: []
          diagnostics: true
        """

        self.filter_qc()

        # required for plotting the unprocessed data later
        self.data_copy = self.data.copy(deep=True)

        # Check this step is being applied to a valid variable.
        self.apply_to, self.output_as = check_chl_variables(
            self,
            ["CHLA",
             "CHLA_ADJUSTED"
             "CHLA_FLUORESCENCE",
             "CHLA_FLUORESCENCE_ADJUSTED"]
        )
        # If a new "_ADJUSTED" variable will be needed, create it
        if self.apply_to != self.output_as:
            self.data[self.output_as] = self.data[self.apply_to]

        # Get the function call for the specified method
        methods = {
            "argo": self.apply_xing2012_quenching_correction
        }
        if self.method.lower() not in methods.keys():
            raise KeyError(f"Method {self.method} is not supported")
        method_function = methods[self.method.lower()]

        # if the method required sunlight angle, find the inputs for the sun angle calculation
        if self.method.lower() in ["argo"]:
            self.sun_args = self.data[
                ["PROFILE_NUMBER",
                 "TIME",
                 "DEPTH",
                 "LATITUDE",
                 "LONGITUDE"]
            ].to_pandas().dropna()

            # only look at the values nearest the surface and find when and where they were taken
            self.sun_args = (
                self.sun_args.groupby(["PROFILE_NUMBER"])
                .apply(lambda x: x.nlargest(50, "DEPTH"))
                .reset_index(drop=True)
                .groupby(["PROFILE_NUMBER"]).agg(
                    {var: "median" for var in ["TIME", "LATITUDE", "LONGITUDE"]}
                )
            )

        # Subset the data
        method_variable_requirements = {
            "argo": {
                "PROFILE_NUMBER",
                "DEPTH",
                self.apply_to,
                self.mld_settings["threshold_on"]
            }
        }
        data_subset = self.data[
            list(method_variable_requirements[self.method.lower()])
        ]

        # Apply the checks across individual profiles
        profile_numbers = np.unique(data_subset["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS"))
        for profile_number in tqdm(profile_numbers, colour="green", desc='\033[97mProgress\033[0m', unit="prof"):

            # Subset the data
            profile = data_subset.where(data_subset["PROFILE_NUMBER"] == profile_number, drop=True)

            corrected_chla = method_function(profile)

            # Stitch back into the full data
            profile_indices = np.where(self.data["PROFILE_NUMBER"] == profile_number)
            self.data[self.output_as][profile_indices] = corrected_chla

        self.reconstruct_data()
        self.update_qc()

        # Generate new QC if a non-adjusted variable was used in processing (this causes an _ADJUSTED variable to be added)"
        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def calculate_mld(self, profile):

        for k, v in self.mld_settings.items():
            setattr(self, k, v)

        # Only look at values that are below the reference depth
        # TODO: -DEPTH
        profile_subset = profile.where(
            profile["DEPTH"] <= self.reference_depth,
            drop=True
        ).dropna(dim="N_MEASUREMENTS", subset=["DEPTH", self.threshold_on])

        # Check there is still data to work with
        if len(profile_subset["DEPTH"]) == 0:
            return np.nan

        # Find the reference point and return nan if it cant be found near the reference depth
        # TODO: -DEPTH
        reference_point = profile_subset.isel(
            {"N_MEASUREMENTS": np.nanargmax(profile_subset["DEPTH"])},
        )
        if reference_point["DEPTH"] < 2 * self.reference_depth:
            return np.nan

        # Find the difference from the reference value along the profile
        reference_value = reference_point[self.threshold_on]
        profile_subset["delta"] = profile_subset[self.threshold_on] - reference_value

        # Filter out below-threshold points, then select the first (to pass the threshold)
        profile_subset = profile_subset.where(
            np.abs(profile_subset["delta"]) >= np.abs(self.threshold),
            drop=True
        )

        # Return the value if found. Otherwise nan.
        mld_value = np.nan
        if len(profile_subset["DEPTH"]) != 0:
            mld_value = float(profile_subset.isel({"N_MEASUREMENTS": 0})["DEPTH"])
        return mld_value

    def apply_xing2012_quenching_correction(self, profile):
        """
        Apply non-photochemical quenching (NPQ) correction following
        Xing et al. (2012, *JGR–Oceans*, 117:C01019).

        The maximum fluorescence within the mixed-layer depth (MLD)
        is taken as the non-quenched reference. All shallower
        (PRES < z_qd) values are adjusted upward to that maximum.
        Correction is only applied when solar elevation > 0°.

        Parameters
        ----------
        chlf : array-like of shape (N,)
            Uncorrected chlorophyll fluorescence profile F_Chl(PRES).
        pres : array-like of shape (N,)
            Pressure (dbar), increasing with depth.
        mld : float
            Mixed-layer depth (m or dbar).
        sun_angle : float
            Solar elevation angle (degrees). NPQ correction is applied
            only if `sun_angle > 0`.

        Returns
        -------
        chl_corr : ndarray of shape (N,)
            NPQ-corrected fluorescence profile.
        npq : ndarray of shape (N,)
            NPQ index = (chl_corr − chlf) / chlf.
        z_qd : float
            Quenching depth (dbar): pressure of maximum fluorescence
            within the MLD. NaN if not computable or if night-time.

        Notes
        -----
        • No correction is applied if solar elevation ≤ 0° (nighttime).
        • Shallower than z_qd → fluorescence set to Fmax (non-quenched reference).
        • Below MLD → unchanged.
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        N = len(chlf)

        # --- Calculate the MLD for this profile
        # TODO: -DEPTH
        mld = self.calculate_mld(profile)

        # --- Night-time or invalid inputs: skip correction
        profile_number = int(profile["PROFILE_NUMBER"][0])
        time, lat, long = self.sun_args.loc[profile_number].to_numpy()
        time_utc = pd.to_datetime(time).tz_localize("UTC")
        solar_position = pvlib.solarposition.get_solarposition(time_utc, lat, long)
        sun_angle = solar_position["elevation"].values
        if (
                sun_angle <= 0
                or N == 0
                or len(depth) != N
                or not np.isfinite(mld)
                or mld >= 0
                or np.all(np.isnan(chlf))
        ):
            return chlf

        # --- Identify max F_Chl within MLD
        # TODO: -DEPTH
        within_mld = depth >= mld
        if not np.any(within_mld):
            return chlf

        chlf_mld = np.where(within_mld, chlf, np.nan)
        idx_max, chlf_max = np.nanargmax(chlf_mld), np.nanmax(chlf_mld)
        chlf_max_depth = float(depth[idx_max])

        # --- Apply correction: flatten shallower than z_qd
        #TODO: -DEPTH
        chl_corr = np.copy(chlf)
        chl_corr[(depth >= chlf_max_depth) & (~np.isnan(chlf))] = chlf_max

        return chl_corr

    def generate_diagnostics(self):
        mpl.use("tkagg")

        if len(self.plot_profiles) == 0:
            self.log("To see diagnostics, please specify the plot_profiles setting.")
            return

        nrows = int(np.ceil(len(self.plot_profiles) / 3))
        fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(12, nrows * 6), dpi=200)

        for profile_number, ax in zip(self.plot_profiles, axs.flatten()):

            for data, var, col, label in zip(
                    [self.data_copy, self.data],
                    [self.apply_to, self.output_as],
                    ["r", "b"],
                    ["Uncorrected", "Corrected"]
            ):
                # Select the raw profile data
                profile = data.where(
                    data["PROFILE_NUMBER"] == profile_number,
                    drop=True
                ).dropna(dim="N_MEASUREMENTS", subset=[var, "DEPTH"])

                if len(profile[var]) == 0:
                    ax.text(0.5, 0.5,
                            f"Missing Data\n--Prof. {profile_number}--",
                            ha='center', va='center',
                            transform=ax.transAxes)
                    continue

                ax.plot(
                    profile[var],
                    profile["DEPTH"],
                    c=col,
                    ls="",
                    marker="o",
                    label=label
                )

                ax.set(
                    xlabel=self.apply_to,
                    ylabel="DEPTH",
                )

                ax.legend(
                    title=f"Prof. {profile_number}",
                    loc="lower right"
                )

        fig.suptitle("Quenching Correction")
        fig.tight_layout()
        plt.show(block=True)