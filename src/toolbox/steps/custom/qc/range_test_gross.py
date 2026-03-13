"""Gross Range Test QC Step."""

#### Mandatory imports ####
import numpy as np
from toolbox.steps.base_test import BaseTest, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib


# TODO: Could be registered within range_test.py
@register_qc
class gross_range_test(BaseTest):
    """
    Outside range test similar to IOOS QC gross range test. Not to be confused with `range test`, which flags within a range.

    Given two values it checks for data points outside of this range and assigns a corresponding flag as defined in the configuration.
    The `variable_ranges` parameter is required for this test, but `also_flag` is not.

    Target Variable: Any
    Flag Number: Any
    Variables Flagged: Any

    EXAMPLE
    -------
    gross range test:
        variable_ranges:
            TEMP:
                3: [0, 30]      #   Flags temperature data outside of this range as probably bad (3)
                4: [-2.5, 40]   #   Flags temperature data outside of this range as bad (4)
            CNDC:
                3: [5, 42]
                4: [2, 45]
        also_flag:
            TEMP: [DOXY] #  Flag DOXY based on TEMP flags
    """

    test_name = "gross range test"
    dynamic = True

    def __init__(self, data, **kwargs):
        required_kwargs = {"variable_ranges"}   #   Removed also_flag, in case test is intended to be run independently
        if not required_kwargs.issubset(kwargs):
            raise KeyError(
                f"{required_kwargs - set(kwargs)} missing from gross range test"
            )
        self.variable_ranges = kwargs["variable_ranges"]
        #   Allow the also_flag param to be blank for this test
        if "also_flag" in kwargs.keys():
            if kwargs["also_flag"] is None:
                self.also_flag = dict() 
            else:
                self.also_flag = kwargs["also_flag"]
        else:
            self.also_flag = dict()
        self.plot = kwargs.get("plot", [])  # Make plotting optional

        self.required_variables = list(self.variable_ranges.keys())
        self.tested_variables = self.required_variables.copy()

        self.qc_outputs = list(
            set(f"{v}_QC" for v in self.tested_variables)
            | set(f"{v}_QC" for v in sum(self.also_flag.values(), []))
        )

        if data is not None:
            self.data = data.copy(deep=True)

    def return_qc(self):
        """Select data outside of the ranges and flag accordingly."""
        # Subset the data
        self.data = self.data[self.required_variables]

        for var in self.tested_variables:
            qc = xr.zeros_like(self.data[var], dtype=int)

            # Apply flags from most severe to least
            for flag in sorted(self.variable_ranges[var], reverse=True):
                low, high = self.variable_ranges[var][flag]

                outside = (self.data[var] < low) | (self.data[var] > high)

                qc = xr.where((qc == 0) & outside, flag, qc)

            # Anything not flagged is good
            qc = xr.where(qc == 0, 1, qc)

            self.data[f"{var}_QC"] = qc

            # Propagate flags
            for extra_var in self.also_flag.get(var, []):
                self.data[f"{extra_var}_QC"] = qc

        # Select just the flags
        self.flags = self.data[[v for v in self.data.data_vars if v.endswith("_QC")]]

        return self.flags

    def plot_diagnostics(self):
        """Visualise the QC results in a similar manner to range_test"""
        matplotlib.use("tkagg")

        # If not plots were specified
        if len(self.plot) == 0:
            self.log_warn(
                "WARNING: In 'range test gross' diagnostics were called but no plots were specified."
            )
            return

        # Plot the QC output
        fig, axs = plt.subplots(nrows=len(self.plot), figsize=(8, 6), dpi=200)
        if len(self.plot) == 1:
            axs = [axs]

        for ax, var in zip(axs, self.plot):
            # Check that the user specified var exists in the test set
            if f"{var}_QC" not in self.qc_outputs:
                self.log_warn(
                    f"WARNING: Cannot plot {var}_QC as it was not included in this test."
                )
                continue

            for i in range(10):
                # Plot by flag number
                plot_data = self.data[[var, "N_MEASUREMENTS"]].where(
                    self.data[f"{var}_QC"] == i, drop=True
                )

                if len(plot_data[var]) == 0:
                    continue

                # Plot the data
                ax.plot(
                    plot_data["N_MEASUREMENTS"],
                    plot_data[var],
                    c=flag_cols[i],
                    ls="",
                    marker="o",
                    label=f"{i}",
                )

            for bounds in self.variable_ranges[var].values():
                for bound in bounds:
                    ax.axhline(bound, ls="--", c="k")

            ax.set(
                xlabel="Index",
                ylabel=var,
                title=f"{var} Range Test",
            )

            ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)
