"""Class definition for exporting data steps."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
import numpy as np
import re

def _parse_duration_to_kwargs(duration: str) -> dict:
    m = re.fullmatch(r"(\d+)([smhd])", duration)
    if not m:
        raise ValueError(f"Unsupported duration format: {duration}")
    val, unit = int(m.group(1)), m.group(2)
    return {"s": dict(seconds=val),
            "m": dict(minutes=val),
            "h": dict(hours=val),
            "d": dict(days=val)}[unit]

def find_profiles(
    df,
    gradient_thresholds=[0.02, -0.02],
    filter_win_sizes=["20s", "10s"],
    time_col="TIME",
    depth_col="DEPTH",
    profile_duration='2m',
    transect_duration='10m',
    transect_depth_range=[10.0],
    transect_depth_bottom_limits=None,
    cust_col=None,
    cust_gradient_thresholds=[0.005, -0.025],
    strict_profiles=False,
    use_only_pit_vel=False,
    backfill_segments=False,
    back_fill_mod=1,
    force_split_up_down_casts=False,  #newline17112025
):

    """
    Identifies vertical profiles in oceanographic or similar data by analyzing depth gradients over time.

    This function processes depth-time data to identify periods where an instrument is performing
    vertical profiling based on gradient thresholds. It handles data interpolation, calculates vertical
    velocities, applies median filtering, and assigns unique profile numbers to identified profiles.

    Parameters
    ----------
    df : polars.DataFrame
        Input dataframe containing time and depth measurements.
        Time column must be in epoch seconds.
    gradient_thresholds : list
        Two-element list [positive_threshold, negative_threshold] defining the vertical velocity
        range (in meters/second) that is NOT considered part of a profile. typical values are around [0.02, -0.02]
    filter_win_sizes : list, default= ['20s', '10s']
        Window sizes for the compound filter applied to gradient calculations, in Polars duration format.
        index 0 controls the rolling median window size and index 1 controls the rolling mean window size.
    time_col : str, default='TIME'
        Name of the column containing timestamp data
    depth_col : str, default='DEPTH'
        Name of the column containing depth measurements
    profile_duration : str, default='2m'
        Minimum duration for a profile to be considered valid, in Polars duration format (e.g. '30s', '5m', '1h').
    transect_duration : str, default='10m'
        Minimum duration for a transect to be considered valid, in Polars duration format (e.g. '30s', '5m', '1h').
   transect_depth_range : int64, default=10.0
        Maximum depth difference (in meters) from transect start for a transect to be considered continuous/valid.
    cust_col : str, default=None
        Name of a data column in the input dataframe, with matching time and depth measurements, to be displayed
        alongside profiling plots, e.g. pitch
    cust_gradient_thresholds : list, default=None
        Two-element list [positive_threshold, negative_threshold] defining the
        range of your custom variable that is NOT considered part of a profile.
    strict_profiles : bool, default=False
        If True, a point must meet BOTH gradient and pitch criteria to be considered part of a profile.
    use_only_pit_vel : bool, default=False
        If True, only a (-)pitch*velocity product is used to threshold profiles. The threshold setting is tied to
        the "cust_gradient_thresholds" parameter. Only the value in index 0 is used, the value in index 1 is ignored.
    backfill_segments : bool, default=False
        If True, extends identified profile and transect segments backwards in time by an amount defined by
        filter_win_sizes[0] * back_fill_mod.
    back_fill_mod : int64, default = 1
        Multiplier for the backfill extension duration. filter_win_sizes[0] * back_fill_mod gives the total 
        backfill duration.
        
    Returns
    -------
    polars.DataFrame
        Dataframe with additional columns:
        - 'dt': Time difference between consecutive points (seconds)
        - 'dz': Depth difference between consecutive points (meters)
        - 'grad': Vertical velocity (dz/dt, meters/second)
        - 'smooth_grad': Median-filtered vertical velocity
        - 'is_profile': Boolean indicating if a point belongs to a profile
        - 'profile_num': Unique identifier for each identified profile (0 for non-profile points)
        - NEW : 'phase': Developed for ALR data, indicates the phase of the platform, 0 for downcast, 1 for upcast, 2 for transect

    Notes
    -----
    - The function considers a point part of a profile when its smoothed vertical velocity
      falls outside the range specified by gradient_thresholds.
    - 'depth_col' does not strictly have to be a depth measurement, any variable which follows
      the profile shape (such as pressure) could also be used, though this would change the units
      and interpretation of grad.
    """

    # Get the unedited shape for padding later (to make the input and outputs the same length)
    df_full_len = len(df)

    # Interpolate missing depth values using time as reference
    # Also removes infinite and NaN values before interpolation
    cols = [pl.col(time_col), pl.col(depth_col)]
    if cust_col and cust_col in df.columns:
        cols.append(pl.col(cust_col))

    df = (
        df.select(
            *cols,
            *(pl.col(c).replace([np.inf, -np.inf, np.nan], None)
                        .interpolate_by(time_col)
                        .name.prefix("INTERP_") for c in [depth_col, cust_col] if c)
        )
        .with_row_index()
        .drop_nulls(subset=[f"INTERP_{depth_col}",f"INTERP_{cust_col}"])
    )

    # Calculate time differences (dt) and depth differences (dz) between consecutive measurements
    df = df.with_columns(
        (pl.col(time_col).diff().cast(pl.Float64) * 1e-9).alias(
            "dt"
        ),  # Convert nanoseconds to seconds
        pl.col(f"INTERP_{depth_col}").diff().alias("dz"),
        pl.col(f"INTERP_{cust_col}").diff().alias("dC")
    )

    # Calculate vertical velocity (gradient) as depth change divided by time change
    df = df.with_columns(
        (pl.col("dz") / pl.col("dt")).alias("grad"),
    ).drop_nulls(subset="grad")

    df = df.with_columns(
        (pl.col("dC") / pl.col("dt")).alias("dC/dt"),
    ).drop_nulls(subset="dC/dt")

    # Apply a compound filter to smooth the gradient values (rolling median
    # supresses spikes, rolling mean smooths noise)
    # TODO: this window size should be checked against the maximum sample period (dt)
    df = df.with_columns(
        pl.col("grad", f"INTERP_{cust_col}", "dC/dt")
        .rolling_median_by(time_col, window_size=filter_win_sizes[0])
        .rolling_mean_by(time_col, window_size=filter_win_sizes[1])
        .name.prefix("smooth_"),
    )

    # Determine which points are part of profiles based on gradient thresholds
    pos_grad, neg_grad = gradient_thresholds
    df = df.with_columns(
        pl.col("smooth_grad").is_between(neg_grad, pos_grad).not_().alias("grad_profile")
    )

    # Determine which points are part of profiles based on pitch angle and vertical velocity
    if cust_col == 'pitch':
        if cust_gradient_thresholds:
            pos_grad, neg_grad = cust_gradient_thresholds

        combined_metric = -pl.col(f"smooth_INTERP_{cust_col}") * pl.col("smooth_grad")

        df = df.with_columns(
            (combined_metric.is_between(neg_grad, pos_grad).not_()).alias("pitch_profile")
        )
    else:
        df = df.with_columns(
            pl.lit(False).alias("pitch_profile")
        )

    if use_only_pit_vel:
        # Ignore gradient thresholding, just use pitch product
        df = df.with_columns(pl.col("pitch_profile").alias("is_profile"))

    else:
        if strict_profiles:
            df = df.with_columns(
                (pl.col("grad_profile") & pl.col("pitch_profile")).alias("is_profile")
            )
        else:
            df = df.with_columns(
                (pl.col("grad_profile") | pl.col("pitch_profile")).alias("is_profile")
            )


    # Assign unique profile numbers to consecutive points identified as profiles
    # This converts the boolean 'is_profile' column into numbered profile segments
    df = df.with_columns(
        (pl.col("is_profile").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
        * pl.col("is_profile")
        - 1
        ).alias("profile_num")
    )

    # Enforce minimum profile duration
    # Compute elapsed time for each profile segment
    profile_durations = (
        df.filter(pl.col("profile_num") >= 0)
          .group_by("profile_num")
          .agg((pl.col(time_col).max() - pl.col(time_col).min()).alias("elapsed"))
    )

    # Join back and mask out short profiles
    df = df.join(profile_durations, on="profile_num", how="left")

    kwargs = _parse_duration_to_kwargs(profile_duration)
    df = df.with_columns(
        pl.when(pl.col("elapsed") >= pl.duration(**kwargs))
        .then(pl.col("is_profile"))
        .otherwise(False)
        .alias("is_profile")
    ).drop("elapsed")

    # Re-number profiles after duration filter (so IDs match valid segments)
    df = df.with_columns(
        (pl.col("is_profile").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
            * pl.col("is_profile")
            - 1
        ).alias("profile_num_new")
    )

    # Parse both filter windows to seconds
    m0 = re.fullmatch(r"(\d+)([smhd])", filter_win_sizes[0])
    if not m0:
        raise ValueError("Unsupported duration format in filter_win_sizes")
    v0, u0 = int(m0.group(1)), m0.group(2)
    w0 = v0 * {"s": 1, "m": 60, "h": 3600, "d": 86400}[u0]

    # Approximate group delay of median+mean (causal): half each window
    # backfill_seconds = back_fill_mod * (w0/2 + w1/2)
    backfill_seconds = back_fill_mod * w0

    if backfill_segments:

        # Build extended intervals for each profile
        profile_bounds = (
            df.filter(pl.col("is_profile"))
            .group_by("profile_num")
            .agg(pl.col(time_col).min().alias("start"),
                pl.col(time_col).max().alias("end"))
            .with_columns((pl.col("start") - pl.duration(seconds=backfill_seconds)).alias("start_ext"))
        )

        # Create a mask by checking if each row falls into ANY extended interval
        # This avoids needing to join on profile_num
        mask = None
        for row in profile_bounds.iter_rows(named=True):
            cond = (
                (pl.col(time_col) >= row["start_ext"]) &
                (pl.col(time_col) <= row["end"])
            )
            mask = cond if mask is None else (mask | cond)

        if mask is not None:
            df = df.with_columns(
                pl.when(mask)
                .then(True)
                .otherwise(pl.col("is_profile"))
                .alias("is_profile")
            )

            # Re-number profiles after back-extension
            df = df.with_columns(
                (pl.col("is_profile").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
                    * pl.col("is_profile")
                    - 1
                ).alias("profile_num_new")
            )

    # --- OPTIONAL: force splits between downcasts and upcasts -----------------  #newline17112025
    # This section modifies ONLY the final profile numbering when                 #newline17112025
    # 'force_split_up_down_casts' is True and pitch is available.                #newline17112025
    #                                                                            #newline17112025
    # Strategy (simpler and more robust):                                        #newline17112025
    #   - Use the sign of the smoothed pitch (smooth_INTERP_pitch).              #newline17112025
    #   - While inside a profile (is_profile == True), start a new profile       #newline17112025
    #     whenever that sign flips.                                              #newline17112025
    #   - Also start a new profile when we first enter a profile segment.        #newline17112025
    #                                                                            #newline17112025
    # This guarantees that a downcast followed by an upcast will always be       #newline17112025
    # given two different profile_num_new values, even if backfill has made      #newline17112025
    # is_profile a single continuous block.                                      #newline17112025
    if force_split_up_down_casts and (cust_col == 'pitch'):                       #newline17112025
        # 1) Instantaneous pitch sign from smoothed pitch                         #newline17112025
        df = df.with_columns(                                                     #newline17112025
            pl.when(pl.col(f"smooth_INTERP_{cust_col}") >= 0)                     #newline17112025
              .then(pl.lit(1))                                                    #newline17112025
              .otherwise(pl.lit(-1))                                              #newline17112025
              .alias("pitch_sign")                                                #newline17112025
        )                                                                          #newline17112025
                                                                                    #newline17112025
        # 2) Previous state for profile mask and pitch sign                       #newline17112025
        df = df.with_columns([                                                    #newline17112025
            pl.col("is_profile").shift(1).fill_null(False).alias("prev_is_profile"), #newline17112025
            pl.col("pitch_sign").shift(1).alias("prev_pitch_sign")                #newline17112025
        ])                                                                         #newline17112025
                                                                                    #newline17112025
        # 3) Segment starts:                                                      #newline17112025
        #    - entering a profile (False -> True), OR                             #newline17112025
        #    - staying in a profile but pitch sign flips.                         #newline17112025
        df = df.with_columns(                                                     #newline17112025
            pl.when(                                                              #newline17112025
                pl.col("is_profile") &                                            #newline17112025
                (                                                                 #newline17112025
                    (~pl.col("prev_is_profile")) |                                #newline17112025
                    (pl.col("prev_is_profile") &                                  #newline17112025
                     (pl.col("pitch_sign") != pl.col("prev_pitch_sign")).fill_null(False)) #newline17112025
                )                                                                 #newline17112025
            )                                                                     #newline17112025
            .then(1)                                                              #newline17112025
            .otherwise(0)                                                         #newline17112025
            .alias("cast_start_flag")                                             #newline17112025
        )                                                                          #newline17112025
                                                                                    #newline17112025
        # 4) Rebuild profile_num_new from these starts.                           #newline17112025
        #    Each cast_start_flag == 1 increments the cast index; we subtract     #newline17112025
        #    1 so the first profile has ID 0, next has ID 1, etc.                 #newline17112025
        df = df.with_columns(                                                     #newline17112025
            pl.when(pl.col("is_profile"))                                         #newline17112025
              .then(                                                              #newline17112025
                  pl.col("cast_start_flag")                                       #newline17112025
                    .cast(pl.Int64)                                               #newline17112025
                    .cum_sum()                                                    #newline17112025
                    - 1                                                           #newline17112025
              )                                                                   #newline17112025
              .otherwise(pl.lit(-1))                                              #newline17112025
              .alias("profile_num_new")                                           #newline17112025
        )                                                                          #newline17112025
                                                                                    #newline17112025
    elif force_split_up_down_casts and (cust_col != 'pitch'):                      #newline17112025
        # No-op: user requested pitch-based splitting, but no pitch column        #newline17112025
        # is available. We deliberately do nothing to avoid breaking workflows.   #newline17112025
        pass                                                                        #newline17112025
    # ---------------------------------------------------------------------------  #newline17112025

    # Rolling min and max of depth over the transect window duration
    df = df.with_columns([
        pl.col(f"INTERP_{depth_col}")
        .rolling_min_by(time_col, window_size=transect_duration)
        .alias("roll_min_depth"),
        pl.col(f"INTERP_{depth_col}")
        .rolling_max_by(time_col, window_size=transect_duration)
        .alias("roll_max_depth"),
    ])
    
    # Check that depth stays within user defined transect depth window
    if isinstance(transect_depth_range, (list, tuple)):
        ref_depth_range = transect_depth_range[0]
    else:
        ref_depth_range = transect_depth_range

    df = df.with_columns(
        ((pl.col("roll_max_depth") - pl.col("roll_min_depth")) <= ref_depth_range)
        .alias("depth_stable")
    )

    # Check that the pitch*vertical velocity product is low
    df = df.with_columns(
        (combined_metric < pos_grad).alias("pitch_flat")
    )

    # Transect defined as stable depth, 'flat' pitch, and NOT a profile
    df = df.with_columns(
        ((pl.col("depth_stable")) & (pl.col("pitch_flat")) & (pl.col("is_profile") == False))
        .alias("is_transect")
    )

    # Numbering transects
    df = df.with_columns(
        (
            pl.col("is_transect").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
            * pl.col("is_transect")
            - 1
        ).alias("transect_num")
    )

    # Enforce minimum transect duration
    transect_durations = (
        df.filter(pl.col("transect_num") >= 0)
          .group_by("transect_num")
          .agg((pl.col(time_col).max() - pl.col(time_col).min()).alias("elapsed"))
    )

    # Enforce maximum transect depth range
    transect_depths = (
        df.filter(pl.col("transect_num") >= 0)
          .group_by("transect_num")
          .agg((pl.col(f"INTERP_{depth_col}").max() - pl.col(f"INTERP_{depth_col}").min()).alias("depth_span"))
    )

    # Compute mean depth per transect
    transect_means = (
        df.filter(pl.col("transect_num") >= 0)
          .group_by("transect_num")
          .agg(pl.col(f"INTERP_{depth_col}").mean().alias("mean_depth"))
    )

    df = (df.join(transect_durations, on="transect_num", how="left")
            .join(transect_depths, on="transect_num", how="left")
            .join(transect_means, on="transect_num", how="left"))

    # Validate and build depth-dependent window mask
    if len(transect_depth_range) > 1:
        if not transect_depth_bottom_limits:
            raise ValueError("transect_depth_bottom_limits must be provided when multiple transect_depth_range values are used.")
        if len(transect_depth_range) - 1 != len(transect_depth_bottom_limits):
            raise ValueError("transect_depth_bottom_limits must have length len(transect_depth_range) - 1")

        expr = None
        for i, rng in enumerate(transect_depth_range):
            if i == 0:
                cond = pl.col("mean_depth") <= transect_depth_bottom_limits[0]
            elif i == len(transect_depth_range) - 1:
                cond = pl.col("mean_depth") > transect_depth_bottom_limits[-1]
            else:
                cond = (
                    (pl.col("mean_depth") > transect_depth_bottom_limits[i - 1]) &
                    (pl.col("mean_depth") <= transect_depth_bottom_limits[i])
                )
            expr = pl.when(cond).then(rng) if expr is None else expr.when(cond).then(rng)
        expr = expr.otherwise(transect_depth_range[-1])
        df = df.with_columns(expr.alias("depth_window"))
    else:
        df = df.with_columns(pl.lit(transect_depth_range[0]).alias("depth_window"))
    
    kwargs = _parse_duration_to_kwargs(transect_duration)
    df = df.with_columns(
        pl.when((pl.col("elapsed") >= pl.duration(**kwargs)) &
                (pl.col("depth_span") <= pl.col("depth_window")))
        .then(pl.col("is_transect"))
        .otherwise(False)
        .alias("is_transect")
    ).drop(["elapsed", "depth_span", "mean_depth", "depth_window"])

        # Re-number transects after filtering (so IDs match valid segments)
    df = df.with_columns(
        (pl.col("is_transect").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
            * pl.col("is_transect")
            - 1
        ).alias("transect_num")
    )

    if backfill_segments:

        # Force back-extension of transects by filter_win_sizes[0] * back_fill_mod
        transect_bounds = (
            df.filter(pl.col("is_transect"))
            .group_by("transect_num")
            .agg(pl.col(time_col).min().alias("start"),
                pl.col(time_col).max().alias("end"))
            .with_columns((pl.col("start") - pl.duration(seconds=backfill_seconds)).alias("start_ext"))
            .select(["transect_num", "start_ext", "end"])
        )
        df = df.join(transect_bounds, on="transect_num", how="left")

        df = df.with_columns(
            pl.when(
                (pl.col("start_ext").is_not_null()) &
                (pl.col(time_col) >= pl.col("start_ext")) &
                (pl.col(time_col) <= pl.col("end")) &
                (~pl.col("is_profile"))  # <- keep transects out of profiles
            )
            .then(True)
            .otherwise(pl.col("is_transect"))   # <- preserve existing mask, only extend
            .alias("is_transect")
        ).drop(["start_ext", "end"])

        # Re-number transects after back-extension
        df = df.with_columns(
            (
                pl.col("is_transect").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
                * pl.col("is_transect")
                - 1
            ).alias("transect_num")
        )

    # Reforming the full length dataframe (This executes faster than polars join or merge methods)
    front_pad = np.full((df["index"].min(), len(df.columns)), np.nan)
    end_pad = np.full((df_full_len - df["index"].max() - 1, len(df.columns)), np.nan)

    data = np.vstack((front_pad, df.to_numpy(), end_pad))
    padded_df = pl.DataFrame(data, schema=df.columns).drop("index")

    return padded_df


@register_step
class FindProfilesStep(BaseStep):
    step_name = "Find Profiles"

    def run(self):
        self.log("Attempting to designate profile numbers")

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        else:
            self.log(f"Data found in context.")
        self.data = self.context["data"]
        self.thresholds = self.parameters["gradient_thresholds"]
        self.win_sizes = self.parameters["filter_window_sizes"]
        self.depth_col = self.parameters["depth_column"]
        self.cust_col = self.parameters.get("custom_column", None)  # <<< NEW

        if self.diagnostics:
            self.log("Generating diagnostics")
            root = self.generate_diagnostics()
            root.mainloop()

        # Convert to polars for processing
        self._df = pl.from_pandas(
            self.data[["TIME", self.depth_col]].to_dataframe(), nan_to_null=False
        )
        self.profile_outputs = find_profiles(
            self._df, self.thresholds, self.win_sizes, depth_col=self.depth_col, cust_col=self.cust_col
        )
        profile_numbers = self.profile_outputs["profile_num"].to_numpy()

        # Add profile numbers to data and update context
        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), profile_numbers)
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. #=-1 indicates no profile, #>=0 are profiles.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": -1,
            "valid_max": np.inf,
        }

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):

        def generate_plot():
            # Allows interactive plots
            mpl.use("TkAgg")

            # Update data for plot
            self._df = pl.from_pandas(
                self.data[["TIME", self.depth_col]].to_dataframe(), nan_to_null=False
            )
            self.profile_outputs = find_profiles(
                self._df, self.thresholds, self.win_sizes, depth_col=self.depth_col
            )

            # Split data into profile and non-profile points for plotting
            profiles = self.profile_outputs.drop_nans().filter(
                pl.col("is_profile").cast(pl.Boolean)
            )
            not_profiles = self.profile_outputs.drop_nans().filter(
                pl.col("is_profile").cast(pl.Boolean).not_()
            )

            fig, axs = plt.subplots(
                4, 1, figsize=(18, 10), height_ratios=[3, 3, 1, 2], sharex=True  # <<< CHANGED
            )
            axs[0].set(
                xlabel="Time", 
                ylabel="Interpolated Depth",
            )
            axs[1].set(
                xlabel="Time", 
                ylabel="Vertical Velocity",
            )
            axs[2].set(
                xlabel="Time", 
                ylabel="Profile Number",
            )
            axs[3].set(
                xlabel="Time", 
                ylabel=f"{self.cust_col}" if self.cust_col else "Custom",
            )   # <<< NEW

            fig.tight_layout()

            # Plot depth vs time, highlighting profile and non-profile points
            for data, col, label in zip(
                [profiles, not_profiles],
                ["tab:blue", "tab:red"],
                ["Profile", "Not Profile"],
            ):
                axs[0].plot(
                    data["TIME"],
                    -data[f"INTERP_{self.depth_col}"],
                    marker=".",
                    markersize=1,
                    ls="",
                    c=col,
                    label=label,
                )
                axs[1].plot(
                    data["TIME"],
                    data["smooth_grad"],
                    marker=".",
                    markersize=1,
                    ls="",
                    c=col,
                    label=label,
                )

            # Plot raw and smoothed gradients with threshold lines
            axs[1].plot(
                self.profile_outputs["TIME"],
                self.profile_outputs["grad"],
                c="k",
                alpha=0.1,
                label="Raw Velocity",
            )
            for val, label in zip(self.thresholds, ["Gradient Thresholds", None]):
                axs[1].axhline(val, ls="--", color="gray", label=label)

            # Plot profile numbers
            axs[2].plot(
                self.profile_outputs["TIME"],
                self.profile_outputs["profile_num"],
                c="gray",
            )

            for ax in axs[:2]:
                ax.legend(loc="upper right")
            plt.show(block=True)

            if self.cust_col and self.cust_col in self.profile_outputs.columns:   # <<< use self.cust_col
                axs[3].plot(
                    self.profile_outputs["TIME"],
                    self.profile_outputs[self.cust_col],   # <<< dynamic
                    c="purple",
                    marker=".",
                    markersize=1,
                    ls="",
                    label=self.cust_col
                )
                axs[3].legend(loc="upper right")

        root = tk.Tk()
        root.title("Parameter Adjustment")
        root.geometry(f"380x{50*len(self.parameters)}")
        entries = {}

        # Gradient thresholds
        row = 0
        values = self.thresholds
        tk.Label(root, text=f"Gradient Thresholds:").grid(row=row, column=0)
        for i, label, value in zip(range(2), ["+ve", "-ve"], values):
            tk.Label(root, text=f"{label}:").grid(row=row + 1, column=2 * i)
            entry = tk.Entry(root, textvariable=label, width=10)
            entry.insert(0, value)
            entry.grid(row=row + 1, column=2 * i + 1)
            entries[label] = entry

        # Filter window sizes
        row = 2
        values = self.win_sizes
        tk.Label(root, text=f"Filter Window Sizes:").grid(
            row=row, column=0, pady=(20, 0)
        )
        for i, label, value in zip(range(2), ["Median Filter", "Mean Filter"], values):
            tk.Label(root, text=f"{label}:").grid(row=row + 1, column=2 * i)
            entry = tk.Entry(root, textvariable=label, width=10)
            entry.insert(0, value)
            entry.grid(row=row + 1, column=2 * i + 1)
            entries[label] = entry

        # Depth column name
        row = 4
        value = self.depth_col
        tk.Label(root, text=f"Depth column name:").grid(row=row, column=0, pady=(20, 0))
        entry = tk.Entry(root, textvariable="depth_column")
        entry.insert(0, value)
        entry.grid(row=row, column=1, pady=(20, 0))
        entries["depth_column"] = entry

        # Function to handle Cancel button click
        def on_cancel():
            plt.close()
            root.destroy()

        def on_regenerate():
            # Update parameter attributes
            self.thresholds = [float(entries["+ve"].get()), float(entries["-ve"].get())]
            self.win_sizes = [
                entries["Median Filter"].get(),
                entries["Mean Filter"].get(),
            ]
            self.depth_col = entries["depth_column"].get()

            # Regenerate data and plot it
            plt.close()
            generate_plot()

        def on_save():
            self.log(
                f"continuing with parameters: \n"
                f"  Gradient Thresholds: {self.thresholds}\n"
                f"  Filter Window Sizes: {self.win_sizes}\n"
                f"  Depth column: {self.depth_col}\n"
            )
            plt.close()
            root.destroy()

        tk.Button(root, text="Regenerate", command=on_regenerate).grid(
            row=row + 1, column=0, pady=(20, 0)
        )
        tk.Button(root, text="Save", command=on_save).grid(
            row=row + 1, column=1, pady=(20, 0)
        )
        tk.Button(root, text="Cancel", command=on_cancel).grid(
            row=row + 1, column=2, pady=(20, 0)
        )

        generate_plot()
        return root