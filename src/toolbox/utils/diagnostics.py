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

"""
A module for diagnostic plotting and data summarization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from geopy.distance import geodesic
from toolbox.utils.time import safe_median_datetime, add_datetime_secondary_xaxis
from typing import Dict, List, Optional


def plot_time_series(
    data, x_var, y_var, title="Time Series Plot", xlabel=None, ylabel=None, **kwargs
):
    """Generates a time series plot for xarray data."""
    if isinstance(data, xr.Dataset):
        # Ensure that the variables exist in the xarray dataset
        if x_var not in data.coords or y_var not in data:
            raise ValueError(
                f"Variables {x_var} and {y_var} must exist in the dataset."
            )
        x_data = data[x_var].values  # Extract x_data (usually time dimension)
        y_data = data[y_var].values  # Extract the y_data (variable to plot)
    else:
        # Assuming custom format such as lists or arrays
        x_data, y_data = data[0], data[1]

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, **kwargs)
    plt.xlabel(xlabel or x_var)
    plt.ylabel(ylabel or y_var)
    plt.title(title)
    plt.show()


def plot_histogram(data, var, bins=30, title="Histogram", xlabel=None, **kwargs):
    """Generates a histogram for a given variable in xarray data."""
    if isinstance(data, xr.Dataset):
        # Ensure that the variable exists in the xarray dataset
        if var not in data:
            raise ValueError(f"Variable {var} must exist in the dataset.")
        data_to_plot = data[var].values
    else:
        # Handle custom data types like lists or arrays
        data_to_plot = data

    plt.figure(figsize=(10, 6))
    plt.hist(data_to_plot, bins=bins, alpha=0.7, **kwargs)
    plt.xlabel(xlabel or var)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def plot_boxplot(data, var, title="Box Plot", xlabel=None, **kwargs):
    """Generates a box plot for a given variable in xarray data."""
    if isinstance(data, xr.Dataset):
        # Ensure that the variable exists in the xarray dataset
        if var not in data:
            raise ValueError(f"Variable {var} must exist in the dataset.")
        data_to_plot = data[var].values
    else:
        # Handle custom data types like lists or arrays
        data_to_plot = data

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_to_plot, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel or var)
    plt.show()


def plot_correlation_matrix(data, variables=None, title="Correlation Matrix", **kwargs):
    """Generates a heatmap of the correlation matrix for xarray data."""
    if isinstance(data, xr.Dataset):
        if variables is None:
            variables = list(data.data_vars)  # Use all variables by default
        # Extract the variables to calculate the correlation matrix
        corr = data[variables].to_array().T.corr(dim="dim_0")
    else:
        raise TypeError("Data must be a Xarray Dataset to generate correlation matrix.")

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, **kwargs)
    plt.title(title)
    plt.show()


def generate_info(data):
    """Generate info for a given dataset"""
    if isinstance(data, xr.Dataset):
        # For xarray, we'll summarize each data variable
        print("Data Info:")
        print(data.info())
    else:
        print("Data Info only supported for xarray Dataset ")


def check_missing_values(data):
    """Check for missing values in the dataset."""
    if isinstance(data, xr.Dataset):
        missing = data.isnull().sum()
        print("Missing Values in Xarray Dataset:\n", missing)
    else:
        print("Missing value check only supported for xarray Dataset ")


#### General Diagnostics Functions ####


def summarising_profiles(ds: xr.Dataset, source_name: str) -> pd.DataFrame:
    """
    Summarise profiles from an xarray Dataset by computing medians of TIME, LATITUDE, LONGITUDE
    grouped by PROFILE_NUMBER. Handles datetime median safely using pandas.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with PROFILE_NUMBER as a coordinate.

    source_name : str
        Name of the glider/source to include in output.

    Returns
    -------
    pd.DataFrame
        Profile-level summary DataFrame.
    """
    if "PROFILE_NUMBER" not in ds:
        raise ValueError("Dataset must include PROFILE_NUMBER.")

    if "PROFILE_NUMBER" not in ds.coords:
        ds = ds.set_coords("PROFILE_NUMBER")

    summary_vars = [v for v in ["TIME", "LATITUDE", "LONGITUDE"] if v in ds]

    medians = {}
    for var in summary_vars:
        if var not in ds:
            continue

        da = ds[var]
        if "PROFILE_NUMBER" not in da.coords:
            da = da.set_coords("PROFILE_NUMBER")

        grouped = da.groupby("PROFILE_NUMBER")

        if np.issubdtype(da.dtype, np.datetime64):
            # Use pandas to compute median datetime safely
            medians[f"median_{var}"] = grouped.reduce(safe_median_datetime)
        else:
            medians[f"median_{var}"] = grouped.median(skipna=True)

    df = xr.Dataset(medians).to_dataframe().reset_index()
    df["glider_name"] = source_name
    df.rename(columns={"PROFILE_NUMBER": "PROFILE_NUMBER"}, inplace=True)
    # sort by time
    df.sort_values(by="median_TIME", inplace=True)
    # also add to the dataset

    return df


def find_closest_prof(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """
    For each profile in df_a, find the closest profile in df_b based on time,
    and calculate spatial distance to it.

    Parameters
    ----------
    df_a : pd.DataFrame
        Summary dataframe for glider A (reference).

    df_b : pd.DataFrame
        Summary dataframe for glider B (comparison).

    Returns
    -------
    pd.DataFrame
        df_a with additional columns:
            - closest_glider_b_profile
            - glider_b_time_diff
            - glider_b_distance_km
    """
    a_times = df_a["median_TIME"].values
    a_lats = df_a["median_LATITUDE"].values
    a_lons = df_a["median_LONGITUDE"].values

    b_times = df_b["median_TIME"].values
    b_lats = df_b["median_LATITUDE"].values
    b_lons = df_b["median_LONGITUDE"].values
    b_ids = df_b["PROFILE_NUMBER"].values

    closest_ids = []
    time_diffs = []
    distances = []

    for a_time, a_lat, a_lon in zip(a_times, a_lats, a_lons):
        time_diff = np.abs(b_times - a_time)
        idx = time_diff.argmin()

        closest_ids.append(b_ids[idx])
        time_diffs.append(time_diff[idx])

        if np.all(np.isfinite([a_lat, a_lon, b_lats[idx], b_lons[idx]])):
            dist_km = geodesic((a_lat, a_lon), (b_lats[idx], b_lons[idx])).km
        else:
            dist_km = np.nan
        distances.append(dist_km)

    df_result = df_a.copy()
    df_result["closest_glider_b_profile"] = closest_ids
    df_result["glider_b_time_diff"] = time_diffs
    df_result["glider_b_distance_km"] = distances

    return df_result


def plot_distance_time_grid(
    summaries: Dict[str, pd.DataFrame],
    output_path: str = None,
    show: bool = True,
    figsize: tuple = (16, 16),
):
    """
    Plot a grid of distance-over-time plots for all glider pair combinations.

    Parameters
    ----------
    summaries : dict
        Dictionary of {glider_name: pd.DataFrame} from summarising_profiles().

    output_path : str, optional
        If provided, the grid will be saved to this path.

    show : bool
        If True, plt.show() will be called.

    figsize : tuple
        Size of the full figure.
    """
    glider_names = list(summaries.keys())
    grid_size = len(glider_names)

    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=figsize, sharex=True, sharey=True
    )
    fig.suptitle("Distance Between Gliders Over Time", fontsize=18)
    combined_summaries = []

    for i, g_id in enumerate(glider_names):
        for j, g_b_id in enumerate(glider_names):
            if g_id == g_b_id:
                axes[i, j].set_title(f"{g_id} vs {g_b_id} (self-comparison)")
                if i != 0 or j != len(glider_names) - 1:
                    axes[i, j].axis("off")
                continue
            ref_df = summaries[g_id]
            comp_df = summaries[g_b_id]

            paired_df = find_closest_prof(ref_df, comp_df)
            # TODO: ------- Rename column headers and add glider name labels to PROFILE_NUMBER -------
            combined_summaries.append(paired_df)

            ax = axes[i, j]
            if paired_df.empty:
                ax.set_title(f"{g_id} vs {g_b_id}\n(no data)")
                ax.axis("off")
                continue

            for name, group in paired_df.groupby("glider_name"):
                ax.plot(
                    group["median_TIME"],
                    group["glider_b_distance_km"],
                    label=name,
                    marker="o",
                    linestyle="-",
                )

                # Rotate X tick labels
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")

            ax.set_title(f"{g_id} vs {g_b_id}")
            ax.grid(True)
            # add additional axis if top row or right column
            if i == 0:
                add_datetime_secondary_xaxis(ax)

            if j == grid_size - 1:
                ax.secondary_yaxis("right")

            if i == grid_size - 1:
                ax.set_xlabel("Datetime")
            if j == 0:
                ax.set_ylabel("Distance (km)")
            if i == j:
                ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path)
        print(f"[Diagnostics] Saved glider distance grid to: {output_path}")
    elif show:
        plt.show()
    else:
        plt.close()
    return pd.concat(combined_summaries, ignore_index=True)


def find_candidate_glider_pairs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    glider_a_name: str,
    glider_b_name: str,
    time_thresh_hr: float = 2.0,
    dist_thresh_km: float = 5.0,
) -> pd.DataFrame:
    """
    Vectorised version: match glider A profiles to glider B profiles within time and space thresholds.
    Returns one match per glider A profile (closest B match within threshold).
    """

    if df_a.empty or df_b.empty:
        return pd.DataFrame()

    # Ensure datetime format
    df_a["median_datetime"] = pd.to_datetime(df_a["median_TIME"])
    df_b["median_datetime"] = pd.to_datetime(df_b["median_TIME"])

    # Cartesian join: every profile A against every profile B
    df_a["_key"] = 1
    df_b["_key"] = 1
    df_cross = pd.merge(df_a, df_b, on="_key", suffixes=("_a", "_b")).drop(
        columns="_key"
    )

    # Time difference
    df_cross["time_diff_hr"] = (
        np.abs(
            (
                df_cross["median_datetime_a"] - df_cross["median_datetime_b"]
            ).dt.total_seconds()
        )
        / 3600.0
    )

    # Filter time threshold early
    df_cross = df_cross[df_cross["time_diff_hr"] <= time_thresh_hr]

    if df_cross.empty:
        return pd.DataFrame()

    # Vectorised geodesic distance using np.vectorize
    def compute_dist_km(lat_a, lon_a, lat_b, lon_b):
        if pd.isna(lat_a) or pd.isna(lon_a) or pd.isna(lat_b) or pd.isna(lon_b):
            return np.nan
        return geodesic((lat_a, lon_a), (lat_b, lon_b)).km

    dist_func = np.vectorize(compute_dist_km)

    df_cross["dist_km"] = dist_func(
        df_cross["median_LATITUDE_a"],
        df_cross["median_LONGITUDE_a"],
        df_cross["median_LATITUDE_b"],
        df_cross["median_LONGITUDE_b"],
    )

    # Filter by distance threshold TODO: ---------- SAVE THIS --------------
    df_cross = df_cross[df_cross["dist_km"] <= dist_thresh_km]

    if df_cross.empty:
        return pd.DataFrame()

    # Keep only best match (min dist) per PROFILE_NUMBER_a TODO: ------------ CHECK IF NECESSARY ------------
    best_matches = df_cross.loc[
        df_cross.groupby("PROFILE_NUMBER_a")["dist_km"].idxmin()
    ].copy()

    # Return clean structure
    best_matches = best_matches.rename(
        columns={
            "glider_name": "glider_a_name",
            "PROFILE_NUMBER_a": "glider_a_PROFILE_NUMBER",
            "PROFILE_NUMBER_b": "glider_b_PROFILE_NUMBER",
        }
    )

    best_matches["glider_a_name"] = glider_a_name
    best_matches["glider_b_name"] = glider_b_name

    return best_matches[
        [
            "glider_a_PROFILE_NUMBER",
            "glider_a_name",
            "glider_b_PROFILE_NUMBER",
            "glider_b_name",
            "time_diff_hr",
            "dist_km",
        ]
    ].reset_index(drop=True)


def plot_heatmap_glider_df(
    ax,
    matchup_df: pd.DataFrame,
    time_bins: np.ndarray,
    dist_bins: np.ndarray,
    glider_a_name: str,
    glider_b_name: str,
    i: int,
    j: int,
    grid_size: int,
):
    """
    Plot cumulative 2D histogram of time/distance matchups for a glider pair on a given axis.
    """
    if matchup_df.empty:
        ax.set_title(f"{glider_a_name} vs {glider_b_name} (no matches)")
        ax.axis("off")
        return

    H, xedges, yedges = np.histogram2d(
        matchup_df["time_diff_hr"], matchup_df["dist_km"], bins=[time_bins, dist_bins]
    )

    H_cum = H.cumsum(axis=0).cumsum(axis=1)
    X, Y = np.meshgrid(yedges, xedges)
    im = ax.pcolormesh(X, Y, H_cum, cmap="PuBu", shading="auto")
    # add additional axis if top row or right column
    if i == 0:
        secax = ax.secondary_xaxis("top")
        secax.set_xlabel("Distance Threshold (km)")
    if j == grid_size - 1:
        secax = ax.secondary_yaxis("right")
        secax.set_ylabel("Time Threshold (hr)")

    if i == grid_size - 1:
        ax.set_xlabel("Distance Threshold (km)")
    if j == 0:
        ax.set_ylabel("Time Threshold (hr)")
    ax.set_title(f"{glider_a_name} vs {glider_b_name}")

    # Annotate values
    for i in range(H_cum.shape[0]):
        for j in range(H_cum.shape[1]):
            val = int(H_cum[i, j])
            if val > 0:
                x_center = (yedges[j] + yedges[j + 1]) / 2
                y_center = (xedges[i] + xedges[i + 1]) / 2
                color = "white" if val > H_cum.max() / 2 else "black"
                ax.text(
                    x_center,
                    y_center,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )


def plot_glider_pair_heatmap_grid(
    summaries: Dict[str, pd.DataFrame],
    time_bins: np.ndarray,
    dist_bins: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (16, 16),
):
    """
    Generate an NxN grid of cumulative heatmaps for all glider pair combinations.
    """
    glider_names = list(summaries.keys())
    grid_size = len(glider_names)

    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=figsize, sharex=True, sharey=True
    )
    fig.suptitle("Heatmap of Matchups Between Gliders", fontsize=18)

    for i, g_a in enumerate(glider_names):
        for j, g_b in enumerate(glider_names):
            df_a = summaries[g_a]
            df_b = summaries[g_b]

            if g_a == g_b:
                axes[i, j].axis("off")
                continue
            ax = axes[i, j]

            matches = find_candidate_glider_pairs(
                df_a,
                df_b,
                glider_a_name=g_a,
                glider_b_name=g_b,
                time_thresh_hr=max(time_bins),
                dist_thresh_km=max(dist_bins),
            )

            plot_heatmap_glider_df(
                ax=ax,
                matchup_df=matches,
                time_bins=time_bins,
                dist_bins=dist_bins,
                glider_a_name=g_a,
                glider_b_name=g_b,
                i=i,
                j=j,
                grid_size=grid_size,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        plt.savefig(output_path)
        print(f"[Diagnostics] Saved glider heatmap grid to: {output_path}")
    elif show:
        plt.show()
    else:
        plt.close()
