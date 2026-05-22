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

import xarray as xr
import pandas as pd
import numpy as np
import warnings
import os
import datetime as _dt

from scipy.stats import pearsonr
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def interpolate_DEPTH(
    ds: xr.Dataset,
    bin_size: int = 5,
    depth_positive: str = "down",  # "down" => positive depth; "up" => negative depth
    plot: bool = False,
    plot_kind: str = "hist",  # "hist" or "heatmap"
    max_profiles_heatmap: int = 50,
    figsize=(9, 4),
    save_path: str | None = None,
    show: bool = True,
) -> xr.Dataset:
    """
    Interpolate missing DEPTH values per PROFILE_NUMBER, normalize depth sign,
    build regular depth bins, and (optionally) visualize the bin coverage.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain 'DEPTH', 'TIME', 'PROFILE_NUMBER' on the 'N_MEASUREMENTS' axis.
    bin_size : int
        Depth bin size in the same units as DEPTH (e.g., meters).
    depth_positive : {"down","up"}
        Normalize depth sign before binning. "down" => positive increasing with depth.
    plot : bool
        If True, draw a quick visualization of the resulting depth bins.
    plot_kind : {"hist","heatmap"}
        - "hist": overall histogram of measurement counts per bin (barh).
        - "heatmap": per-profile vs bin coverage (first `max_profiles_heatmap` profiles).
    max_profiles_heatmap : int
        Max number of profiles to include in the heatmap (to avoid huge plots).
    figsize : tuple
        Matplotlib figure size for the plot.
    save_path : str or None
        If provided, save the plot to this path.
    show : bool
        If True, plt.show() the figure; otherwise close it.

    Returns
    -------
    xr.Dataset
        With:
          - DEPTH_interpolated (float64, N_MEASUREMENTS)
          - DEPTH_bin (float32, N_MEASUREMENTS)
          - DEPTH_range (str, N_MEASUREMENTS) e.g. "50–55"
    """
    # Validate required variables
    required_vars = ["DEPTH", "TIME", "PROFILE_NUMBER"]
    for var in required_vars:
        if var not in ds:
            raise ValueError(f"Dataset must contain variable: {var}")

    if "N_MEASUREMENTS" not in ds.dims:
        raise ValueError(
            "Dataset must have a profile-wise measurement dimension 'N_MEASUREMENTS'."
        )
    if "PROFILE_NUMBER" not in ds:
        raise ValueError("Dataset must contain 'PROFILE_NUMBER'.")

    # # Filter out surfacing/invalid profiles (PROFILE_NUMBER > 0)
    # mask = ds["PROFILE_NUMBER"] > 0
    # ds = ds.sel(N_MEASUREMENTS=mask)
    #
    # # Ensure int dtype and set as coord
    # ds["PROFILE_NUMBER"] = ds["PROFILE_NUMBER"].astype(np.int32)
    if "PROFILE_NUMBER" not in ds.coords:
        ds = ds.set_coords("PROFILE_NUMBER")

    print("[Pipeline Manager] Interpolating missing DEPTH values by PROFILE_NUMBER...")

    # Interpolate DEPTH within each profile along N_MEASUREMENTS
    DEPTH_interp = (
        ds["DEPTH"]
        .groupby("PROFILE_NUMBER")
        .map(
            lambda g: g.interpolate_na(
                dim="N_MEASUREMENTS", method="linear", fill_value="extrapolate"
            ).reindex_like(g)
        )
    )
    ds = ds.assign({"DEPTH_interpolated": DEPTH_interp.astype("float64")})

    # --- Normalize depth sign BEFORE binning ---
    dep = ds["DEPTH_interpolated"]
    med = np.nanmedian(dep.values)
    if depth_positive == "down" and med < 0:
        dep = -dep
    elif depth_positive == "up" and med > 0:
        dep = -dep
    ds["DEPTH_interpolated"] = dep

    # --- Compute regular bins with the configured bin_size ---
    DEPTH_bin = (dep // bin_size) * bin_size
    ds["DEPTH_bin"] = DEPTH_bin.astype(np.float32)

    # Add a friendly range label per measurement (e.g., "50–55")
    ds["DEPTH_range"] = xr.apply_ufunc(
        lambda start, end: (
            f"{int(start)}–{int(end)}"
            if np.isfinite(start) and np.isfinite(end)
            else ""
        ),
        ds["DEPTH_bin"],
        ds["DEPTH_bin"] + bin_size,
        vectorize=True,
        dask="parallelized" if ds.chunks else False,
        output_dtypes=[str],
    )

    # --- Optional visualization of depth bins ---
    if plot:
        if plot_kind == "hist":
            # Overall measurement counts per bin
            bins_vals = ds["DEPTH_bin"].values
            bins_vals = bins_vals[np.isfinite(bins_vals)]
            if bins_vals.size:
                unique_bins, counts = np.unique(bins_vals, return_counts=True)
                fig, ax = plt.subplots(figsize=figsize)
                ax.barh(unique_bins.astype(float), counts, height=bin_size * 0.8)
                ax.set_xlabel("Measurement count")
                ax.set_ylabel("Depth bin")
                # Deeper at bottom if positive-down
                if depth_positive == "down":
                    ax.invert_yaxis()
                ax.set_title(f"Depth-bin coverage (bin={bin_size})")
                fig.tight_layout()
                if save_path:
                    plt.savefig(save_path, dpi=300)
                    print(f"[Diagnostics] Saved depth-bin histogram to: {save_path}")
                if show:
                    plt.show()
                else:
                    plt.close(fig)
            else:
                print("[Diagnostics] No finite DEPTH_bin values to plot (hist).")

        elif plot_kind == "heatmap":
            # Per-profile × depth-bin coverage (counts), limited to first N profiles
            profs = np.unique(ds["PROFILE_NUMBER"].values)
            if profs.size == 0:
                print("[Diagnostics] No profiles available for heatmap.")
            else:
                profs = profs[:max_profiles_heatmap]
                # Build a small 2D grid of counts
                df = (
                    xr.Dataset(
                        {
                            "PROFILE_NUMBER": (
                                "N_MEASUREMENTS",
                                ds["PROFILE_NUMBER"].values,
                            ),
                            "DEPTH_bin": ("N_MEASUREMENTS", ds["DEPTH_bin"].values),
                        }
                    )
                    .to_dataframe()
                    .dropna()
                )
                df = df[df["PROFILE_NUMBER"].isin(profs)]
                if df.empty:
                    print("[Diagnostics] No data to plot in heatmap (after filter).")
                else:
                    # pivot: rows=profile, cols=depth_bin, values=count
                    pivot = (
                        df.assign(count=1)
                        .groupby(["PROFILE_NUMBER", "DEPTH_bin"])["count"]
                        .sum()
                        .unstack(fill_value=0)
                    )
                    # Ensure numeric order of bins
                    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
                    fig, ax = plt.subplots(figsize=figsize)
                    im = ax.imshow(
                        pivot.values,
                        aspect="auto",
                        interpolation="nearest",
                        cmap="PuBu",
                    )
                    ax.set_yticks(range(len(pivot.index)))
                    ax.set_yticklabels(pivot.index.astype(int))
                    ax.set_xticks(range(len(pivot.columns)))
                    ax.set_xticklabels(pivot.columns.astype(int), rotation=90)
                    ax.set_xlabel("Depth bin")
                    ax.set_ylabel("Profile #")
                    ax.set_title(
                        f"Per-profile depth-bin coverage (first {len(pivot.index)} profiles)"
                    )
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label("Count")
                    fig.tight_layout()
                    if save_path:
                        plt.savefig(save_path, dpi=300)
                        print(f"[Diagnostics] Saved depth-bin heatmap to: {save_path}")
                    if show:
                        plt.show()
                    else:
                        plt.close(fig)
        else:
            print(f"[Diagnostics] Unknown plot_kind='{plot_kind}'. Skipping plot.")

    return ds


def aggregate_vars(
    ds: xr.Dataset, vars_to_aggregate, profile_dim="PROFILE_NUMBER", bin_dim="DEPTH_bin"
) -> xr.Dataset:
    """
    Compute medians per (PROFILE_NUMBER, DEPTH_bin) and return a NEW dataset
    with variables named 'median_{var}', each shaped (PROFILE_NUMBER, DEPTH_bin).

    Notes:
    - This does NOT attach medians back to the raw dataset (which would broadcast
      onto N_MEASUREMENTS). Use the returned dataset for downstream steps.
    - Requires that both PROFILE_NUMBER and DEPTH_bin are coordinates aligned
      to the measurement dimension (typically 'N_MEASUREMENTS').
    """
    # basic checks
    if "N_MEASUREMENTS" not in ds.dims:
        raise ValueError("Expected 'N_MEASUREMENTS' dimension in input dataset.")
    if profile_dim not in ds:
        raise ValueError(f"Dataset must contain '{profile_dim}'.")
    if bin_dim not in ds:
        raise ValueError(f"Dataset must contain '{bin_dim}'.")

    # ensure the two keys are *coordinates on the measurement axis*
    # (i.e., both should be 1-D arrays aligned to N_MEASUREMENTS)
    for key in (profile_dim, bin_dim):
        if key not in ds.coords:
            ds = ds.set_coords(key)
        # sanity: the coord should broadcast to N_MEASUREMENTS (usually same length)
        if (
            ds[key].sizes.get("N_MEASUREMENTS", None) is None
            and ds[key].sizes != ds["N_MEASUREMENTS"].sizes
        ):
            raise ValueError(
                f"Coordinate '{key}' must align with 'N_MEASUREMENTS' for aggregation."
            )

    out_vars = {}
    for var in vars_to_aggregate:
        if var not in ds:
            print(f"[aggregate_vars] Skipping missing variable: {var}")
            continue

        da = ds[var]
        # Build a MultiIndex on the measurement axis: (PROFILE_NUMBER, DEPTH_bin)
        da_mi = da.set_index(N_MEASUREMENTS=(profile_dim, bin_dim))

        # Median within each (profile, bin) group across any duplicate measurements
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            da_groupmed = da_mi.groupby("N_MEASUREMENTS").median(skipna=True)

        # Unstack back to a dense 2D grid -> dims: (PROFILE_NUMBER, DEPTH_bin)
        da_2d = da_groupmed.unstack("N_MEASUREMENTS")

        # Ensure nice dim order
        if (profile_dim, bin_dim) not in [da_2d.dims, da_2d.dims[::-1]]:
            da_2d = da_2d.transpose(profile_dim, bin_dim)

        out_vars[f"median_{var}"] = da_2d

    if not out_vars:
        raise ValueError("No variables were aggregated (none found).")

    agg = xr.Dataset(out_vars)

    # (optional) keep PROFILE_NUMBER & DEPTH_bin as coords on the result
    for key in (profile_dim, bin_dim):
        if key in ds.coords and key not in agg.coords:
            agg = agg.assign_coords(
                {key: ds.coords[key] if key in ds.coords else agg[key]}
            )

    # Drop bins that are NaN for every variable
    all_meds = xr.concat([agg[v] for v in agg.data_vars], dim="__vars__")
    # drop PROFILE_NUMBERs with all-NaN across all vars/bins
    mask_prof = all_meds.isnull().all(dim=["__vars__", bin_dim])
    if mask_prof.any():
        agg = agg.sel({profile_dim: ~mask_prof})
    # drop DEPTH_bin with all-NaN across all vars/profiles
    mask_bin = all_meds.isnull().all(dim=["__vars__", profile_dim])
    if mask_bin.any():
        agg = agg.sel({bin_dim: ~mask_bin})

    return agg


def filter_xarray_by_profile_ids(
    ds: xr.Dataset,
    profile_id_var: str,
    valid_ids: np.ndarray | list,
) -> xr.Dataset:
    """
    Filter an xarray.Dataset to keep only rows / entries with profile IDs in `valid_ids`.
    Works for both raw (with N_MEASUREMENTS) and aggregated (PROFILE_NUMBER as a dim) datasets.
    """
    if (
        profile_id_var not in ds
        and profile_id_var not in ds.coords
        and profile_id_var not in ds.dims
    ):
        raise KeyError(
            f"'{profile_id_var}' not found as a variable/coord/dim in dataset."
        )

    # Normalize ids to dataset coord dtype (avoid float->int mismatches from pandas)
    if profile_id_var in ds.coords:
        coord_dtype = ds[profile_id_var].dtype
    elif profile_id_var in ds.dims:
        coord_dtype = ds[
            profile_id_var
        ].dtype  # xarray exposes dim coord dtype via coords of same name if present
    else:
        # fallback if it's only a data_var
        coord_dtype = ds[profile_id_var].dtype

    ids = np.asarray(valid_ids).astype(coord_dtype, copy=False)

    # Keep only IDs that actually exist (prevents KeyError on .sel)
    present = np.intersect1d(ds[profile_id_var].values, ids)
    print(
        f"[Filter] Aggregated case: {len(ds[profile_id_var])} → {len(present)} profiles retained."
    )

    if present.size == 0:
        # Return empty along the profile dimension, preserving structure
        return ds.isel({profile_id_var: slice(0, 0)})

    # Select profiles (order follows `present`;
    filtered = ds.sel({profile_id_var: present})
    print(f"[Filter] Resulting dims: {filtered.dims}")
    return filtered


def find_profile_pair_metadata(
    df_target: pd.DataFrame,
    df_ancillary: pd.DataFrame,
    target_name: str,
    ancillary_name: str,
    time_thresh_hr: float = 2.0,
    dist_thresh_km: float = 5.0,
) -> pd.DataFrame:
    """
    Identify profile pairs between a target and ancillary glider within time/distance thresholds.

    Parameters
    ----------
    df_target : pd.DataFrame
        Summary dataframe for the target glider (from `summarising_profiles()`).

    df_ancillary : pd.DataFrame
        Summary dataframe for the ancillary glider.

    target_name : str
        Name of the target glider (used in output column names).

    ancillary_name : str
        Name of the ancillary glider (used in output column names).

    Returns
    -------
    pd.DataFrame
        Matched profile pairs with columns:
        - [target_name]_PROFILE_NUMBER
        - [ancillary_name]_PROFILE_NUMBER
        - time_diff_hr
        - dist_km
    """
    if df_target.empty or df_ancillary.empty:
        return pd.DataFrame()

    df_target = df_target.copy()
    df_ancillary = df_ancillary.copy()

    # Parse datetime columns
    df_target["median_datetime"] = pd.to_datetime(df_target["median_TIME"])
    df_ancillary["median_datetime"] = pd.to_datetime(df_ancillary["median_TIME"])

    # Cartesian join
    df_target["_key"] = 1
    df_ancillary["_key"] = 1
    df_cross = pd.merge(
        df_target, df_ancillary, on="_key", suffixes=("_target", "_ancillary")
    ).drop("_key", axis=1)

    # Compute time difference (in hours)
    df_cross["time_diff_hr"] = (
        np.abs(
            df_cross["median_datetime_target"] - df_cross["median_datetime_ancillary"]
        ).dt.total_seconds()
        / 3600.0
    )

    df_cross = df_cross[df_cross["time_diff_hr"] <= time_thresh_hr]
    if df_cross.empty:
        return pd.DataFrame()

    # Geodesic distance
    def compute_dist_km(lat1, lon1, lat2, lon2):
        if pd.isnull(lat1) or pd.isnull(lon1) or pd.isnull(lat2) or pd.isnull(lon2):
            return np.nan
        return geodesic((lat1, lon1), (lat2, lon2)).km

    df_cross["dist_km"] = np.vectorize(compute_dist_km)(
        df_cross["median_LATITUDE_target"],
        df_cross["median_LONGITUDE_target"],
        df_cross["median_LATITUDE_ancillary"],
        df_cross["median_LONGITUDE_ancillary"],
    )

    df_cross = df_cross[df_cross["dist_km"] <= dist_thresh_km]
    if df_cross.empty:
        return pd.DataFrame()

    # Get best match per target profile
    best_matches = df_cross.loc[
        df_cross.groupby("PROFILE_NUMBER_target")["dist_km"].idxmin()
    ].copy()

    # Rename columns to final format
    best_matches.rename(
        columns={
            "PROFILE_NUMBER_target": f"{target_name}_PROFILE_NUMBER",
            "PROFILE_NUMBER_ancillary": f"{ancillary_name}_PROFILE_NUMBER",
        },
        inplace=True,
    )

    return best_matches[
        [
            f"{target_name}_PROFILE_NUMBER",
            f"{ancillary_name}_PROFILE_NUMBER",
            "time_diff_hr",
            "dist_km",
        ]
    ].reset_index(drop=True)


def merge_pairs_from_filtered_aggregates(
    paired_df,
    agg_target: xr.Dataset,
    agg_anc: xr.Dataset,
    target_name: str,
    ancillary_name: str,
    variables,  # e.g. ["TEMP","CNDC","CHLA",...]
    bin_dim: str = "DEPTH_bin",
    pair_dim: str = "PAIR_INDEX",
) -> xr.Dataset:
    """
    Build a dataset with one row per (target_profile, ancillary_profile, depth_bin).
    Each row has: target/ancillary profile IDs, time/distance diffs, and median_{var} for both sides.
    """
    tgt_id_col = f"{target_name}_PROFILE_NUMBER"
    anc_id_col = f"{ancillary_name}_PROFILE_NUMBER"

    # Make sure PROFILE_NUMBER is the dimension we select on
    if "PROFILE_NUMBER" not in agg_target.dims or "PROFILE_NUMBER" not in agg_anc.dims:
        raise ValueError("Expected aggregated datasets with dim 'PROFILE_NUMBER'.")

    # pre-cast ids for safe selection
    agg_target = agg_target.assign_coords(PROFILE_NUMBER=agg_target.PROFILE_NUMBER)
    agg_anc = agg_anc.assign_coords(PROFILE_NUMBER=agg_anc.PROFILE_NUMBER)

    out = []

    # restrict to just the columns we need
    pairs = paired_df[[tgt_id_col, anc_id_col, "time_diff_hr", "dist_km"]]

    for pair_idx, row in pairs.reset_index(drop=True).iterrows():
        pid_t = row[tgt_id_col]
        pid_a = row[anc_id_col]

        # skip if either profile missing
        if (
            pid_t not in agg_target["PROFILE_NUMBER"].values
            or pid_a not in agg_anc["PROFILE_NUMBER"].values
        ):
            continue

        # select per-profile slabs (dims: DEPTH_bin)
        t = agg_target.sel(PROFILE_NUMBER=pid_t)
        a = agg_anc.sel(PROFILE_NUMBER=pid_a)

        # keep only requested median variables that exist
        t_keep = [f"median_{v}" for v in variables if f"median_{v}" in t]
        a_keep = [f"median_{v}" for v in variables if f"median_{v}" in a]
        if not t_keep or not a_keep:
            continue
        t = t[t_keep]
        a = a[a_keep]

        # align on common depth bins
        common_bins = np.intersect1d(t[bin_dim].values, a[bin_dim].values)
        if common_bins.size == 0:
            continue
        t = t.sel({bin_dim: common_bins})
        a = a.sel({bin_dim: common_bins})

        # suffix variable names
        t = t.rename({v: f"{v}_TARGET_{target_name}" for v in t.data_vars})
        a = a.rename({v: f"{v}_{ancillary_name}" for v in a.data_vars})

        # merge the two sides (only bin_dim left)
        pair = xr.merge(
            [t, a], join="exact", compat="override", combine_attrs="override"
        )

        # annotate pair metadata on a new pair dimension
        pair = pair.expand_dims({pair_dim: [pair_idx]})
        pair[f"TARGET_{target_name}_PROFILE_NUMBER"] = (pair_dim, [pid_t])
        pair[f"{ancillary_name}_PROFILE_NUMBER"] = (pair_dim, [pid_a])
        pair["time_diff_hr"] = (pair_dim, [float(row["time_diff_hr"])])
        pair["dist_km"] = (pair_dim, [float(row["dist_km"])])

        out.append(pair)

    if not out:
        raise ValueError("No valid pairs produced during merge.")

    return xr.concat(out, dim=pair_dim)


def major_axis_r2_xr(x: xr.DataArray, y: xr.DataArray) -> float:
    """
    Compute R² (coefficient of determination) for Major Axis (Type II) regression using xarray.

    Parameters
    ----------
    x : xr.DataArray
        First variable (e.g., target glider).

    y : xr.DataArray
        Second variable (e.g., ancillary glider).

    Returns
    -------
    float
        R² value, or NaN if fewer than 2 valid observations.
    """
    if not isinstance(x, xr.DataArray) or not isinstance(y, xr.DataArray):
        raise TypeError("Inputs must be xarray.DataArray.")

    # Apply masking
    mask = ~xr.ufuncs.isnan(x) & ~xr.ufuncs.isnan(y)
    x_clean = x.where(mask, drop=True)
    y_clean = y.where(mask, drop=True)

    if x_clean.size < 2 or y_clean.size < 2:
        return np.nan

    r, _ = pearsonr(x_clean.values, y_clean.values)
    return r**2


def compute_r2_for_merged_profiles_xr(
    ds: xr.Dataset,
    variables: list[str],
    target_name: str,
    ancillary_name: str,
) -> xr.Dataset:
    """
    Compute R² for each profile-pair in an xarray.Dataset, and append the results directly to the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with one row per (PAIR_INDEX, DEPTH_bin), and aligned variables for target and ancillary gliders.

    variables : list of str
        List of variable base names (e.g., ["salinity", "temperature"]).

    target_name : str
        Name of the target glider (used in suffix suffix: _TARGET_{name}).

    ancillary_name : str
        Name of the ancillary glider (used in suffix: _{name}).

    Returns
    -------
    xr.Dataset
        Same dataset with new variables: r2_{var}_{ancillary_name}, one per profile pair.
        These will be aligned with the "PAIR_INDEX" dimension only.
    """

    pair_index = ds["PAIR_INDEX"].values
    results = {}

    for var in variables:
        target_var = f"median_{var}_TARGET_{target_name}"
        anc_var = f"median_{var}_{ancillary_name}"

        if target_var not in ds or anc_var not in ds:
            print(f"[R²] Skipping variable '{var}' — missing data.")
            continue

        r2_values = []
        for pid in pair_index:
            tgt = ds[target_var].sel(PAIR_INDEX=pid)
            anc = ds[anc_var].sel(PAIR_INDEX=pid)

            x = tgt.values
            y = anc.values

            # Remove NaNs
            mask = ~np.isnan(x) & ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) < 2:
                r2 = np.nan
            else:
                r, _ = pearsonr(x_clean, y_clean)
                r2 = r**2

            r2_values.append(r2)

        results[f"r2_{var}_{ancillary_name}"] = xr.DataArray(
            r2_values, dims="PAIR_INDEX", coords={"PAIR_INDEX": pair_index}
        )

    # Attach R² variables to original dataset
    ds = ds.assign(results)
    return ds


def plot_r2_heatmaps_per_pair(
    r2_datasets,
    variables,
    target_name=None,  # optional, used for plot titles/files
    r2_thresholds=[0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    time_thresh_hr=None,  # e.g. 5 to filter by time ≤ 5 hr
    dist_thresh_km=None,  # e.g. 10 to filter by dist ≤ 10 km
    figsize=(9, 6),
    save_plots=False,
    output_path=None,  # directory to save plots if save_plots is True
    show_plots=True,
):
    """
    Create ONE heatmap per ancillary pairing showing counts of unique ancillary profiles
    that meet R² thresholds for each variable. Optionally filter by time/dist thresholds.
    """
    var_labels = variables

    for ancillary_name, ds in r2_datasets.items():
        if not isinstance(ds, xr.Dataset):
            print(f"[Plot] Skipping '{ancillary_name}': not an xarray Dataset.")
            continue

        # Build a dataframe with the needed columns
        r2_cols = [f"r2_{v}_{ancillary_name}" for v in variables]
        needed = []
        for c in r2_cols:
            if c in ds:
                needed.append(c)
            else:
                print(f"[Plot] Warning: missing {c} in dataset for {ancillary_name}")

        # We’ll also try to include the ancillary profile id + optional filters
        anc_prof_col = f"{ancillary_name}_PROFILE_NUMBER"
        extra = []
        if anc_prof_col in ds:
            extra.append(anc_prof_col)
        else:
            print(
                f"[Plot] Warning: missing {anc_prof_col} in dataset for {ancillary_name}"
            )

        for meta in ["time_diff_hr", "dist_km"]:
            if meta in ds:
                extra.append(meta)

        if not needed:
            print(f"[Plot] No R² columns present for '{ancillary_name}'. Skipping.")
            continue

        df = ds[needed + extra].to_dataframe().reset_index()

        # Optional filtering by time/distance (pair-level)
        if time_thresh_hr is not None and "time_diff_hr" in df.columns:
            df = df[df["time_diff_hr"] <= float(time_thresh_hr)]
        if dist_thresh_km is not None and "dist_km" in df.columns:
            df = df[df["dist_km"] <= float(dist_thresh_km)]

        # Build heatmap matrix: rows = variables, cols = thresholds
        heatmap = np.zeros((len(variables), len(r2_thresholds)), dtype=int)

        for i, var in enumerate(variables):
            col = f"r2_{var}_{ancillary_name}"
            if col not in df.columns:
                continue
            for j, thr in enumerate(r2_thresholds):
                mask = df[col] >= thr
                if anc_prof_col in df.columns:
                    # Count unique ancillary profiles that pass the threshold
                    count = df.loc[mask, anc_prof_col].nunique()
                else:
                    # Fallback: count number of pairs
                    count = int(mask.sum())
                heatmap[i, j] = count

        # Plot single heatmap for this pairing
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(heatmap, aspect="auto", cmap="PuBu")

        ax.set_xticks(np.arange(len(r2_thresholds)))
        ax.set_xticklabels([f"{t:.2f}" for t in r2_thresholds], rotation=45, fontsize=9)
        ax.set_yticks(np.arange(len(variables)))
        ax.set_yticklabels(var_labels, fontsize=10)

        title_target = target_name if target_name else "Target"
        subtitle = []
        if time_thresh_hr is not None:
            subtitle.append(f"Time ≤ {time_thresh_hr}hr")
        if dist_thresh_km is not None:
            subtitle.append(f"Dist ≤ {dist_thresh_km}km")
        sub = " | ".join(subtitle)
        ax.set_title(
            f"R² Threshold Coverage: {title_target} vs {ancillary_name}\n{sub}",
            fontsize=12,
        )

        # Annotate cells
        for i in range(len(variables)):
            for j in range(len(r2_thresholds)):
                val = int(heatmap[i, j])
                value_midpoint = (heatmap.max() - heatmap.min()) / 2.0 + heatmap.min()
                color = "white" if val > value_midpoint else "black"

                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Unique ancillary profiles", rotation=90)

        plt.tight_layout()

        # Save or show
        if save_plots:
            # if output_path is a file, use its directory
            if output_path is not None and not os.path.isdir(output_path):
                output_path = os.path.dirname(output_path)
            if output_path is None or output_path == "":
                output_path = "."
            # ensure directory exists
            os.makedirs(output_path, exist_ok=True)
            fname = f"r2_heatmap_{title_target}_vs_{ancillary_name}.png"
            path = os.path.join(output_path, fname)
            plt.savefig(path, dpi=300)
            print(f"[Plot] Saved: {path}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)


def collect_xy_from_r2_ds(
    r2_ds,
    var,
    target_name,
    ancillary_name,
    r2_min=None,
    time_max=None,
    dist_max=None,
):
    """
    From an R² dataset (one ancillary), collect all (x=ancillary, y=target) binned points
    for a given variable across selected profile pairs, flattening PAIR_INDEX×DEPTH_bin.

    Returns: x, y (1D np.ndarrays)
    """
    y_name = f"median_{var}_TARGET_{target_name}"
    x_name = f"median_{var}_{ancillary_name}"
    r2_name = f"r2_{var}_{ancillary_name}"

    # Sanity checks
    for need in [x_name, y_name]:
        if need not in r2_ds:
            return np.array([]), np.array([])

    # Build a boolean mask over PAIR_INDEX
    n_pairs = r2_ds.sizes.get("PAIR_INDEX", 0)
    if n_pairs == 0:
        return np.array([]), np.array([])

    mask = xr.DataArray(
        np.ones(n_pairs, dtype=bool),
        dims=["PAIR_INDEX"],
        coords={"PAIR_INDEX": r2_ds["PAIR_INDEX"]},
    )

    if (r2_min is not None) and (r2_name in r2_ds):
        mask = mask & (r2_ds[r2_name] >= float(r2_min))

    if (time_max is not None) and ("time_diff_hr" in r2_ds):
        mask = mask & (r2_ds["time_diff_hr"] <= float(time_max))

    if (dist_max is not None) and ("dist_km" in r2_ds):
        mask = mask & (r2_ds["dist_km"] <= float(dist_max))

    # Select pairs and flatten PAIR_INDEX×DEPTH_bin to 1D
    X = r2_ds[x_name].where(mask, drop=True).values.reshape(-1)
    Y = r2_ds[y_name].where(mask, drop=True).values.reshape(-1)

    m = np.isfinite(X) & np.isfinite(Y)
    return X[m], Y[m]


def fit_linear_map(x, y):
    """
    Fit y = slope * x + intercept using sklearn LinearRegression.
    Returns dict with slope, intercept, R² (model score), and n.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(x.size)}
    X = x.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(model.score(X, y)),
        "n": int(x.size),
    }


def plot_pair_scatter_grid(
    r2_datasets,
    variables,
    target_name,
    variable_r2_criteria=None,
    max_time_hr=None,
    max_dist_km=None,
    ancillaries_order=None,
    figsize=None,
    point_alpha=0.6,
    point_size=8,
    equal_axes=True,  # new: keep x/y scales identical per panel
):
    """
    Grid of scatter plots: rows = ancillary, cols = variables.
    Each panel: ancillary median_{var} vs target median_{var}, all depth bins from
    pairs that pass R²/time/distance filters. Plots 1:1 and fitted line.
    """
    ancillaries = (
        list(ancillaries_order) if ancillaries_order else list(r2_datasets.keys())
    )
    n_rows, n_cols = len(ancillaries), len(variables)
    if figsize is None:
        figsize = (4.0 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharex=False, sharey=False
    )
    fig.suptitle(
        f"Ancillary vs Target ({target_name}) — depth-aligned medians with linear fits",
        fontsize=14,
    )

    # Simple type probe for guard
    _any_ds = next(iter(r2_datasets.values())) if r2_datasets else None
    _ds_type = type(_any_ds)

    for i, anc in enumerate(ancillaries):
        if anc not in r2_datasets or not isinstance(r2_datasets[anc], _ds_type):
            for j in range(n_cols):
                ax = axes[i, j]
                ax.axis("off")
                ax.set_title(f"{anc} (no data)")
            continue

        ds = r2_datasets[anc]
        pair_mask = np.ones(ds.sizes.get("PAIR_INDEX", 0), dtype=bool)
        if "time_diff_hr" in ds and max_time_hr is not None:
            pair_mask &= ds["time_diff_hr"].values <= max_time_hr
        if "dist_km" in ds and max_dist_km is not None:
            pair_mask &= ds["dist_km"].values <= max_dist_km
        all_pairs = (
            ds["PAIR_INDEX"].values
            if "PAIR_INDEX" in ds.coords
            else np.array([], dtype=int)
        )

        for j, var in enumerate(variables):
            ax = axes[i, j]
            anc_var = f"median_{var}_{anc}"
            tgt_var = f"median_{var}_TARGET_{target_name}"
            r2_var = f"r2_{var}_{anc}"

            if anc_var not in ds or tgt_var not in ds or r2_var not in ds:
                ax.axis("off")
                ax.set_title(f"{anc} — {var} (missing)")
                continue

            # Per-variable R² threshold
            r2_thresh = (
                None
                if variable_r2_criteria is None
                else variable_r2_criteria.get(var, None)
            )
            r2_mask = np.ones_like(pair_mask, dtype=bool)
            if r2_thresh is not None:
                r2_mask &= ds[r2_var].values >= r2_thresh

            valid_pairs = all_pairs[pair_mask & r2_mask]
            if valid_pairs.size == 0:
                ax.axis("off")
                ax.set_title(f"{anc} — {var} (no pairs)")
                continue

            # x: ancillary, y: target
            x = ds[anc_var].sel(PAIR_INDEX=valid_pairs).values.ravel()
            y = ds[tgt_var].sel(PAIR_INDEX=valid_pairs).values.ravel()

            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            n = x.size
            if n == 0:
                ax.axis("off")
                ax.set_title(f"{anc} — {var} (no valid points)")
                continue

            ax.scatter(x, y, s=point_size, alpha=point_alpha, edgecolor="k")

            # 1:1 line & limits
            lo = min(np.nanmin(x), np.nanmin(y))
            hi = max(np.nanmax(x), np.nanmax(y))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                if equal_axes:
                    ax.set_xlim(lo, hi)
                    ax.set_ylim(lo, hi)
                ax.plot([lo, hi], [lo, hi], linestyle="--", color="red")

            # Fit y = a*x + b
            eq = f"N={n} (no fit)"
            if n >= 2 and np.any(x != x[0]):
                model = LinearRegression()
                model.fit(x.reshape(-1, 1), y)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
                r2 = float(model.score(x.reshape(-1, 1), y))
                xline = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
                yline = slope * xline + intercept
                ax.plot(xline, yline, color="green", linewidth=1.5)
                eq = f"y = {slope:.2f}x + {intercept:.2f}\nFitted R²={r2:.2f}, N={n}"

            # Labels & titles
            if i == 0:
                ax.set_title(var)
            if j == 0:
                ax.set_ylabel(f"{target_name} {var}")  # target on Y
            ax.set_xlabel(f"{anc} {var}")  # ancillary on X
            ax.grid(True, alpha=0.3)
            ax.text(
                0.02,
                0.98,
                eq,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.8", alpha=0.8),
            )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


def apply_linear_map_to_da(da, slope, intercept, out_name=None):
    """
    Apply y = slope * da + intercept to an xarray.DataArray.
    Returns a new DataArray (optionally renamed).
    """
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return da * np.nan
    out = slope * da + intercept
    if out_name:
        out = out.rename(out_name)
    return out
