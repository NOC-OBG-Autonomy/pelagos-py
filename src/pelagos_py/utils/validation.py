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

# validation.py

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from pelagos_py.utils.diagnostics import summarising_profiles

from pelagos_py.utils.alignment import (
    interpolate_DEPTH,
    aggregate_vars,
    find_profile_pair_metadata,
    merge_pairs_from_filtered_aggregates,
    compute_r2_for_merged_profiles_xr,
    plot_r2_heatmaps_per_pair,
    filter_xarray_by_profile_ids,
)

from testing.sandbox import target_ds_raw


def load_device_folder_to_xarray(
    path_or_glob,
    alias_map=None,
    depth_candidates=("DEPTH", "depth"),
    time_candidates=("TIME", "time", "DateTime", "datetime"),
    lat_candidates=("LATITUDE", "latitude", "lat"),
    lon_candidates=("LONGITUDE", "longitude", "lon"),
    profile_start_index=1,
):
    """
    Read many device NetCDF files and combine into one xarray.Dataset with:
      - dim: N_MEASUREMENTS
      - vars/coords: PROFILE_NUMBER (int), DEPTH, TIME, LATITUDE, LONGITUDE, + data cols
    """
    if os.path.isdir(path_or_glob):
        files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(path_or_glob)
            for f in fs
            if f.endswith((".nc", ".nc4", ".cdf", ".netcdf"))
        ]
    else:
        files = [
            f
            for f in glob.glob(path_or_glob)
            if f.endswith((".nc", ".nc4", ".cdf", ".netcdf"))
        ]

    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for '{path_or_glob}'")

    def _pick(ds, cands):  # choose first available var name
        for c in cands:
            if c in ds:
                return c
        return None

    rows = []
    pid = int(profile_start_index)

    for fp in files:
        ds = xr.open_dataset(fp, decode_times=True)

        if alias_map:
            to_rename = {k: v for k, v in alias_map.items() if k in ds}
            if to_rename:
                ds = ds.rename(to_rename)

        depth_col = _pick(ds, depth_candidates) or "DEPTH"
        time_col = _pick(ds, time_candidates)
        lat_col = _pick(ds, lat_candidates)
        lon_col = _pick(ds, lon_candidates)

        df = ds.to_dataframe().reset_index()

        n = len(df)
        if time_col is None and "TIME" not in df.columns:
            # try attributes or 0-D variables
            t = ds.attrs.get("time") or ds.attrs.get("TIME")
            if t is not None:
                df["TIME"] = np.repeat(pd.to_datetime(t), n)
        elif time_col is not None:
            df["TIME"] = pd.to_datetime(df[time_col], errors="coerce")

        if lat_col is not None and "LATITUDE" not in df.columns:
            df["LATITUDE"] = df[lat_col]
        if lon_col is not None and "LONGITUDE" not in df.columns:
            df["LONGITUDE"] = df[lon_col]
        if depth_col != "DEPTH" and "DEPTH" not in df.columns:
            df["DEPTH"] = pd.to_numeric(df[depth_col], errors="coerce")

        df["PROFILE_NUMBER"] = pid
        rows.append(df)
        pid += 1
        ds.close()

    big = pd.concat(rows, ignore_index=True)

    ds_out = xr.Dataset(
        {c: (("N_MEASUREMENTS",), big[c].to_numpy()) for c in big.columns},
        coords={"N_MEASUREMENTS": np.arange(len(big))},
    )
    ds_out["PROFILE_NUMBER"] = ds_out["PROFILE_NUMBER"].astype("int32")
    if "PROFILE_NUMBER" not in ds_out.coords:
        ds_out = ds_out.set_coords("PROFILE_NUMBER")

    if "TIME" in ds_out:
        ds_out["TIME"] = xr.DataArray(
            pd.to_datetime(big["TIME"]).to_numpy(), dims=("N_MEASUREMENTS",)
        )
    for k in ("LATITUDE", "LONGITUDE", "DEPTH"):
        if k in ds_out:
            ds_out[k] = ds_out[k].astype("float64")

    print(
        f"[Device] Loaded {len(files)} files → {ds_out.sizes['N_MEASUREMENTS']} rows, "
        f"{len(np.unique(ds_out['PROFILE_NUMBER'].values))} profiles."
    )
    # output variables
    print(
        f"Variables: {', '.join([v for v in ds_out.data_vars if v != 'PROFILE_NUMBER'])}"
    )
    return ds_out


def validate(pmanager, target="None"):
    """
    End-to-end validation using settings.validation:
      - load device NetCDFs
      - summarise & pair profiles
      - (re)use cached target medians if available; otherwise interpolate+aggregate once
      - interpolate, bin, aggregate device (2-D medians)
      - merge per-pair on depth bins
      - compute per-pair R²
      - plot heatmaps per variable using plot_r2_heatmaps_per_pair
    """
    # --- config ---
    vcfg = pmanager.settings.get("validation", {}) or {}
    device_name = vcfg.get("device_name", "DEVICE")
    variables = vcfg.get("variable_names", list(pmanager.alignment_map.keys()))
    folder_path = vcfg.get("folder_path", "")
    max_time_hr = vcfg.get("max_time_threshold", 12)
    max_dist_km = vcfg.get("max_distance_threshold", 10)
    var_r2_criteria = vcfg.get("variable_r2_criteria", {})
    save_plots = bool(vcfg.get("save_plots", False))
    show_plots = bool(vcfg.get("show_plots", True))
    plot_output_path = vcfg.get("plot_output_path", "")
    apply_and_save = bool(vcfg.get("apply_and_save", False))
    out_path = vcfg.get("output_path", "")

    # ---- Target: prefer cached aggregated medians from preview_alignment() ----
    # Check the target(s) exist
    if type(target) is not list:
        target_name = target
        target = [target]
    else:
        target_name = "_".join(target)
    for platform in target:
        if platform not in pmanager.pipelines or platform not in pmanager._contexts:
            raise ValueError(f"Target '{platform}' not available.")

    # Make merged dataset from all of the targets
    to_merge = []
    for platform in target:
        # Append the target name to the profile number
        raw_ds = pmanager._contexts[platform]["data"][
            ["PROFILE_NUMBER", "DEPTH", "TIME", "LATITUDE", "LONGITUDE"] + variables
        ]
        raw_ds["PROFILE_NUMBER"] = (
            ("N_MEASUREMENTS",),
            raw_ds["PROFILE_NUMBER"].values.astype("str") + f"_{platform}",
        )

        # Remap the variable names if specified
        rename_map = {
            alias: std
            for std, alias_map in pmanager.alignment_map.items()
            if (alias := alias_map.get(platform)) and alias in raw_ds
        }
        raw_ds.rename(rename_map)

        to_merge.append(raw_ds)

    target_ds_raw = to_merge[0]
    target_ds_raw = target_ds_raw.assign_coords(
        {"N_MEASUREMENTS": target_ds_raw["N_MEASUREMENTS"]}
    )
    if len(to_merge) > 1:
        for ds in to_merge[1:]:
            offset = len(target_ds_raw["N_MEASUREMENTS"])
            ds = ds.assign_coords(N_MEASUREMENTS=ds["N_MEASUREMENTS"] + offset)
            target_ds_raw = xr.concat([target_ds_raw, ds], dim="N_MEASUREMENTS")

    # Create caches if not present
    if not hasattr(pmanager, "processed_per_glider"):
        pmanager.processed_per_glider = {}
    if not hasattr(pmanager, "_exportables"):
        pmanager._exportables = {"raw": {}, "processed": {}, "lite": {}}

    # Use cached medians if available, else compute once and cache
    if (
        target_name in pmanager.processed_per_glider
        and "agg" in pmanager.processed_per_glider[target_name]
    ):
        t_med = pmanager.processed_per_glider[target_name]["agg"]
    else:
        # standardize names → interpolate → aggregate → cache + export handle
        t_interp = interpolate_DEPTH(target_ds_raw)
        t_med = aggregate_vars(t_interp, variables)  # dims: PROFILE_NUMBER, DEPTH_bin
        pmanager.processed_per_glider[target_name] = {
            "interp": t_interp,
            "agg": t_med,
        }
        pmanager._exportables["raw"][target_name] = target_ds_raw
        pmanager._exportables["processed"][target_name] = t_med

    # ---- Device: load & aggregate (external; not part of pipelines) ----
    device_alias = vcfg.get("aliases", None)  # {STD: device_col}
    # Convert to {device_col: STD} for loader renaming
    dev_to_std = (
        {dev: std for std, dev in device_alias.items() if dev} if device_alias else None
    )
    device_ds_raw = load_device_folder_to_xarray(folder_path, alias_map=dev_to_std)

    # Summaries (for pairing)
    target_summary = summarising_profiles(target_ds_raw, target_name).reset_index(
        drop=True
    )
    device_summary = summarising_profiles(device_ds_raw, device_name).reset_index(
        drop=True
    )

    # Pairs
    paired_df = find_profile_pair_metadata(
        df_target=target_summary,
        df_ancillary=device_summary,
        target_name=target_name,
        ancillary_name=device_name,
        time_thresh_hr=max_time_hr,
        dist_thresh_km=max_dist_km,
    )
    if paired_df.empty:
        print("[Validation] No matched target/device profiles found.")
        return {"paired_df": paired_df, "r2_ds": None, "merged": None}

    print(f"[Validation] Matched {len(paired_df)} pairs with {device_name}.")

    # Device medians (computed here)
    d_interp = interpolate_DEPTH(device_ds_raw)
    d_med = aggregate_vars(d_interp, variables)  # dims: PROFILE_NUMBER, DEPTH_bin

    # IDs from the pairs
    t_ids = paired_df[f"{target_name}_PROFILE_NUMBER"].values
    d_ids = paired_df[f"{device_name}_PROFILE_NUMBER"].values

    # filter to just those profiles (works on aggregated 2-D medians)
    t_med = filter_xarray_by_profile_ids(t_med, "PROFILE_NUMBER", t_ids)
    d_med = filter_xarray_by_profile_ids(d_med, "PROFILE_NUMBER", d_ids)

    # Trim pairs to those actually present in the aggregated sets
    t_present = set(t_med["PROFILE_NUMBER"].values.tolist())
    d_present = set(d_med["PROFILE_NUMBER"].values.tolist())

    mask_pairs = paired_df[f"{target_name}_PROFILE_NUMBER"].isin(t_present) & paired_df[
        f"{device_name}_PROFILE_NUMBER"
    ].isin(d_present)
    paired_df = paired_df.loc[mask_pairs].reset_index(drop=True)

    # Log what was dropped
    dropped_t = set(t_ids) - t_present
    dropped_d = set(d_ids) - d_present
    if dropped_t:
        print(
            f"[Validation] Dropped {len(dropped_t)} target profiles with no aggregated data."
        )
    if dropped_d:
        print(
            f"[Validation] Dropped {len(dropped_d)} device profiles with no aggregated data."
        )

    # Merge pairs → dims: PAIR_INDEX, DEPTH_bin
    merged = merge_pairs_from_filtered_aggregates(
        paired_df=paired_df,
        agg_target=t_med,
        agg_anc=d_med,
        target_name=target_name,
        ancillary_name=device_name,
        variables=variables,
        bin_dim="DEPTH_bin",
        pair_dim="PAIR_INDEX",
    )

    print(f"[Validation] Merged data has {merged.sizes['PAIR_INDEX']} pairs.")

    # R² per pair
    r2_ds = compute_r2_for_merged_profiles_xr(
        merged, variables=variables, target_name=target_name, ancillary_name=device_name
    )

    # Plot heatmaps with the shared helper
    align_cfg = pmanager.settings.get("alignment", {}) or {}
    r2_thresholds = align_cfg.get(
        "r2_thresholds", [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    )

    r2_datasets_for_plot = {device_name: r2_ds}
    plot_r2_heatmaps_per_pair(
        r2_datasets=r2_datasets_for_plot,
        variables=variables,
        target_name=target_name,
        r2_thresholds=r2_thresholds,
        time_thresh_hr=max_time_hr,
        dist_thresh_km=max_dist_km,
        figsize=(9, 6),
        save_plots=save_plots,
        output_path=plot_output_path or None,
        show_plots=show_plots,
    )

    return {"paired_df": paired_df, "merged": merged, "r2_ds": r2_ds}
