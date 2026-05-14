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

from pelagos_py.utils.config_mirror import ConfigMirrorMixin

import os
import yaml
import pandas as pd
import numpy as np
import xarray as xr
import datetime as _dt

from pelagos_py.pipeline import Pipeline
from pelagos_py.utils.diagnostics import (
    summarising_profiles,
    plot_distance_time_grid,
    plot_glider_pair_heatmap_grid,
)
from pelagos_py.utils.alignment import (
    interpolate_DEPTH,
    aggregate_vars,
    merge_pairs_from_filtered_aggregates,
    filter_xarray_by_profile_ids,
    find_profile_pair_metadata,
    compute_r2_for_merged_profiles_xr,
    plot_r2_heatmaps_per_pair,
    plot_pair_scatter_grid,
    collect_xy_from_r2_ds,
    fit_linear_map,
)
from pelagos_py.utils.validation import validate


class PipelineManager(ConfigMirrorMixin):
    """A class enabling the execution of multiple pipelines in sequence."""

    def __init__(self):
        # init regular state
        self.pipelines = {}  # {pipeline_name: Pipeline instance}
        self.alignment_map = {}  # {standard_name: {pipeline_name: alias}}
        self._contexts = {}
        self.settings = {}
        self._summary_ran = False
        # NEW: private config
        self._init_config_mirror()

    def load_mission_control(self, config_path, mirror_keys=None):
        """
        Load MissionControl YAML into private self._parameters.
        - Builds pipelines
        - Builds alignment_map
        - Mirrors selected keys as attributes (e.g., 'settings')
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        # 1) Store full mission config in private _parameters
        self.load_config(config, mirror_keys=mirror_keys or ["settings"])

        # 2) Build pipelines (also load each pipeline's config file into its own private store)
        for entry in self._parameters.get("pipelines", []) or []:
            name = entry["name"]
            cfg_path = entry["config"]
            self.add_pipeline(name, cfg_path)

        # 3) Alignment aliases → alignment_map
        alignment_vars = (
            self._parameters.get("alignment", {}).get("variables", {}) or {}
        )
        self.alignment_map = {
            std: (details or {}).get("aliases", {}) or {}
            for std, details in alignment_vars.items()
        }

        # 4) Mirror settings (and any other mirrored keys) into attributes
        self._reset_parameter_bridge(mirror_keys=self._param_attr_keys or {"settings"})

    def add_pipeline(self, name, config_path):
        """Add a single pipeline with a unique name."""
        if name in self.pipelines:
            raise ValueError(f"Pipeline '{name}' already added.")
        pl = Pipeline(config_path)  # assumes Pipeline accepts path (see section C)
        self.pipelines[name] = pl
        print(f"[Pipeline Manager] Pipeline '{name}' added from {config_path}.")

    def save_manager_config(self, path: str):
        """Save MissionControl/Manager config from self._parameters."""
        self.save_config(path)

    def save_pipeline_configs(self, out_dir: str, filename="{name}.yaml"):
        """
        Ask each Pipeline to write its private config to YAML.
        The pipeline file content comes from pipeline._parameters (including its steps).
        """
        os.makedirs(out_dir, exist_ok=True)
        for name, pl in self.pipelines.items():
            out = os.path.join(out_dir, filename.format(name=name))
            pl.save_config(out)
        print(f"[Config] Saved pipeline configs → {out_dir}")

    def save_all_configs(
        self, manager_path: str, pipelines_dir: str, pipeline_filename="{name}.yaml"
    ):
        """Convenience: save manager config and all pipeline configs."""
        self.save_manager_config(manager_path)
        self.save_pipeline_configs(pipelines_dir, filename=pipeline_filename)

    def run_all(self):
        """Run all registered pipelines and cache the resulting contexts."""
        for name, pipeline in self.pipelines.items():
            print("#" * 20)
            print(f"Running pipeline: {name}")
            pipeline.run()

        self._contexts = self.get_contexts()
        print("#" * 20)
        print("All pipelines executed successfully.")
        print(f"Contexts cached: {self._contexts.keys()}")
        print("#" * 20)

    def get_contexts(self):
        """Retrieve the context dictionary from each pipeline."""
        return {name: p._context for name, p in self.pipelines.items()}

    def load_data(self, filepath, platform_name):
        context = {"data": xr.load_dataset(filepath)}
        self._contexts[platform_name] = context
        print(f"[Pipeline Manager] {platform_name} sucessfully added added")

    def summarise_all_profiles(self) -> pd.DataFrame:
        """
        For all pipelines, summarise profiles and plot glider-to-glider distance time series.
        This includes:
            - Computing median TIME, LATITUDE, LONGITUDE per profile
            - Matching each profile to its closest in time from another source
            - Plotting a distance grid comparing all gliders

        Returns
        -------
        pd.DataFrame
            Concatenated summary of all glider profiles, with closest match info appended.
        """
        self._summary_ran = True
        if self._contexts is None:
            raise RuntimeError("Pipelines must be run before generating summaries.")

        print("[Pipeline Manager] Generating glider distance summaries...")
        # Step 1: Generate per-glider summaries
        self.summary_per_glider = {}
        for pipeline_name, context in self._contexts.items():
            ds = context["data"]
            if not isinstance(ds, xr.Dataset):
                raise TypeError(f"Pipeline '{pipeline_name}' has invalid dataset.")
            else:
                print(f"[Pipeline Manager] Processing dataset for {pipeline_name}...")

            summary_df = summarising_profiles(ds, pipeline_name)
            print("Summary Columns:", summary_df.columns.tolist())
            self.summary_per_glider[pipeline_name] = summary_df

        # Step 2: Find closest profiles across gliders
        # Extract diagnostic flags from settings
        show_plots = self.settings.get("diagnostics", {}).get("show_plots", True)
        save_plots = self.settings.get("diagnostics", {}).get("save_plots", False)
        distance_over_time_matrix = self.settings.get("diagnostics", {}).get(
            "distance_over_time_matrix", False
        )
        self.matchup_thresholds = self.settings.get("diagnostics", {}).get(
            "matchup_thresholds", {}
        )
        max_time_threshold = (
            self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("max_time_threshold", 12)
        )
        max_distance_threshold = (
            self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("max_distance_threshold", 20)
        )
        bin_size = (
            self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("bin_size", 2)
        )

        if not distance_over_time_matrix:
            print("[Pipeline Manager] Distance over time matrix is disabled.")
        else:
            print("[Pipeline Manager] Plotting distance time grid...")
            # After generating all summaries...
            combined_summaries = plot_distance_time_grid(
                summaries=self.summary_per_glider,
                output_path=self.settings.get("diagnostics", {}).get(
                    "distance_plot_output", None
                ),
                show=self.settings.get("diagnostics", {}).get("show_plots", True),
            )

        if not self.matchup_thresholds:
            print(
                "[Pipeline Manager] Matchup thresholds are not set. Skipping heatmap grid."
            )
        else:
            print("[Pipeline Manager] Finding closest profiles across gliders...")
            # compute time taken for caluclations
            start_time = pd.Timestamp.now()
            plot_glider_pair_heatmap_grid(
                summaries=self.summary_per_glider,
                time_bins=np.arange(0, max_time_threshold + 1, bin_size),
                dist_bins=np.arange(0, max_distance_threshold + 1, bin_size),
                output_path=self.settings.get("diagnostics", {}).get(
                    "heatmap_output", None
                ),
                show=self.settings.get("diagnostics", {}).get("show_plots", True),
            )
            end_time = pd.Timestamp.now()
            print(f"[Pipeline Manager] Heatmap grid plotted in {end_time - start_time}")

        return

    def preview_alignment(self, target="None"):
        """
        Align all datasets to a target dataset and compute R² against ancillary sources.

        This version:
        - Renames each pipeline's variables to the standard names (from alignment_map)
        - Runs interpolate + aggregate ONCE per pipeline and caches the results
        - Uses the cached medians for pairing/merging/R²
        - Populates exportable handles for raw/processed/lite data
        """

        # === PRECONDITIONS ===
        if not self._summary_ran:
            raise RuntimeError("Run summarise_all_profiles() before alignment.")

        if target not in self.pipelines:
            raise ValueError(f"Target pipeline '{target}' not found.")

        # === CONFIG ===
        alignment_vars = list(self.alignment_map.keys())
        self.r2_datasets = {}  # Reset R² result container

        # ---- Helper: alias -> std renamer for a given pipeline name ----
        def _rename_to_standard(name: str, ds):
            rename_map = {
                alias: std
                for std, alias_map in self.alignment_map.items()
                if (alias := alias_map.get(name)) and alias in ds
            }
            return ds.rename(rename_map) if rename_map else ds, rename_map

        if not hasattr(self, "processed_per_glider"):
            self.processed_per_glider = {}

        # export registry the rest of your workflow can use later to write files
        if not hasattr(self, "_exportables"):
            self._exportables = {"raw": {}, "processed": {}, "lite": {}}

        # === COLLECT: standardised & processed datasets for ALL pipelines (target + ancillaries) ===
        for name, ctx in self._contexts.items():
            raw_ds = ctx["data"]

            # Keep a pointer to raw data for export (no copy)
            self._exportables["raw"][name] = raw_ds

            # If we already processed this pipeline, skip recomputation
            if (
                name in self.processed_per_glider
                and "agg" in self.processed_per_glider[name]
            ):
                continue

            # 1) rename variables to standard names
            ds_std, used_map = _rename_to_standard(name, raw_ds)

            # 2) interpolate depth
            print(f"[Pipeline Manager] Interpolating DEPTH for '{name}'...")
            ds_interp = interpolate_DEPTH(ds_std)

            # 3) aggregate medians (2-D by PROFILE_NUMBER × DEPTH_bin)
            print(f"[Pipeline Manager] Aggregating medians for '{name}'...")
            ds_agg = aggregate_vars(ds_interp, alignment_vars)

            # store in cache
            self.processed_per_glider[name] = {
                "renamed": ds_std,
                "interp": ds_interp,
                "agg": ds_agg,
            }

            # make processed export handle available
            self._exportables["processed"][name] = ds_agg
            if "lite" not in self._exportables:
                self._exportables["lite"] = {}

        # Prepare target objects
        target_summary = self.summary_per_glider[target].reset_index()
        target_agg = self.processed_per_glider[target]["agg"]

        # === LOOP: align each ancillary to target using the cached medians ===
        for ancillary_name, ctx in self._contexts.items():
            if ancillary_name == target:
                continue

            print(
                f"\n[Pipeline Manager] Aligning '{ancillary_name}' to target '{target}'..."
            )

            ancillary_summary = self.summary_per_glider[ancillary_name]
            if ancillary_summary.index.names[0] is not None:
                ancillary_summary = ancillary_summary.reset_index()

            # === STEP 1: Find Matched Profile Pairs ===
            paired_df = find_profile_pair_metadata(
                df_target=target_summary,
                df_ancillary=ancillary_summary,
                target_name=target,
                ancillary_name=ancillary_name,
                time_thresh_hr=self.settings.get("diagnostics", {})
                .get("matchup_thresholds", {})
                .get("max_time_threshold", 12),
                dist_thresh_km=self.settings.get("diagnostics", {})
                .get("matchup_thresholds", {})
                .get("max_distance_threshold", 20),
            )

            if paired_df.empty:
                print(
                    f"[Pipeline Manager] No matched profiles between {target} and {ancillary_name}."
                )
                continue

            print(f"[Pipeline Manager] Found {len(paired_df)} matched profile pairs.")

            # === STEP 2: Use CACHED aggregated medians ===
            binned_ds = {
                target: target_agg,
                ancillary_name: self.processed_per_glider[ancillary_name]["agg"],
            }

            # === STEP 3: Filter Datasets by Matched Profile IDs ===
            filtered_ds = {}
            for glider_name, agg_ds in [
                (target, binned_ds[target]),
                (ancillary_name, binned_ds[ancillary_name]),
            ]:
                profile_ids = paired_df[f"{glider_name}_PROFILE_NUMBER"].values
                filtered_ds[glider_name] = filter_xarray_by_profile_ids(
                    ds=agg_ds,
                    profile_id_var="PROFILE_NUMBER",
                    valid_ids=profile_ids,
                )

            # === STEP 4: Build pairwise merged dataset ===
            merged = merge_pairs_from_filtered_aggregates(
                paired_df=paired_df,
                agg_target=filtered_ds[target],
                agg_anc=filtered_ds[ancillary_name],
                target_name=target,
                ancillary_name=ancillary_name,
                variables=alignment_vars,  # the raw names; helper will use median_{var}
            )

            print("[Align] Merged dims:", merged.dims)
            print("[Align] Vars:", list(merged.data_vars))

            # === STEP 5: Compute R² ===
            print(f"[Pipeline Manager] Computing R² for '{ancillary_name}'...")
            r2_ds = compute_r2_for_merged_profiles_xr(
                ds=merged,
                variables=alignment_vars,
                target_name=target,
                ancillary_name=ancillary_name,
            )

            self.r2_datasets[ancillary_name] = r2_ds

            # add to cache
            self._exportables["processed"][f"{target}_vs_{ancillary_name}"] = r2_ds
            print(
                f"[Pipeline Manager] R² dataset stored for '{target}' vs '{ancillary_name}'."
            )

        print("\n[Pipeline Manager] Alignment complete for all datasets.")

        # Set R² thresholds
        r2_thresholds = self.settings.get("alignment", {}).get(
            "r2_thresholds", [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        )

        # Call the plotting function
        # r2_datasets produced by align_to_target
        plot_r2_heatmaps_per_pair(
            r2_datasets=self.r2_datasets,
            variables=list(self.alignment_map.keys()),
            target_name=target,  # e.g. "Doombar"
            r2_thresholds=r2_thresholds,
            time_thresh_hr=self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("max_time_threshold", 12),
            dist_thresh_km=self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("max_distance_threshold", 20),
            figsize=(9, 6),
            save_plots=self.settings.get("alignment", {}).get("save_plots", False),
            output_path=self.settings.get("alignment", {}).get(
                "plot_output_path", "r2_heatmap_grid.png"
            ),
            show_plots=self.settings.get("alignment", {}).get("show_plots", True),
        )

    def fit_and_save_to_target(
        self,
        target,
        out_dir=None,
        variable_r2_criteria=None,
        max_time_hr=None,
        max_dist_km=None,
        ancillaries=None,
        overwrite=False,
        show_plots=True,
    ):
        """
        Fit ancillary variables to target datasets using profile-pair medians and per-variable R² criteria.
        """
        if out_dir is None:
            # get directory from settings or create timestamped dir
            datetime_str = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            out_dir = self.settings.get("alignment", {}).get(
                "output_path", datetime_str
            )
        if not getattr(self, "r2_datasets", None):
            raise RuntimeError("Run preview_alignment() before fitting.")
        if target not in self.pipelines or target not in self._contexts:
            raise ValueError(f"Target '{target}' not available in manager contexts.")

        alignment_vars = list(self.alignment_map.keys())
        if variable_r2_criteria is None:
            variable_r2_criteria = self.settings.get("alignment", {}).get(
                "variable_r2_criteria", {}
            )
        missing = [v for v in alignment_vars if v not in variable_r2_criteria]
        if missing:
            raise ValueError(f"Missing R² thresholds for variables: {missing}")

        os.makedirs(out_dir, exist_ok=True)

        all_sources = [n for n in self.pipelines.keys() if n != target]
        anc_list = list(ancillaries) if ancillaries else all_sources

        if show_plots and hasattr(self, "plot_pair_scatter_grid"):
            try:
                plot_pair_scatter_grid(
                    r2_datasets=self.r2_datasets,
                    variables=alignment_vars,
                    target_name=target,
                    variable_r2_criteria=variable_r2_criteria,
                    max_time_hr=max_time_hr,
                    max_dist_km=max_dist_km,
                )
            except Exception as e:
                print(f"[Fit] (Plot) Skipped grid due to: {e}")

        def _alias_map_for(source_name):
            return {
                alias: std
                for std, mapping in self.alignment_map.items()
                if (alias := mapping.get(source_name))
            }

        saved_paths = {}
        fits_summary = {}

        for anc in anc_list:
            if anc not in self._contexts:
                print(f"[Fit] Skipping '{anc}' (no context).")
                continue

            print(f"\n[Fit] === {anc} → align to {target} ===")
            r2_ds = self.r2_datasets.get(anc)
            if r2_ds is None or not isinstance(r2_ds, xr.Dataset):
                print(f"[Fit] No R² dataset for '{anc}'. Skipping.")
                continue

            anc_ds = self._contexts[anc]["data"]
            rename_map = _alias_map_for(anc)
            anc_ds_std = anc_ds.rename(rename_map) if rename_map else anc_ds

            # Compute per-variable fits
            anc_fits = {}
            for var in alignment_vars:
                x, y = collect_xy_from_r2_ds(
                    r2_ds,
                    var=var,
                    target_name=target,
                    ancillary_name=anc,
                    r2_min=variable_r2_criteria.get(var),
                    time_max=max_time_hr,
                    dist_max=max_dist_km,
                )
                fit = fit_linear_map(x, y)
                anc_fits[var] = fit
                print(
                    f"[Fit] {anc}:{var} slope={fit['slope']:.4g} intercept={fit['intercept']:.4g} "
                    f"R²={fit['r2']:.3f} N={fit['n']}"
                )

            # Apply to full ancillary dataset (creates {VAR}_ALIGNED_TO_{target})
            ds_out = anc_ds_std.copy()
            created_vars = []
            for var in alignment_vars:
                if var not in ds_out:
                    print(f"[Fit] [{anc}] missing '{var}' — skip.")
                    continue
                slope = anc_fits[var]["slope"]
                intercept = anc_fits[var]["intercept"]
                npts = anc_fits[var]["n"]
                out_name = f"{var}_ALIGNED_TO_{target}"

                aligned = slope * ds_out[var] + intercept
                aligned = aligned.astype(ds_out[var].dtype, copy=False)
                aligned.name = out_name
                aligned.attrs.update(
                    {
                        "long_name": f"{var} aligned to {target}",
                        "alignment_target": target,
                        "alignment_source": anc,
                        "alignment_slope": float(slope),
                        "alignment_intercept": float(intercept),
                        "alignment_fit_points": int(npts),
                        "alignment_generated": _dt.datetime.utcnow().isoformat() + "Z",
                    }
                )
                ds_out[out_name] = aligned
                created_vars.append(out_name)

            if not created_vars:
                print(f"[Fit] No aligned variables for '{anc}'. Skipping save.")
                continue

            # Keep aligned vars in memory
            try:
                self._contexts[anc]["data"] = ds_out
            except Exception:
                pass

            # immediately build & cache aggregated aligned medians for this ancillary
            try:
                _ = self._aggregate_aligned_vars_for_ancillary(
                    anc_name=anc, target=target, vars_to_aggregate=alignment_vars
                )  # populates self.processed_per_glider[anc][f'agg_aligned_to_{target}']
            except Exception as e:
                print(
                    f"[Fit] Warning: failed to cache aggregated aligned medians for '{anc}': {e}"
                )

            # Save per-ancillary file
            out_path = os.path.join(out_dir, f"{anc}_aligned_to_{target}.nc")
            if (not overwrite) and os.path.exists(out_path):
                print(f"[Fit] File exists, not overwriting: {out_path}")
            else:
                encoding = {
                    name: {"zlib": True, "complevel": 2} for name in created_vars
                }
                try:
                    ds_out.to_netcdf(out_path, encoding=encoding)
                    print(f"[Fit] Saved: {out_path}")
                    saved_paths[anc] = out_path
                    fits_summary[anc] = anc_fits
                except Exception as e:
                    print(f"[Fit] Failed to save '{anc}': {e}")

            # Cache fits for metadata
            if anc in self.processed_per_glider:
                self.processed_per_glider[anc][
                    f"last_fit_to_target_{target}"
                ] = anc_fits

        return {"paths": saved_paths, "fits": fits_summary}

    def validate_with_device(self, target="None", **overrides):
        """
        Run the validation workflow using settings['validation'].
        Optionally pass keyword overrides (e.g., show_plots=False) for this call only.

        Examples:
            mngr.validate_with_device("Doombar")
            mngr.validate_with_device("Doombar", show_plots=False, apply_and_save=True)
        """
        # Fast path: no overrides → just call through
        if not overrides:
            validate(self, target=target)
            return

        # One-shot overrides: temporarily update settings['validation']
        vcfg_orig = dict(self.settings.get("validation", {}))  # shallow copy
        try:
            vcfg = self.settings.setdefault("validation", {})
            vcfg.update(overrides)
            validate(self, target=target)
        finally:
            # restore original validation config
            self.settings["validation"] = vcfg_orig

    def fit_to_device(self, target="None"):
        """
        Fit TARGET variables to a validation device using profile-pair medians and per-variable R² criteria.
        The mapping is fit as: device = slope * target + intercept, then applied to the FULL target dataset
        to create new variables `{VAR}_ALIGNED_TO_{DEVICE}`.

        Reads options from self.settings['validation']:
        validation:
            device_name: "<device label>"
            variable_names: ["CNDC","TEMP", ...]        # optional; defaults to alignment_map keys
            variable_r2_criteria: {CNDC: 0.95, TEMP: 0.9, ...}
            max_time_threshold: <float>
            max_distance_threshold: <float>
            save_plots: <bool>
            show_plots: <bool>
            plot_output_path: "<file or dir>"
            apply_and_save: <bool>
            output_path: "<dir or empty for timestamped dir>"

        Returns
        -------
        dict with:
        - "path": output NetCDF (if saved)
        - "fits": {var: {slope, intercept, r2, n}, ...}
        - "device_name": device label used
        """
        # --- Preconditions ---
        if type(target) == str:
            if target not in self.pipelines:
                raise ValueError(f"Target pipeline '{target}' not found.")
            if target not in self._contexts:
                raise ValueError(f"Target pipeline '{target}' has no context data.")
            target_name = target
        elif type(target) == list:
            for platform in target:
                if platform not in self.pipelines or platform not in self._contexts:
                    raise ValueError(f"Target '{platform}' not available.")
            target_name = "_".join(target)

        # --- Validation config ---
        vcfg = self.settings.get("validation", {}) or {}
        device_name = vcfg.get("device_name", "DEVICE")
        variables = vcfg.get("variable_names", list(self.alignment_map.keys()))
        var_r2_criteria = vcfg.get("variable_r2_criteria", {}) or {}
        max_time_hr = vcfg.get("max_time_threshold", None)
        max_dist_km = vcfg.get("max_distance_threshold", None)
        show_plots = bool(vcfg.get("show_plots", True))
        save_plots = bool(vcfg.get("save_plots", False))
        plot_output_path = vcfg.get("plot_output_path", "device_fit_scatter_grid.png")
        apply_and_save = bool(vcfg.get("apply_and_save", False))
        out_dir = vcfg.get("output_path", "") or ""

        # Validate thresholds exist for all requested variables
        missing = [v for v in variables if v not in var_r2_criteria]
        if missing:
            raise ValueError(
                f"[Fit→Device] R² threshold missing for variables: {missing}"
            )
        print(f"[Fit→Device] Using device='{device_name}', variables={variables}")
        print(f"[Fit→Device] R² thresholds: {var_r2_criteria}")

        # --- Ensure we have the R² dataset for TARGET vs DEVICE ---
        # This will run the whole validation pipeline (load device, pair, aggregate, merge, compute R²)
        from .utils.validation import validate  # adjust import if your layout differs

        val_res = validate(self, target=target)
        r2_ds = val_res.get("r2_ds", None)
        if r2_ds is None or not isinstance(r2_ds, xr.Dataset):
            raise RuntimeError(
                "[Fit→Device] No R² dataset available from validation()."
            )

        # --- QA scatter grid (X=device, Y=target) before fitting ---
        if show_plots or save_plots:
            try:
                # plot_pair_scatter_grid expects a dict of {ancillary_name: ds}
                ds_map = {device_name: r2_ds}
                fig, _ = plot_pair_scatter_grid(
                    r2_datasets=ds_map,
                    variables=variables,
                    target_name=target_name,
                    variable_r2_criteria=var_r2_criteria,
                    max_time_hr=max_time_hr,
                    max_dist_km=max_dist_km,
                    ancillaries_order=[device_name],
                )
                if save_plots:
                    # If path looks like a directory, drop a default filename into it
                    out_is_dir = (plot_output_path.endswith(os.sep)) or (
                        os.path.isdir(plot_output_path)
                    )
                    if out_is_dir:
                        os.makedirs(plot_output_path, exist_ok=True)
                        fout = os.path.join(
                            plot_output_path,
                            f"{target_name}_vs_{device_name}_fit_grid.png",
                        )
                    else:
                        os.makedirs(
                            os.path.dirname(plot_output_path) or ".", exist_ok=True
                        )
                        fout = plot_output_path
                    fig.savefig(fout, dpi=300)
                    print(f"[Fit→Device] Saved scatter grid to: {fout}")
                if not show_plots:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
            except Exception as e:
                print(f"[Fit→Device] (Plot) Skipped grid due to: {e}")

        # --- Compute fits to map TARGET → DEVICE for each variable ---
        # collect_xy_from_r2_ds returns (X=device, Y=target). For TARGET→DEVICE we invert to (x=target, y=device).
        fits = {}
        for var in variables:
            X_dev, Y_tgt = collect_xy_from_r2_ds(
                r2_ds,
                var=var,
                target_name=target_name,
                ancillary_name=device_name,
                r2_min=var_r2_criteria.get(var),
                time_max=max_time_hr,
                dist_max=max_dist_km,
            )
            # invert orientation for target->device mapping
            x = Y_tgt  # target
            y = X_dev  # device
            info = fit_linear_map(x, y)  # fits y_device = a * x_target + b
            fits[var] = info
            print(
                f"[Fit→Device] {var}: device ≈ {info['slope']:.4g}·target + {info['intercept']:.4g} "
                f"(R²={info['r2']:.3f}, N={info['n']})"
            )

        return

    def apply_adjustment(self, target, fit_params):

        if target not in self._contexts:
            raise ValueError(f"Target pipeline '{target}' has no context data.")

        # --- config ---
        vcfg = self.settings.get("validation", {}) or {}
        device_name = vcfg.get("device_name", "DEVICE")

        # --- Apply mapping to the FULL target dataset (create {var}_ALIGNED_TO_{device}) ---
        # Rename target variables to standard names based on alignment_map aliases
        target_ds_raw = self._contexts[target]["data"]
        rename_map = {
            alias: std
            for std, alias_map in self.alignment_map.items()
            if (alias := alias_map.get(target)) and alias in target_ds_raw
        }
        target_ds_std = (
            target_ds_raw.rename(rename_map) if rename_map else target_ds_raw
        )

        ds_out = target_ds_std.copy()
        for var, info in fit_params.items():
            if var not in ds_out:
                print(f"[Fit→Device] Target missing variable '{var}' — skipping.")
                continue
            slope, intercept, npts = info["slope"], info["intercept"], info["n"]
            out_name = f"{var}_ALIGNED_TO_{device_name}"
            aligned = (slope * ds_out[var] + intercept).astype(
                ds_out[var].dtype, copy=False
            )
            aligned.name = out_name
            aligned.attrs.update(
                {
                    "long_name": f"{var} aligned to {device_name}",
                    "alignment_target": target,
                    "alignment_reference_device": device_name,
                    "alignment_direction": "target_to_device",
                    "alignment_slope": float(slope),
                    "alignment_intercept": float(intercept),
                    "alignment_fit_points": int(npts),
                }
            )
            self._contexts[target]["data"][out_name] = aligned

        # update processed_per_glider for potential later use
        if not hasattr(self, "processed_per_glider"):
            self.processed_per_glider = {}
        if target not in self.processed_per_glider:
            self.processed_per_glider[target] = {}
        self.processed_per_glider[target][
            f"last_fit_to_device_{device_name}"
        ] = fit_params

        return {"fits": fit_params, "device_name": device_name}

    def save(self, dir, raw=True, processed=True):

        saving = {"raw": raw, "processed": processed}

        for data_output, to_save in saving.items():
            print(f"Saving {data_output} outputs.")
            if to_save:
                if len(self._exportables) == 0:
                    print(f"There is no {data_output} data to save.")
                    continue
                for platform_name, data in self._exportables[data_output].items():
                    data.to_netcdf(
                        os.path.join(dir, f"{platform_name}_{data_output}.nc")
                    )
