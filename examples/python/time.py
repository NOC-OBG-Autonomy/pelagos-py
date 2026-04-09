import os
import sys
import time
import yaml
import requests
import numpy as np
import xarray as xr
from pathlib import Path
from toolbox.pipeline import Pipeline, _setup_logging

# --- Timing Configuration ---
DATA_URL = "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001/Nelson_20240528/Nelson_646_R.nc"
TEMP_INPUT = Path("Nelson_temp_input.nc")
TEMP_OUTPUT = Path("Nelson_temp_output.nc")
CHLA_DEPTH_RATIO = 0.9 
DEFAULT_CHLA_DEPTH = 550

BASE_CONFIG_YAML = """
pipeline:
  name: Performance Profiling Pipeline
  description: Timing each step of the CTD process
  visualisation: false

steps:
  - name: Load OG1
    parameters:
      file_path: PLACEHOLDER
    diagnostics: false

  - name: Apply QC
    parameters:
      qc_settings:
        impossible date qc: {}
        impossible location qc: {}
        position on land qc: {}
        impossible speed qc: {}
    diagnostics: false

  - name: Apply QC
    parameters:
      qc_settings:
        impossible range qc:
          variable_ranges:
            PRES:
              3: [-5, -2.4]
              4: [-.inf, -5]
          also_flag:
            PRES: [CNDC, TEMP]
          plot: [PRES]
        stuck value qc:
          variables:
            PRES: 2
          also_flag:
            PRES: [CNDC, TEMP]
          plot: [PRES]
    diagnostics: false

  - name: Interpolate Data
    parameters:
      qc_handling_settings:
        flag_filter_settings:
          PRES: [3, 4, 9]
          LATITUDE: [3, 4, 9]
          LONGITUDE: [3, 4, 9]
        reconstruction_behaviour: replace
        flag_mapping:
          3: 8
          4: 8
          9: 8
    diagnostics: false

  - name: Derive CTD
    parameters:
      to_derive: [DEPTH]
    diagnostics: false

  - name: Find Profiles Beta
    diagnostics: false

  - name: Apply QC
    parameters:
      qc_settings:
          valid profile qc:
            profile_length: 50
            depth_range: [0, 1000]
    diagnostics: false

  - name: Salinity Adjustment
    parameters:
      qc_handling_settings:
        flag_filter_settings:
          CNDC: [3, 4, 9]
          TEMP: [3, 4, 9]
          PROFILE_NUMBER: [3, 4, 9]
        reconstruction_behaviour: reinsert
        flag_mapping:
          0: 5
          1: 5
          2: 5
      filter_window_size: 21
    diagnostics: false

  - name: Derive CTD
    parameters:
      to_derive: [PRAC_SALINITY, ABS_SALINITY, CONS_TEMP, DENSITY]
    diagnostics: false

  - name: Chla Deep Correction
    parameters:
      apply_to: CHLA
      dark_value: null
      depth_threshold: -550
    diagnostics: false

  - name: Chla Quenching Correction
    parameters:
      method: Argo
      apply_to: CHLA
      mld_settings:
        threshold_on: DENSITY
        reference_depth: -10
        threshold: 0.05
    diagnostics: false

  - name: BBP from Beta
    parameters:
      theta: 124
      xfactor: 1.076
    diagnostics: false

  - name: Isolate BBP Spikes
    parameters:
      window_size: 50
      method: median
    diagnostics: false

  - name: Data Export
    parameters:
      export_format: netcdf
      output_path: PLACEHOLDER
"""

def setup_data():
    if not TEMP_INPUT.exists():
        print(f"Downloading test data from {DATA_URL}...")
        response = requests.get(DATA_URL)
        with open(TEMP_INPUT, "wb") as f:
            f.write(response.content)
    return str(TEMP_INPUT.resolve())

def get_dynamic_config(input_path):
    config = yaml.safe_load(BASE_CONFIG_YAML)
    config["steps"][0]["parameters"]["file_path"] = input_path
    config["steps"][-1]["parameters"]["output_path"] = str(TEMP_OUTPUT.resolve())
    
    with xr.open_dataset(input_path) as ds:
        has_location = "LATITUDE" in ds.variables and "LONGITUDE" in ds.variables
        max_depth = float(ds["PRES"].max()) if "PRES" in ds.variables else 1000

    if not has_location:
        qc_settings = config["steps"][1]["parameters"]["qc_settings"]
        for test in ["impossible location qc", "position on land qc", "impossible speed qc"]:
            qc_settings.pop(test, None)
            
    calculated_threshold = -int(max_depth * CHLA_DEPTH_RATIO)
    final_threshold = max(-DEFAULT_CHLA_DEPTH, calculated_threshold)
    
    for step in config["steps"]:
        if step["name"] == "Chla Deep Correction":
            step["parameters"]["depth_threshold"] = final_threshold
            
    return config

def run_profiler():
    input_file = setup_data()
    full_config = get_dynamic_config(input_file)
    
    pipe = Pipeline()
    pipe.global_parameters = full_config["pipeline"]
    pipe.logger = _setup_logging(log_file=None)
    
    step_timings = []
    total_start = time.perf_counter()
    
    print("\n" + "="*60)
    print(f"{'STEP NAME':<35} | {'TIME (s)':<10}")
    print("-"*60)
    
    for step_cfg in full_config["steps"]:
        step_name = step_cfg["name"]
        
        # Specific sub-labeling for Apply QC blocks to distinguish them
        if step_name == "Apply QC":
            tests = list(step_cfg["parameters"]["qc_settings"].keys())
            display_name = f"Apply QC ({', '.join(tests[:2])}...)"
        else:
            display_name = step_name

        start_time = time.perf_counter()
        pipe._context = pipe.execute_step(step_cfg, pipe._context)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        step_timings.append((display_name, duration))
        print(f"{display_name:<35} | {duration:>10.3f}s")

    total_end = time.perf_counter()
    
    print("-"*60)
    print(f"{'TOTAL PIPELINE TIME':<35} | {total_end - total_start:>10.3f}s")
    print("="*60 + "\n")

    # Cleanup
    if TEMP_INPUT.exists():
        os.remove(TEMP_INPUT)
    if TEMP_OUTPUT.exists():
        os.remove(TEMP_OUTPUT)
    print("Temporary files deleted. Profiling complete.")

if __name__ == "__main__":
    try:
        run_profiler()
    except Exception as e:
        print(f"Profiler failed: {e}")
        # Ensure cleanup even on failure
        if TEMP_INPUT.exists(): os.remove(TEMP_INPUT)
        if TEMP_OUTPUT.exists(): os.remove(TEMP_OUTPUT)