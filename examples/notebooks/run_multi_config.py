import sys
import yaml
import xarray as xr
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from toolbox.pipeline import Pipeline, _setup_logging

# --- Configuration Variables ---
INPUT_DIRECTORY = "/Users/orlpru/Desktop/OG1_Data/test_data"
OUTPUT_DIRECTORY = "/Users/orlpru/Desktop/OG1_Data/test_output"
FILE_EXTENSION = ".nc"

# Calculates the deep dark value using the deepest 10% of the profile 
# or a maximum of 550m, whichever is shallower.
CHLA_DEPTH_RATIO = 0.9 
DEFAULT_CHLA_DEPTH = 550

BASE_CONFIG_YAML = """
pipeline:
  name: Example CTD Processing Pipeline
  description: A pipeline for processing CTD data
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
            depth_range: [-1000, 0]
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
      plot_profiles_in_range: [100, 150]
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
      plot_profiles: [101, 200, 201, 300, 301, 400]
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

def generate_file_config(input_path: Path, output_path: Path, config_path: Path) -> None:
    config = yaml.safe_load(BASE_CONFIG_YAML)
    config["steps"][0]["parameters"]["file_path"] = str(input_path)
    config["steps"][-1]["parameters"]["output_path"] = str(output_path)
    
    with xr.open_dataset(input_path) as ds:
        has_location = "LATITUDE" in ds.variables and "LONGITUDE" in ds.variables
        
        max_depth = 1000 
        if "DEPTH" in ds.variables:
            max_depth = float(ds["DEPTH"].max())
        elif "PRES" in ds.variables:
            max_depth = float(ds["PRES"].max())

    # Safely remove all location-based QC if coordinates are missing
    if not has_location:
        qc_settings = config["steps"][1]["parameters"]["qc_settings"]
        for test in ["impossible location qc", "position on land qc", "impossible speed qc"]:
            qc_settings.pop(test, None)
            
    # Dynamically scale deep CHLA threshold
    # Taking the max ensures we pick the shallower (less negative) value
    calculated_threshold = -int(max_depth * CHLA_DEPTH_RATIO)
    final_threshold = max(-DEFAULT_CHLA_DEPTH, calculated_threshold)
    
    for step in config["steps"]:
        if step["name"] == "Chla Deep Correction":
            step["parameters"]["depth_threshold"] = final_threshold

    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

def run_pipeline(config_path: Path) -> str:
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        p = Pipeline()
        p.global_parameters = config.get("pipeline", {})
        p.logger = _setup_logging() 
        p.build_steps(config.get("steps", []))
        
        p.run()
        return "Success"
        
    except FileNotFoundError:
        return f"Failed: Could not find config at {config_path.name}"
    except yaml.YAMLError as exc:
        return f"Failed: Error parsing YAML file. {exc}"
    except Exception as e:
        return f"Failed: {e}"

def process_folder(input_dir: str, output_dir: str) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    successful = []
    failed = []
    
    for input_file in in_dir.glob(f"*{FILE_EXTENSION}"):
        file_stem = input_file.stem
        
        output_file = out_dir / f"{file_stem}_Processed.nc"
        config_file = out_dir / f"{file_stem}_config.yaml"
        log_file = out_dir / f"{file_stem}_log.txt"
        
        print(f"Processing {input_file.name}...", end=" ", flush=True)
        
        generate_file_config(input_file, output_file, config_file)
        
        with open(log_file, 'w') as f:
            # Redirecting stderr here captures the pipeline logs and progress bars
            with redirect_stdout(f), redirect_stderr(f):
                status = run_pipeline(config_file)
        
        print(status)
        
        if status == "Success":
            successful.append(input_file.name)
        else:
            failed.append((input_file.name, status))

    print("\n--- Batch Processing Summary ---")
    print(f"Total processed: {len(successful) + len(failed)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    for filename, error_msg in failed:
        print(f"  * {filename}: {error_msg}")

if __name__ == "__main__":
    process_folder(INPUT_DIRECTORY, OUTPUT_DIRECTORY)