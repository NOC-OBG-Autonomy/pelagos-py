import sys
import yaml
import xarray as xr
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from toolbox.pipeline import Pipeline, _setup_logging

# --- Configuration Variables ---
# Strictly separate raw data from processed results
INPUT_DIRECTORY = "input"
OUTPUT_DIRECTORY = "output"

FILE_EXTENSION = ".nc"

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
        ctd qc: {}
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

def get_friendly_date_folder(base_path: Path) -> Path:
    """Generates a human-readable folder name like '9th Apr 2026' and handles duplicates."""
    now = datetime.now()
    
    # Add the proper ordinal suffix (st, nd, rd, th)
    day = now.day
    if 11 <= (day % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        
    date_str = f"{day}{suffix} {now.strftime('%b %Y')}"
    
    # Check for collisions and append [2], [3], etc.
    target_dir = base_path / date_str
    counter = 2
    while target_dir.exists():
        target_dir = base_path / f"{date_str} [{counter}]"
        counter += 1
        
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

def process_folder(input_dir: str, base_output_dir: str) -> None:
    in_dir = Path(input_dir).resolve()
    base_out_dir = Path(base_output_dir).resolve()
    
    # 1. Generate the human-readable master folder for this run
    run_out_dir = get_friendly_date_folder(base_out_dir)
    
    successful = []
    failed = []
    
    print(f"Starting batch run. Outputting to: {run_out_dir.name}\n")
    
    # 2. Use rglob to recursively find all files in subfolders
    for input_file in in_dir.rglob(f"*{FILE_EXTENSION}"):
        
        # Safety Check: Skip any files that are located inside your output directory
        if base_out_dir in input_file.parents:
            continue
            
        # 3. Calculate relative path to maintain folder structure (e.g. "BIO-Carbon")
        relative_subfolder = input_file.parent.relative_to(in_dir)
        file_stem = input_file.stem
        
        # 4. Create the specific output directory for this file (using original filename)
        file_out_dir = run_out_dir / relative_subfolder / file_stem
        file_out_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = file_out_dir / f"{file_stem}_Processed.nc"
        config_file = file_out_dir / f"{file_stem}_config.yaml"
        log_file = file_out_dir / f"{file_stem}_log.txt"
        
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