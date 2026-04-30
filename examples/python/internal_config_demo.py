import yaml
from toolbox.pipeline import Pipeline, _setup_logging

# --- Configuration Variables ---
INPUT_FILE = "/Users/orlpru/Desktop/OG1_Data/input/BIO-Carbon/Cabot_645.nc"

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

  - name: Find Profiles
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
    diagnostics: true

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
"""

try:
    config = yaml.safe_load(BASE_CONFIG_YAML)
    
    for step in config.get("steps", []):
        if step.get("name") == "Load OG1":
            step["parameters"]["file_path"] = INPUT_FILE

    p = Pipeline()
    p.global_parameters = config.get("pipeline", {})
    p.logger = _setup_logging() 
    p.build_steps(config.get("steps", []))
    
    p.run()
    
except Exception as e:
    print(f"\nPipeline Stopped: {e}")