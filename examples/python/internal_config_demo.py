import sys
import numpy as np
from toolbox.pipeline import Pipeline, _setup_logging

# --- Configuration Variables ---
FILE_PATH = "/Users/orlpru/Desktop/OG1_Data/input/BIO-Carbon/Churchill_647.nc"
PIPELINE_NAME = "Minimal Test Pipeline"

config = {
    "pipeline": {
        "name": PIPELINE_NAME,
        "out_directory": "./",
        "visualisation": False
    },
    "steps": [
        {
            "name": "Load OG1",
            "parameters": {"file_path": FILE_PATH},
            "diagnostics": False
        },
        {
            "name": "Apply QC",
            "parameters": {
                "qc_settings": {
                    "ctd qc": {},
                    "impossible date qc": {},
                    "impossible location qc": {},
                    "position on land qc": {},
                }
             },
             "diagnostics": False
        },
{
            "name": "Interpolate Data",
            "parameters": {
                "qc_handling_settings": {
                    "flag_filter_settings": {
                        "LATITUDE": [3, 4, 9],
                        "LONGITUDE": [3, 4, 9]
                    },
   
                    "reconstruction_behaviour": "replace"
                }
            },
            "diagnostics": False
        },
        {
            "name": "Derive CTD",
            "parameters": {
                "to_derive": ["DEPTH", "PRAC_SALINITY", "ABS_SALINITY", "CONS_TEMP", "DENSITY"]
            },
            "diagnostics": False
        },
        {
          "name": "Find Profiles Beta",
          "diagnostics": False
        }
    ]
}

try:
    # Initialise empty pipeline
    p = Pipeline()
    
    # Apply configuration directly to memory, bypassing file creation
    p.global_parameters = config.get("pipeline", {})
    p.logger = _setup_logging() 
    p.build_steps(config.get("steps", []))
    
    # Execute pipeline
    p.run()
    
except Exception as e:
    print(f"\nPipeline Stopped: {e}")