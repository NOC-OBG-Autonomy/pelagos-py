import sys
import numpy as np
from toolbox.pipeline import Pipeline, _setup_logging

# --- Configuration Variables ---
FILE_PATH = "/Users/orlpru/Desktop/OG1_Data/Growler_677_R.nc"
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
          "name": "Find Profiles Beta",
          "diagnostics": True  # Set to False so the plot doesn't block the print statements
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
    
    # --- Profile Validation ---
    print("\n" + "="*40)
    print("PROFILE_NUMBER VALIDATION")
    print("="*40)
    
    data = p.get_data()
    
    if "PROFILE_NUMBER" in data:
        prof_var = data["PROFILE_NUMBER"]
        prof_vals = prof_var.values
        
        # Isolate the actual numbers from the NaNs
        valid_vals = prof_vals[~np.isnan(prof_vals)]
        
        print(f"Memory Data Type (dtype) : {prof_vals.dtype}")
        print(f"Xarray Saved Encoding    : {prof_var.encoding}")
        
        if len(valid_vals) > 0:
            print(f"Minimum Value            : {np.min(valid_vals)}")
            print(f"Maximum Value            : {np.max(valid_vals)}")
        else:
            print("Result: All values are NaN.")
    else:
        print("Error: PROFILE_NUMBER was not generated.")
        
    print("="*40 + "\n")
    
except Exception as e:
    print(f"\nPipeline Stopped: {e}")