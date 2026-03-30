import sys
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
            "name": "Apply QC",
            "parameters": {
                "qc_settings": {
                    "impossible speed qc": {}
                }
            },
            "diagnostics": True
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