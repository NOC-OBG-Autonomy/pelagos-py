import pytest
import yaml
import os
from pathlib import Path
from toolbox.pipeline import Pipeline

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"

PIPELINE_YAML = """
pipeline:
  name: Minimal Integration Pipeline
  description: A barebones pipeline to test file loading and exporting
  visualisation: false

steps:
  - name: Load OG1
    parameters:
      file_path: placeholder.nc 
    diagnostics: false

  - name: Apply QC
    parameters:
      qc_settings:
        impossible date qc: {}  
    diagnostics: false 

  - name: "Data Export"
    parameters:
      export_format: "netcdf"
      output_path: "placeholder_out.nc"
"""

def get_test_files():
    if not TEST_DATA_DIR.exists():
        return []
    return [str(f) for f in TEST_DATA_DIR.glob("*.nc")]

@pytest.mark.filterwarnings("ignore:.*monotonically increasing.*")
@pytest.mark.parametrize("file_path", get_test_files())
def test_full_pipeline_execution(tmp_path, file_path):
    
    config = yaml.safe_load(PIPELINE_YAML)
    
    config_file = tmp_path / "temp_pipeline.yaml"
    log_file = tmp_path / "test.log"
    output_nc = tmp_path / "processed_output.nc"

    config["pipeline"]["out_directory"] = str(tmp_path)
    config["pipeline"]["log_file"] = str(log_file)

    for step in config["steps"]:
        if step["name"] == "Load OG1":
            step["parameters"]["file_path"] = file_path
            
        elif step["name"] == "Data Export":
            step["parameters"]["output_path"] = str(output_nc)

    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    try:
        p = Pipeline(config_path=str(config_file))
        p.run()
        
        assert output_nc.exists(), "The pipeline completed, but the output file was not created."
        
    except Exception as e:
        pytest.fail(f"Pipeline stopped unexpectedly on {os.path.basename(file_path)}: {e}")