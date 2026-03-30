import yaml
from toolbox.pipeline import Pipeline, _setup_logging

# --- Configuration Variables ---
CONFIG_FILE_PATH = "/Users/orlpru/Desktop/OG1_Data/configs/example_config_Growler_677_R.yaml"

def run_pipeline(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            
        p = Pipeline()
        p.global_parameters = config.get("pipeline", {})
        p.logger = _setup_logging() 
        p.build_steps(config.get("steps", []))
        
        p.run()
        print(f"Successfully executed pipeline using {file_path}")
        
    except FileNotFoundError:
        print(f"\nPipeline Stopped: Could not find the file at {file_path}")
    except yaml.YAMLError as exc:
        print(f"\nPipeline Stopped: Error parsing YAML file. {exc}")
    except Exception as e:
        print(f"\nPipeline Stopped: {e}")

if __name__ == "__main__":
    run_pipeline(CONFIG_FILE_PATH)