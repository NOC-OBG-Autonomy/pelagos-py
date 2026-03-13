# Pelagos-py
Copyright 2025-2026 The National Oceanography Centre and The Contributors

## Documentation
The documentation for this package is available [here](https://noc-obg-autonomy.github.io/pelagos-py/)
> Please note that the documentation is still under construction.

## Building Your Own Steps
[Click here](https://noc-obg-autonomy.github.io/pelagos-py/developer_guide.html) for a guide on how to build your own steps.

## Development Information
Python >= 3.10

### Installation

For a local, editable version of the toolbox

```bash
git clone https://github.com/NOC-OBG-Autonomy/pelagos-py.git
cd toolbox
# create/activate a virtual environment
pip install -e . 
```

# About
## 🧭 Pipeline Architecture

The *toolbox pipeline* provides a flexible, modular framework for defining, executing, and visualising multi-step data-processing workflows — such as those used for *autonomous underwater glider missions*.

Each pipeline is composed of a series of steps that are automatically discovered, registered, and executed in sequence. This enables users to build, extend, and visualise complex data workflows using a simple YAML configuration file.

## 🏗️ Overview

The `Pipeline` class orchestrates the flow of data through a sequence of modular “steps.”
Each step performs a specific processing task (e.g., data loading, quality control, profile detection, export).

Key characteristics:
| Component                | Description                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| **Configuration-driven** | Users define the workflow in a YAML file describing each step, its parameters, and diagnostics options. |
| **Dynamic discovery**    | New steps are auto-registered via decorators and discovered dynamically when imported.                  |
| **Composable**           | Steps can be nested to form sub-pipelines (e.g., a calibration block within a larger workflow).         |
| **Context-aware**        | Each step passes its results into a shared `context`, which is used as input for subsequent steps.      |
| **Visualisable**         | A Graphviz diagram can be automatically generated to visualise pipeline structure and dependencies.     |

## ⚙️ How It Works

1. ### Initialization
   ```python
    from toolbox.pipeline import Pipeline
    pipeline = Pipeline(config_path="my_pipeline.yaml")
    ```
    - Loads the pipeline configuration from YAML.
    - Discovers all available step classes using `@register_step`.
    - Validates step dependencies defined in `STEP_DEPENDENCIES`.
2. ### Step Discovery & Registration
   Each step inherits from `BaseStep` and registers itself via a decorator:
   ```python
   from toolbox.steps.base_step import register_step, BaseStep

    @register_step
    class LoadOG1(BaseStep):
        step_name = "Load OG1"

        def run(self):
            self.log("Loading NetCDF data...")
            # Do processing
            return self.context
    ```
    Registered steps are stored in a global registry (`REGISTERED_STEPS`) and automatically imported at runtime by:
    ```python
    toolbox.steps.discover_steps()
    ```
3. ### Pipeline Execution
   Running the pipeline executes each step in order, passing context forward:
    ```python
     results = pipeline.run()
     ```
    Internally:
    - Each step is instantiated via `create_step()`.
    - The `run()` method is called.
    - The returned context (e.g., processed `xarray.Dataset`) is merged and passed to the next step.
4. ### Diagnostics & Visualization
    Steps can optionally include diagnostic plots or summaries by setting:
    ```yaml
    diagnostics: true
    ```
    If the visualisation option is enabled, a Graphviz diagram of the pipeline is generated:
    ```yaml
    pipeline:
      visualize: true
    ```
    The pipeline renders a Graphviz diagram showing step dependencies and flow.
    ```css
    [Load OG1] → [Derive CTD*] → [Find Profiles] → [Data Export]
                      *diagnostics enabled
    ```
5. ### Exporting Pipeline Configuration
    The entire pipeline configuration can be exported to a YAML file for reproducibility:
    ```python
    pipeline.export_config("exported_pipeline.yaml")
    ```
## 🧩 Example Configuration
An example YAML configuration for a simple pipeline:
```yaml
pipeline:
  name: "Data Processing Pipeline"
  description: "Process and analyze multi-dimensional glider data"
  visualisation: false

steps:
  - name: "Load OG1"
    parameters:
      file_path: "../../examples/data/OG1/Doombar_648_R.nc"
      add_meta: false
    diagnostics: false

  - name: "Derive CTD"
    parameters:
      interpolate_latitude_longitude: true
    diagnostics: true

  - name: "Find Profiles"
    parameters:
      gradient_thresholds: [0.02, -0.02]
    diagnostics: false

  - name: "Data Export"
    parameters:
      export_format: "hdf5"
      output_path: "../../examples/data/OG1/exported_Doombar_648_R.nc"
```

## 🔁 Extending the Pipeline
A full breakdown can be found here: [Developer Guide](https://noc-obg-autonomy.github.io/pelagos-py/developer_guide.html).

TL;DR:
1. Create a file under `toolbox/steps/custom/variables` (e.g., `qc_salinity.py`).
2. Define a class that inherits from `BaseStep`.
3. Decorate it with `@register_step` and provide a unique step_name.
4. Implement a `run()` method that processes data and returns an updated context.
   
## 🧠 Key Takeaways

- Declarative workflow: Define what happens, not how it’s executed.
- Extensible design: Add new steps without modifying core code.
- Integrated diagnostics: Flag and visualise data quality issues inline.
- Portable & reproducible: YAML configurations make it easy to rerun or share pipelines.


# License

[Apache 2.0 License](LICENSE)
