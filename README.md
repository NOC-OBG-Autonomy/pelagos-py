# Pelagos-py
Copyright 2025 The National Oceanography Centre and The Contributors

## Documentation
The documentation for this package is available [here](https://noc-obg-autonomy.github.io/pelagos-py/)
> Please note that the documentation is still under construction.

# About

Pelagos-py provides a flexible, modular pipeline framework for defining, executing, and visualising multi-step data-processing
workflows for oceanographic data.

Each pipeline is composed of a series of steps that are automatically built from a central user-defined YAML configuration file. 
As Pelagos-py only depends on the config file to construct a pipeline, processing can be easily reproduced by others through 
sharing of configs.

## Overview

The `Pipeline` class orchestrates the flow of data through a sequence of modular “steps.”
Each step performs a specific processing task (e.g., data loading, quality control, profile detection, export).

Key characteristics:
| Component                    | Description                                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Configuration-driven**     | Users define the workflow in a YAML file describing each step, its parameters, and diagnostics options. |
| **In-build Quality Control** | Pre-build quality control tests can be specified in the config by the user to flag bad data.            |
| **Diagnostics**              | Where possible, data is can be visualized to see the effect of each component of the pipeline.          |

### Installation

Pelagos-py is not *yet* and installable package, so for the moment you have to make a local copy to run it:
```bash
git clone https://github.com/NOC-OBG-Autonomy/pelagos-py.git
cd pelagos-py
# create/activate a virtual environment
pip install -e . 
```
See [Getting Started](https://noc-obg-autonomy.github.io/pelagos-py/getting_started.html) for more details.

## How to run

1. ### Initialization
   Import the 'Pipeline' class and create a pipeline using your config (see below for example)
   ```python
    from toolbox.pipeline import Pipeline
    pipeline = Pipeline(config_path="my_pipeline.yaml")
   ```
2. ### Pipeline Execution
   Running the pipeline executes each step defined by the config in order
   ```python
    results = pipeline.run()
   ```
3. ### Diagnostics & Visualization
   Steps can optionally include diagnostic plots or summaries by setting:
   ```yaml
    diagnostics: true
   ```
4. ### Exporting Pipeline Configuration
   The entire pipeline configuration can be exported to a YAML file for reproducibility:
   ```python
   pipeline.export_config("exported_pipeline.yaml")
   ```
## 🧩 Example Configuration
An example YAML configuration for a simple pipeline. See examples/notebooks/pipeline_demo.ipynb for a full demo
```yaml
 # Pipeline Configuration
 pipeline:
   name: Example CTD Processing Pipeline
   description: A pipeline for processing CTD data
   visualisation: false
 
 steps:
   - name: Load OG1
     parameters:
       file_path: ../examples/data/OG1/Nelson_646_R.nc # Path to the input NetCDF file
     diagnostics: false
 
   - name: Derive CTD
     parameters:
       to_derive: [
         DEPTH,
         PRAC_SALINITY,
         ABS_SALINITY,
         CONS_TEMP,
         DENSITY
       ]
     diagnostics: false
 
   - name: "Data Export"
     parameters:
       export_format: "netcdf"
       output_path: "../examples/data/OG1/Nelson_646_R_Processed.nc"
```

## 🔁 Extending the Pipeline
A full breakdown can be found here: [Developer Guide](https://noc-obg-autonomy.github.io/pelagos-py/developer_guide.html).

# License

[Apache 2.0 License](LICENSE)
