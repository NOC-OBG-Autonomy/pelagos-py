Getting Started
===============

pelagos-py provides a modular pipeline framework for defining, running and visualising multi-step processing workflows for oceanographic data.

Each pipeline is a series of steps built from a single YAML configuration file. Because the pipeline depends only on the config, processing can be reproduced by sharing the config.

Installation
------------

Install **pelagos-py** with pip:

.. code-block:: bash

   pip install pelagos-py

Or install from source:

.. code-block:: bash

   git clone https://github.com/NOC-OBG-Autonomy/pelagos-py.git
   cd pelagos-py
   pip install -e .

Both install all required dependencies.

**Using Anaconda**

If you prefer conda, create an environment from the provided ``environment.yaml``. It includes everything needed to run pelagos-py and the example notebooks.

.. code-block:: bash

   conda env create -f environment.yaml

Overview
--------

The ``Pipeline`` class runs data through a sequence of modular steps. Each step performs one processing task (e.g. data loading, quality control, profile detection, export).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Description
   * - Configuration-driven
     - Users define the workflow in a YAML file describing each step, its parameters, and diagnostics options.
   * - Built-in quality control
     - Pre-built quality control tests can be specified in the config to flag bad data.
   * - Diagnostics
     - Where possible, data can be visualised to see the effect of each step of the pipeline.

How to Run
----------

**Initialisation**

Import the ``Pipeline`` class and create a pipeline, either by letting pelagos-py build a config
for your OG1 file or from a config you already have:

.. code-block:: python

   from pelagos_py import Pipeline

   # Build a config for the file from the default template, saved as glider.yaml next to it
   pipeline = Pipeline.make_config("glider.nc")

   # Same, but ask about each choice (e.g. what to do about missing PAR) instead of using the defaults
   pipeline = Pipeline.make_config("glider.nc", ask=True)

   # Or load a config you already have, e.g. glider.yaml after editing it
   pipeline = Pipeline.load_config("my_pipeline.yaml")

To try it without your own data, ``get_demo_file`` downloads a demo OG1 file (a month of NOC glider data from the North Atlantic, ~35 MB) into
``examples/data/OG1`` (only the first time) and returns its path. ``examples/python/run_demo.py`` does this end to end:

.. code-block:: python

   from pelagos_py import Pipeline, get_demo_file

   pipeline = Pipeline.make_config(get_demo_file())

**Pipeline Execution**

Running the pipeline executes each step defined by the config in order:

.. code-block:: python

   pipeline.run()
   data = pipeline.get_data()

**Diagnostics & Visualisation**

Steps can optionally include diagnostic plots or summaries by setting ``diagnostics: true`` in the config.

**Exporting Pipeline Configuration**

The entire pipeline configuration can be exported to a YAML file for reproducibility:

.. code-block:: python

   pipeline.export_config("exported_pipeline.yaml")

Example Configuration
---------------------

A minimal YAML configuration for the demo file. See ``examples/notebooks/pipeline_demo.ipynb`` for a full demo.

.. code-block:: yaml

   # Pipeline Configuration
   pipeline:
     name: Example CTD Processing Pipeline
     description: A pipeline for processing CTD data

   steps:
     - name: Load OG1
       parameters:
         file_path: examples/data/OG1/demo_OG1.nc
       diagnostics: false

     - name: Derive CTD
       parameters:
         # written as DEPTH_ADJUSTED, PSAL_ADJUSTED, ...
         to_derive: [
           DEPTH,
           PSAL,
           ASAL,
           CONS_TEMP,
           DENSITY
         ]
       diagnostics: false

     - name: "Data Export"
       parameters:
         export_format: "netcdf"
         output_path: "examples/data/OG1/demo_OG1_Processed.nc"
         include_intermediates: false # true also exports working variables

Example Pipeline
----------------

If you are new here, we recommend checking out the ``examples/notebooks/pipeline_demo.ipynb`` Jupyter notebook, or running ``examples/python/run_demo.py``. This provides an example use case for processing CTD measurements from the demo glider data.

Both build a fully commented configuration file for the demo data, which serves as an excellent template for your own projects.

Documentation and Feedback
--------------------------

The details of how each step works can be found in the :doc:`user_guide` and the :doc:`api_reference`.

If you spot any mistakes or would like specific documentation to be prioritised, please open an issue on the `GitHub issues page <https://github.com/NOC-OBG-Autonomy/pelagos-py/issues>`_.
