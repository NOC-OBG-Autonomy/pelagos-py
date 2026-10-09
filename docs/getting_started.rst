Getting Started
===============

Pelagos-py provides a flexible, modular pipeline framework for defining, executing, and visualising multi-step data-processing workflows for oceanographic data.

Each pipeline is composed of a series of steps that are automatically built from a central user-defined YAML configuration file. As Pelagos-py only depends on the config file to construct a pipeline, processing can be easily reproduced by others through sharing of configs.

Installation
------------

The easiest way to install **pelagos-py** is via pip:

.. code-block:: bash

   pip install pelagos-py

Alternatively, you can install directly from the source using git:

.. code-block:: bash

   git clone https://github.com/NOC-OBG-Autonomy/pelagos-py.git
   cd pelagos-py
   pip install -e .

Dependencies
------------

Once you have cloned the repository, navigate into the project folder to install the required dependencies.

**Using pip**

.. code-block:: bash

   pip install -r requirements.txt

**Using Anaconda**

If you prefer using conda, you can create a dedicated environment using the provided ``environment.yaml`` file. This environment includes all packages required to run pelagos_py and the example notebooks.

.. code-block:: bash

   conda env create -f environment.yaml

Overview
--------

The ``Pipeline`` class orchestrates the flow of data through a sequence of modular steps. Each step performs a specific processing task (e.g., data loading, quality control, profile detection, export).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Description
   * - Configuration-driven
     - Users define the workflow in a YAML file describing each step, its parameters, and diagnostics options.
   * - In-built Quality Control
     - Pre-built quality control tests can be specified in the config by the user to flag bad data.
   * - Diagnostics
     - Where possible, data can be visualized to see the effect of each component of the pipeline.

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

To try it without your own data, ``get_demo_file`` downloads a demo OG1 file into
``~/Documents/pelagos-py/demo_data`` (only the first time) and returns its path; ``get_demo_file()`` lists the demos. ``examples/python/run_demo.py`` does this end to end:

.. code-block:: python

   from pelagos_py import Pipeline, get_demo_file

   pipeline = Pipeline.make_config(get_demo_file("nelson_646_r"))

**Pipeline Execution**

Running the pipeline executes each step defined by the config in order:

.. code-block:: python

   results = pipeline.run()

**Diagnostics & Visualisation**

Steps can optionally include diagnostic plots or summaries by setting ``diagnostics: true`` in the config.

**Exporting Pipeline Configuration**

The entire pipeline configuration can be exported to a YAML file for reproducibility:

.. code-block:: python

   pipeline.export_config("exported_pipeline.yaml")

Example Configuration
---------------------

A minimal YAML configuration for a simple pipeline. See ``examples/notebooks/pipeline_demo.ipynb`` for a full demo.

.. code-block:: yaml

   # Pipeline Configuration
   pipeline:
     name: Example CTD Processing Pipeline
     description: A pipeline for processing CTD data

   steps:
     - name: Load OG1
       parameters:
         file_path: ../examples/data/OG1/Nelson_646_R.nc
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

Example Pipeline
----------------

If you are new here, we recommend checking out the ``examples/notebooks/pipeline_demo.ipynb`` Jupyter notebook, or running ``examples/python/run_demo.py``. This provides an example use case for processing CTD measurements from glider data hosted by the British Oceanographic Data Centre (BODC).

Both build a fully commented configuration file for the demo data, which serves as an excellent template for your own projects.

Documentation and Feedback
--------------------------

The details of how each step works can be found in the following sections of this documentation. Please note that this is a work in progress. Some steps may be missing or require further formatting.

If you spot any mistakes or would like specific documentation to be prioritised, please open an issue on the `GitHub issues page <https://github.com/NOC-OBG-Autonomy/pelagos-py/issues>`_.