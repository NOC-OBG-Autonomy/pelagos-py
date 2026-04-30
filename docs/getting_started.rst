Getting Started
===============

Installation
------------

The NOC Autonomy Toolbox, **pelagos-py**, is not yet available as an installable package via PyPI. To use the software, you should clone the repository directly from GitHub.

You can choose between the stable version or the latest development features.

**Stable Version (Main)**

This is the recommended version for most users. It contains the most thoroughly tested code.

.. code-block:: bash

   git clone https://github.com/NOC-OBG-Autonomy/pelagos-py.git

**Development Version (Dev)**

If you want to access the latest features or contribute to the project, use the development branch. Note that this code may be subject to frequent changes.

.. code-block:: bash

   git clone -b dev https://github.com/NOC-OBG-Autonomy/pelagos-py.git

Dependencies
------------

Once you have cloned the repository, navigate into the project folder to install the required dependencies.

**Using pip**

.. code-block:: bash

   pip install -r requirements.txt

**Using Anaconda**

If you prefer using conda, you can create a dedicated environment using the provided ``environment.yaml`` file. This environment includes all packages required to run the toolbox and the example notebooks.

.. code-block:: bash

   conda env create -f environment.yaml

Example Pipeline
----------------

If you are new here, we recommend checking out the ``example/notebooks/pipeline_demo.ipynb`` Jupyter notebook. This provides an example use case for processing CTD measurements from glider data hosted by the British Oceanographic Data Centre (BODC). 

A fully commented configuration file is used for this process, which serves as an excellent template for your own projects.

Documentation and Feedback
--------------------------

The details of how each step works can be found in the following sections of this documentation. Please note that this is a work in progress. Some steps may be missing or require further formatting. 

If you spot any mistakes or would like specific documentation to be prioritised, please open an issue on the `GitHub issues page <https://github.com/NOC-OBG-Autonomy/pelagos-py/issues>`_.