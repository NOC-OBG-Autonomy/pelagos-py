NOC Autonomy Toolbox Docs
=========================

Welcome!
--------
Thanks for checking out this software! The NOC (National Oceanography Centre) Autonomy Toolbox (pending a formal name - maybe "NOCAT"?) is a modular processing pipeline tool designed to operate on raw
OG1-like format glider/ALR data, outputting "science ready" datasets. The user interfaces with the tool through a single YAML config file, allowing for easy definition and dissemination 
of processing protocols. This allows academics to "standardise" their processing tools and easily share their methods by simply sharing raw data and config files.

As we hope this tool will be adopted by the glider community, effort has been made to implement a broad range of desirable processing steps, covering a large number of oceanographic 
variables which continue to grow. We are always welcome to new suggestions and for those who want to get their hands dirty, we have attempted to make implementation of custom steps as 
simple as possible. See :doc:`Contributing<contributing>` for more details.

Whilst the tool itself is written in python, interfacing with it requires little prior knowledge of the language. In most cases, processing is achieved through a few lines of code:

.. code-block:: python

   # Create the pipeline using the specified config
   Pipe = Pipeline(
      config_path=r"config.yaml"
   )
   Pipe.run()

The only involved part is defining your ``config.yaml`` which determines the details of how your raw data will be processed. YAML files are designed to be "human readable" so it should
be fairly intuitive to set them up. We have provided extensive details in ``examples/configs/all_step_configs.yaml`` which should help you - but if you see anything that doesn't 
make sense, let us know so we can improve it.

Installation
------------
Unfortunately, this software is not yet available as an installable package. We hope to change this soon.

If you would like to use in now however, you are welcome to clone/copy the 
repository onto your own device. The requirements for the toolbox can be installed using ``pip install -r requirements.txt``.

If using Anaconda, the ``environment.yaml`` file can be used to create a conda environment using:
``conda env create -f environment.yaml``.
This should include all of the packages required to operate the software and run the example notebooks.

Getting Started
---------------
If you are new here, we recommend that you check out the ``example/notebooks/pipeline_demo.ipynb`` jupyter notebook. This presents an example use case of processing CTD measurements from freely 
available glider data hosted by the British Oceanographic Data Centre (BODC). A fully commented config file is also used for this processing, so make sure you check it out!

The details of how each step works can be found in the documentation (see below). This is currently a work in progress so some steps may be missing or badly formatted. If you spot any mistakes 
or would like a specific steps documentation to be prioritized, please leave an issue in the `github issues page <https://github.com/NOC-OBG-Autonomy/toolbox/issues>`_.

Contents
--------
.. toctree::
   :maxdepth: 4
   :caption: Contents:

   installation
   usage
   api/src/toolbox/index
   contributing
