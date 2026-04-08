Steps
=====

Steps are the individual operations that make up a pipeline. They are executed in the order they appear in your configuration file. Each step can be customised with parameters to suit your specific dataset.

Non-specific Steps
------------------

These steps are general utility tools used for data management, preparation, and export. They are not tied to a specific sensor or variable.

* :doc:`Apply QC <api/src/toolbox/steps/custom/apply_qc/index>`: A container step used to run various quality control.
* :doc:`Derive CTD <api/src/toolbox/steps/custom/derive_ctd/index>`: Calculates derived physical properties such as density or potential temperature.
* :doc:`Export <api/src/toolbox/steps/custom/export/index>`: Saves the final processed dataset to a specified format.
* :doc:`Find Profiles <api/src/toolbox/steps/custom/find_profiles/index>`: Segments a time series into individual vertical profiles.
* :doc:`Generate Data <api/src/toolbox/steps/custom/gen_data/index>`: Creates synthetic data for testing and validation purposes.
* :doc:`Interpolate Data <api/src/toolbox/steps/custom/interpolate_data/index>`: Fills gaps in variables using mathematical interpolation.
* :doc:`Load Data <api/src/toolbox/steps/custom/load_data/index>`: The entry point for importing data into the toolbox.
* :doc:`Find Profile Direction <api/src/toolbox/steps/custom/profile_direction/index>`: Identifies whether data was collected during an ascent or descent.

Variable Specific Steps
-----------------------

These steps contain logic tailored to specific oceanographic instruments or variables.

* :doc:`CTD <api/src/toolbox/steps/custom/variables/salinity/index>`: Specialised adjustments for Conductivity, Temperature, and Depth data.
* :doc:`Oxygen <api/src/toolbox/steps/custom/variables/oxygen/index>`: Corrections and calibrations for dissolved oxygen sensors.
* :doc:`Chlorophyll <api/src/toolbox/steps/custom/variables/chla/index>`: Processing steps for fluorescence and chlorophyll-a concentration.
* :doc:`Backscatter <api/src/toolbox/steps/custom/variables/bbp/index>`: Optical backscatter processing and scaling.

Quality Control (QC)
--------------------

Quality Control is handled differently from other steps. Rather than performing a single calculation, the ``Apply QC`` step acts as a manager for multiple sub-steps called **QC**.

When you add ``Apply QC`` to your configuration, you define a list of QC (such as range checks or spike tests) within its parameters. This step is responsible for:

1. **Running QC**: Executing individual quality checks on your variables.
2. **Merging Flags**: Combining results from multiple QC into a single QC column using standardised Argo flagging logic.
3. **Diagnostics**: Generating visualisations to show which data points were flagged and why.

For a full list of available QC and how to configure them, please refer to the :doc:`qc` page.