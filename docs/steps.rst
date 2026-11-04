Steps
=====

Steps are the individual operations that make up a pipeline. They are executed in the order they appear in your configuration file. Each step can be customised with parameters to suit your specific dataset.

A step is defined in Python and configured through the pipeline config file. Examples include:

* **I/O steps** — e.g. reading in data with :doc:`Load OG1 <api/pelagos_py/steps/input_output/load_data/index>`.
* **Variable processing steps** — e.g. adjusting existing salinity measurements with the :doc:`Salinity Adjustment <api/pelagos_py/steps/processing/salinity/index>` step.

Steps are not limited to one per file — a single module can define multiple steps — and any step can be called multiple times within a single config.

Some steps can filter out bad data before processing, using the QC handling mechanism. The pipeline preserves the original dataset dimensions, so filtered data is either replaced or reinserted once the step completes. For how to build a step with this behaviour, see :doc:`development`.

Raw and Adjusted Variables
--------------------------

Variables loaded from the input file are never overwritten. A step that changes the values of an existing variable, such as ``CHLA``, writes the result to ``CHLA_ADJUSTED`` instead, and every variable keeps its own QC column (``CHLA_QC`` and ``CHLA_ADJUSTED_QC``). The derived CTD variables (``PSAL``, ``ASAL``, ``CONS_TEMP``, ``DENSITY`` and ``DEPTH``) are always written with the ``_ADJUSTED`` suffix, so a value shipped in the file stays alongside the one the pipeline computed. The final oxygen product is ``DOXY_ADJUSTED``.

Steps and QC tests read the ``_ADJUSTED`` version of a variable when it exists, and the raw one otherwise. This means a config can name ``PSAL`` and still use ``PSAL_ADJUSTED`` once a step has produced it, and corrections stack: a second correction of ``CHLA`` is applied to ``CHLA_ADJUSTED``. Because of this the order of the steps matters. Before running, the pipeline warns when a step reads a raw variable whose ``_ADJUSTED`` version is only produced by a later step, and when a config names an ``_ADJUSTED`` variable that will not exist yet, so the raw one is used instead.

Some steps make working variables along the way, for example the intermediate stages of the oxygen processing. ``Data Export`` leaves these out of the exported file unless ``include_intermediates: true`` is set.

Input and Output Steps
----------------------

Loading, preparing, checking and exporting data.

* :doc:`Load OG1 <api/pelagos_py/steps/input_output/load_data/index>`: Loads an OG1 NetCDF file and removes records outside the deployment time window (see :doc:`user guide <user_guide/Pelagos-py_POSITION>`).
* :doc:`Prepare OG1 <api/pelagos_py/steps/input_output/prepare_og1/index>`: Normalises a loaded file's variable names and units onto what the other steps expect.
* :doc:`Format Checker <api/pelagos_py/steps/input_output/format_check/index>`: Runs the IOOS compliance checker on the file and reports the result.
* :doc:`Correct Values <api/pelagos_py/steps/input_output/correct_values/index>`: Applies a linear correction (slope and intercept) to a variable, e.g. a cross-calibration, unit conversion or rename (see :doc:`user guide <user_guide/Pelagos-py_CROSSCAL>`).
* :doc:`Data Export <api/pelagos_py/steps/input_output/export/index>`: Saves the processed dataset to file.
* :doc:`Write Data Report <api/pelagos_py/steps/input_output/write_report/index>`: Writes a PDF report summarising the data, plots and QC at the end of a pipeline.
* :doc:`Generate Data <api/pelagos_py/steps/input_output/gen_data/index>`: Generates random synthetic data for development and testing.

General Processing Steps
------------------------

Steps that are not tied to one sensor.

* :doc:`Apply QC <api/pelagos_py/steps/quality_control/apply_qc/index>`: Runs one or more QC tests and merges their flags (see :doc:`quality_control`).
* :doc:`Derive CTD <api/pelagos_py/steps/processing/derive_ctd/index>`: Derives depth, practical and absolute salinity, conservative temperature and density with TEOS-10 (see :doc:`user guide <user_guide/Pelagos-py_DERIVED>`).
* :doc:`Find Profiles <api/pelagos_py/steps/processing/find_profiles/index>`: Labels each sample with an OG1 phase and numbers profiles and dive cycles (see :doc:`user guide <user_guide/Pelagos-py_PROFILES>`).
* :doc:`Interpolate Data <api/pelagos_py/steps/processing/interpolate_data/index>`: Fills short gaps in variables by linear interpolation in time (see :doc:`user guide <user_guide/Pelagos-py_PRESSURE>`).
* :doc:`Mixed Layer Depth <api/pelagos_py/steps/processing/mixed_layer_depth/index>`: Calculates the mixed layer depth of each profile with a density or temperature threshold (see :doc:`user guide <user_guide/Pelagos-py_MLD>`).
* :doc:`Deep Correction <api/pelagos_py/steps/processing/deep_correction/index>`: Subtracts a deep dark offset from a profile variable (e.g. chlorophyll fluorescence) (see :doc:`user guide <user_guide/Pelagos-py_CHLA>`).

Variable-specific Steps
-----------------------

**Salinity**

* :doc:`Salinity Adjustment <api/pelagos_py/steps/processing/salinity/index>`: Corrects the conductivity-temperature lag and thermal-mass error, and derives ``PSAL_ADJUSTED`` (see :doc:`user guide <user_guide/Pelagos-py_SALINITY>`).

**Oxygen** (all in the :doc:`oxygen module <api/pelagos_py/steps/processing/oxygen/index>`; see the :doc:`user guide <user_guide/Pelagos-py_DOXY>`)

* **Derive Uncalibrated Phase**: Derives the optode's uncalibrated phase from the blue (and red) phase.
* **Derive Optode Temperature**: Derives optode temperature from its temperature voltage.
* **Phase Pressure Correction**: Corrects the uncalibrated phase for pressure.
* **Shift Oxygen To CTD**: Time-shifts optode measurements onto the CTD sampling, correcting for the sensors' separation.
* **Derive Calibrated Phase**: Derives the calibrated phase from the uncalibrated phase.
* **Derive Oxygen Concentration**: Converts calibrated phase to molar oxygen concentration.
* **Molar DOXY Salinity Correction**: Corrects molar oxygen for salinity.
* **Molar DOXY Pressure Correction**: Corrects molar oxygen for pressure.
* **Derive DOXY**: Converts molar oxygen (µmol/L) to ``DOXY_ADJUSTED`` (µmol/kg).

**PAR**

* :doc:`Interpolate PAR <api/pelagos_py/steps/processing/interpolate_par/index>`: Derives the euphotic depth (``ZEU``) and iPAR isolume depth (``Z_IPAR``) per profile (see :doc:`user guide <user_guide/Pelagos-py_PAR>`).

**Chlorophyll-a**

* :doc:`CHLA Quenching <api/pelagos_py/steps/processing/chla_quenching/index>`: Corrects non-photochemical quenching of chlorophyll fluorescence (see :doc:`user guide <user_guide/Pelagos-py_CHLA>`).

**Backscatter** (both in the :doc:`bbp module <api/pelagos_py/steps/processing/bbp/index>`; see the :doc:`user guide <user_guide/Pelagos-py_BBP>`)

* **BBP from Beta**: Converts measured beta backscatter to the particulate backscattering coefficient (BBP).
* **Isolate BBP Spikes**: Splits a backscatter record into a smooth baseline and isolated spikes.

Quality Control (QC)
--------------------

Quality Control is handled differently from other steps. Rather than performing a single calculation, the ``Apply QC`` step acts as a manager for multiple sub-steps called **QC**.

When you add ``Apply QC`` to your configuration, you define a list of QC (such as range checks or spike tests) within its parameters. This step is responsible for:

1. **Running QC**: Executing individual quality checks on your variables.
2. **Merging Flags**: Combining results from multiple QC into a single QC column using standardised Argo flagging logic.
3. **Diagnostics**: Generating visualisations to show which data points were flagged and why.

For a full list of available QC and how to configure them, please refer to the :doc:`quality_control` page.
