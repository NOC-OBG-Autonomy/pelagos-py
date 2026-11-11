Quality Control
===============

Quality control (QC) are sub-steps run within the ``Apply QC`` process. These do not modify your data values: they update a corresponding QC column (e.g. ``TEMP_QC``) to indicate the reliability of each measurement.

Every QC check is run through the ``Apply QC`` step, which transfers the individual results onto the existing QC columns. A given check can only be called once per ``Apply QC``, but you can include multiple ``Apply QC`` steps in a single config.

QC checks come in two types:

* **Static** checks always operate on the same variable(s) and produce the same QC outputs — e.g. :doc:`Impossible Date <api/pelagos_py/steps/quality_control/impossible_date_qc/index>` always checks ``TIME``.
* **Dynamic** checks let the user choose which variables they apply to, so the QC output is not pre-determined — e.g. :doc:`Range <api/pelagos_py/steps/quality_control/range_qc/index>` and :doc:`Stuck Value <api/pelagos_py/steps/quality_control/stuck_value_qc/index>`.

To add your own QC check, see :doc:`development`.

pelagos-py follows the standardised Argo flagging system:

* **0**: No QC performed.
* **1**: Good data.
* **2**: Probably good data.
* **3**: Probably bad data (potentially correctable).
* **4**: Bad data (not correctable).
* **5**: Value changed (only CHLA Quenching sets this, as in Argo).
* **8**: Interpolated value.
* **9**: Missing value.

Automated QC tests flag values that pass with the pipeline's ``pass_flag`` setting: 1 by default, 2 (probably good) in the default config, so that only a manual Visual QC check marks data good (1). Flags of 1 already in the input file are kept.

Steps leave samples flagged 3, 4 or 9 out of their calculations (the samples are still corrected). Chlorophyll is the exception: raw ``CHLA`` is flagged 3 until it is corrected, so steps only leave out ``CHLA`` samples flagged 4 or 9.

Flags are merged so the worse one wins: a sample flagged 3 stays 3 when a later check says 2. Chlorophyll is the exception the default config makes on purpose. Raw ``CHLA`` always needs correcting, so ``CHLA_QC`` is 3, or 4 outside the global range. After Deep Correction and CHLA Quenching (which flags the values it changes 5), a final range test with ``reset_flags: [3, 5]`` re-assesses ``CHLA_ADJUSTED_QC``, so it ends as 2 (probably good) or 4 (bad).

Configuration
-------------

QC is defined inside the ``qc_settings`` block of the ``Apply QC`` step. You can specify parameters for each QC, such as thresholds or which variables to target.

.. code-block:: yaml

   - name: "Apply QC"
     parameters:
       qc_settings:
         "range qc":
           variable_ranges: {"TEMP": {4: [-2, 35, "outside"]}}
           also_flag: {"TEMP": ["CNDC"]}

Variable names resolve to their ``_ADJUSTED`` version when one exists (see :doc:`steps`), so the test above flags ``TEMP_ADJUSTED`` if a correction has already produced it, and ``TEMP`` otherwise.

Available QC
------------

Spatiotemporal Checks
~~~~~~~~~~~~~~~~~~~~~

These verify that the data was collected at a logical time and place.

* :doc:`Impossible Date <api/pelagos_py/steps/quality_control/impossible_date_qc/index>`: Flags times before 1985-01-01 or after the current time (see :doc:`user guide <user_guide/Pelagos-py_POSITION>`).
* :doc:`Impossible Location <api/pelagos_py/steps/quality_control/impossible_location_qc/index>`: Flags latitudes outside ±90° and longitudes outside ±180° (see :doc:`user guide <user_guide/Pelagos-py_POSITION>`).
* :doc:`Impossible Speed <api/pelagos_py/steps/quality_control/impossible_speed_qc/index>`: Flags GPS fixes that imply an unphysical horizontal speed (see :doc:`user guide <user_guide/Pelagos-py_POSITION>`).
* :doc:`Position on Land <api/pelagos_py/steps/quality_control/position_on_land_qc/index>`: Flags positions on land, using the Natural Earth 1:110m land polygons (see :doc:`user guide <user_guide/Pelagos-py_POSITION>`).

Range and Value Checks
~~~~~~~~~~~~~~~~~~~~~~

These identify data points that fall outside expected physical or sensor limits.

* :doc:`Range <api/pelagos_py/steps/quality_control/range_qc/index>`: Flags values by range, per variable. Each band carries an ``inside``/``outside`` keyword: ``[low, high, "outside"]`` is a band of good values (data outside it is flagged), while ``[low, high, "inside"]`` is an impossible band (data within it is flagged). A flag may list several bands. If the keyword is omitted the bound order is the fallback (ascending → outside, descending → inside). A single scalar flags exact matches (e.g. fill values). ``reset_flags`` lists existing flags the test replaces instead of merging with, so a re-test after a correction can turn a 3 into a 2.
* :doc:`Visual <api/pelagos_py/steps/quality_control/visual_qc/index>`: Flags samples inside (or outside) hand-drawn boxes on an ``x_variable``/``y_variable`` plot — each box is a 2D range test with its own flag and target variables.
* :doc:`Stuck Value <api/pelagos_py/steps/quality_control/stuck_value_qc/index>`: Identifies sensor "freezing" by looking for sequences of identical values where variation is expected.
* :doc:`Spike <api/pelagos_py/steps/quality_control/spike_qc/index>`: Flags spikes: values that depart from a rolling median by more than a threshold.
* :doc:`PAR Irregularity <api/pelagos_py/steps/quality_control/par_irregularity_qc/index>`: A specialised check for PAR (photosynthetically active radiation) sensors that flags irregular light readings (see :doc:`user guide <user_guide/Pelagos-py_PAR>`).

Profile Integrity
~~~~~~~~~~~~~~~~~

These assess the quality of a vertical profile as a whole.

* :doc:`Valid Profile <api/pelagos_py/steps/quality_control/valid_profile_qc/index>`: Flags whole profiles that are too short or never reach a target depth range (see :doc:`user guide <user_guide/Pelagos-py_PROFILES>`).
* :doc:`Flag Full Profile <api/pelagos_py/steps/quality_control/flag_full_profile/index>`: Flags an entire profile once it accumulates too many bad measurements.