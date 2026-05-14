Quality Control
=====================

Quality control (QC) are sub-steps run within the ``Apply QC`` process. These do not modify your data values: they update a corresponding QC column (e.g. ``TEMP_QC``) to indicate the reliability of each measurement.

The Toolbox follows the standardised Argo flagging system:

* **0**: No QC performed.
* **1**: Good data.
* **2**: Probably good data.
* **3**: Probably bad data (potentially correctable).
* **4**: Bad data (not correctable).
* **9**: Missing value.

Configuration
-------------

QC os defined inside the ``qc_settings`` block of the ``Apply QC`` step. You can specify parameters for each QC, such as thresholds or which variables to target.

.. code-block:: yaml

   - name: "Apply QC"
     parameters:
       qc_settings:
         "impossible range qc":
           variable_ranges: {"TEMP": {4: [-2, 35]}}
           also_flag: {"TEMP": ["CNDC"]}

Available QC
---------------

Spatiotemporal Checks
~~~~~~~~~~~~~~~~~~~~~

These verify that the data was collected at a logical time and place.

* :doc:`Impossible Date <api/src/toolbox/steps/custom/qc/impossible_date_qc/index>`: Checks that the timestamps fall within a realistic range (typically from 1985 to the present day).
* :doc:`Impossible Location <api/src/toolbox/steps/custom/qc/impossible_location_qc/index>`: Verifies that latitude and longitude coordinates are within global bounds.
* :doc:`Impossible Speed <api/src/toolbox/steps/custom/qc/impossible_speed_qc/index>`: Calculates the velocity between points to ensure the platform is not moving at unphysical speeds.
* :doc:`Position on Land <api/src/toolbox/steps/custom/qc/position_on_land_qc/index>`: Uses a bathymetry mask to check if coordinates incorrectly place the platform on land.

Range and Value Checks
~~~~~~~~~~~~~~~~~~~~~~

These identify data points that fall outside expected physical or sensor limits.

* :doc:`Gross Range <api/src/toolbox/steps/custom/qc/gross_range_qc/index>`: Applies broad, non-configurable physical limits to catch extreme sensor failures.
* :doc:`Impossible Range <api/src/toolbox/steps/custom/qc/impossible_range_qc/index>`: Allows users to define specific, narrower thresholds for any variable.
* :doc:`Stuck Value <api/src/toolbox/steps/custom/qc/stuck_value_qc/index>`: Identifies sensor "freezing" by looking for sequences of identical values where variation is expected.
* :doc:`Spike <api/src/toolbox/steps/custom/qc/spike_qc/index>`: Detects sudden, unrealistic jumps in data values between adjacent measurements.
* :doc:`PAR Irregularity <api/src/toolbox/steps/custom/qc/par_irregularity_qc/index>`: A specialised check for Photosynthetically Active Radiation sensors to identify inconsistent light readings.

Profile Integrity
~~~~~~~~~~~~~~~~~

These assess the quality of a vertical profile as a whole.

* :doc:`Valid Profile <api/src/toolbox/steps/custom/qc/valid_profile_qc/index>`: Ensures a profile contains enough data points or covers a sufficient depth range to be useful.
* :doc:`Flag Full Profile <api/src/toolbox/steps/custom/qc/flag_full_profile/index>`: If a certain percentage of points in a profile are flagged as bad, this qc can be configured to flag the entire profile.