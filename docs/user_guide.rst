User Guide
==========

Detailed explanations of the processing steps and quality control tools.

pelagos-py is built around two extensible concepts: **steps**, the stages that make up a pipeline, and **QC checks**, the sub-stages that flag data reliability. Both are defined in Python and configured through your YAML config file. The pages below describe the ones that ship with pelagos-py; to add your own, see :doc:`development`.

----

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :text-align: center

      .. raw:: html

         <i class="fa-solid fa-list-check card-icon"></i>
         <span class="card-title">Steps</span>

      The individual operations that make up a pipeline — what they do and how to configure them.

      .. button-ref:: steps
         :ref-type: doc
         :expand:
         :color: primary
         :click-parent:

         To the steps guide

   .. grid-item-card::
      :text-align: center

      .. raw:: html

         <i class="fa-solid fa-magnifying-glass card-icon"></i>
         <span class="card-title">Quality Control</span>

      Sub-steps that flag data reliability without modifying values — how QC works and how to configure it.

      .. button-ref:: quality_control
         :ref-type: doc
         :expand:
         :color: primary
         :click-parent:

         To the QC guide

.. toctree::
   :maxdepth: 2
   :hidden:

   steps
   quality_control

.. toctree::
   :maxdepth: 1
   :caption: Variable processing
   :hidden:

   Position and time <user_guide/Pelagos-py_POSITION>
   Pressure <user_guide/Pelagos-py_PRESSURE>
   Profile finding <user_guide/Pelagos-py_PROFILES>
   Temperature <user_guide/Pelagos-py_TEMP>
   Salinity <user_guide/Pelagos-py_SALINITY>
   Derived CTD variables <user_guide/Pelagos-py_DERIVED>
   Mixed layer depth <user_guide/Pelagos-py_MLD>
   Oxygen <user_guide/Pelagos-py_DOXY>
   PAR <user_guide/Pelagos-py_PAR>
   Chlorophyll-a <user_guide/Pelagos-py_CHLA>
   Backscatter <user_guide/Pelagos-py_BBP>
   Cross-calibration <user_guide/Pelagos-py_CROSSCAL>
