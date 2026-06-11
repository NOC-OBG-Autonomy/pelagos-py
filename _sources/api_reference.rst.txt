API Reference
=============

This page documents the public API of pelagos_py. The library is organised into four main modules:

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`pipeline <api/pelagos_py/pipeline/index>`
     - The :class:`~pelagos_py.pipeline.Pipeline` class — entry point for building and running a processing pipeline from a YAML config.
   * - :doc:`pipeline_manager <api/pelagos_py/pipeline_manager/index>`
     - The :class:`~pelagos_py.pipeline_manager.PipelineManager` class — runs multiple pipelines in sequence and handles cross-calibration alignment between them.
   * - :doc:`steps <api/pelagos_py/steps/index>`
     - All built-in steps and the base classes used to define them.
   * - :doc:`utils <api/pelagos_py/utils/index>`
     - Shared utility functions for data alignment, QC handling, validation, and more.

.. toctree::
   :hidden:

   api/pelagos_py/pipeline/index
   api/pelagos_py/pipeline_manager/index
   api/pelagos_py/steps/index
   api/pelagos_py/utils/index
