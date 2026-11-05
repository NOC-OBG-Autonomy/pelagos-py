import numpy as np
import xarray as xr

from pelagos_py.steps.processing.interpolate_data import InterpolateVariables


def test_saved_config_can_rebuild_the_step():
    data = xr.Dataset(
        {
            "TEMP": ("N_MEASUREMENTS", np.array([1.0, np.nan, 3.0])),
            "TEMP_QC": ("N_MEASUREMENTS", np.array([1, 9, 1], dtype=np.int8)),
        }
    )
    context = {"data": data}
    step = InterpolateVariables(
        "Interpolate Data", {"variables": {"TEMP": [4]}}, context=context
    )
    config = step.generate_config()
    assert "qc_handling_settings" not in config["parameters"]
    InterpolateVariables("Interpolate Data", config["parameters"], context=context)
