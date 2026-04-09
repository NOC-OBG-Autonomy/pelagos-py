import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.apply_qc import ApplyQC

# Setup variables for easy tweaking
MOCK_ATTRS = {
    "long_name": "Sea Water Variable",
    "standard_name": "sea_water_variable"
}

class MockQC:
    """Dummy QC test for testing ApplyQC logic."""
    qc_name = "MockQC"
    required_variables = ["TEMP"]
    qc_outputs = ["TEMP_QC"]

    def __init__(self, data, **kwargs):
        self.data = data

    def return_qc(self):
        return xr.Dataset({
            "TEMP_QC": ("N_MEASUREMENTS", np.zeros(len(self.data["N_MEASUREMENTS"]), dtype=np.int8))
        })

    def plot_diagnostics(self):
        pass


@pytest.fixture
def apply_qc_step():
    step = ApplyQC("Apply QC")
    step.qc_settings = {"MockQC": {}}
    step.diagnostics = False
    step.context = {}
    
    step.check_data = lambda: None
    step.log = lambda x: None
    step.log_warn = lambda x: None
    
    return step


def test_sanitises_all_existing_qc_columns(apply_qc_step):
    """Checks both tested and untested variables keep valid flags and fix invalid ones."""
    data = xr.Dataset({
        # Tested variable: mix of valid, NaN, string, and out-of-bounds
        "TEMP": ("N_MEASUREMENTS", [10.0, 11.0, 12.0, 13.0], MOCK_ATTRS),
        "TEMP_QC": ("N_MEASUREMENTS", np.array([1, np.nan, "1b", 99], dtype=object)),
        
        # Untested variable: same mix of dirty data
        "PHASE": ("N_MEASUREMENTS", [1.0, 2.0, 3.0, 4.0], MOCK_ATTRS),
        "PHASE_QC": ("N_MEASUREMENTS", np.array([4, np.nan, "bad", -5], dtype=object))
    })
    apply_qc_step.context = {"data": data}

    with patch.dict("toolbox.steps.custom.apply_qc.QC_CLASSES", {"MockQC": MockQC}):
        result = apply_qc_step.run()

    # Expected: valid numbers stay, everything else becomes 0
    assert list(result["data"]["TEMP_QC"].values) == [1, 0, 0, 0]
    assert list(result["data"]["PHASE_QC"].values) == [4, 0, 0, 0]


def test_initialises_all_missing_qc_columns(apply_qc_step):
    """Checks both tested and untested variables get new QC columns with 0 for data and 9 for NaNs."""
    data = xr.Dataset({
        "TEMP": ("N_MEASUREMENTS", [10.0, np.nan, 12.0], MOCK_ATTRS),       # Tested
        "SALINITY": ("N_MEASUREMENTS", [35.1, np.nan, 35.3], MOCK_ATTRS)    # Untested
    })
    apply_qc_step.context = {"data": data}

    with patch.dict("toolbox.steps.custom.apply_qc.QC_CLASSES", {"MockQC": MockQC}):
        result = apply_qc_step.run()

    # Expected: 0 for valid data points, 9 for missing data points
    assert list(result["data"]["TEMP_QC"].values) == [0, 9, 0]
    assert list(result["data"]["SALINITY_QC"].values) == [0, 9, 0]


def test_combinatrix_flag_upgrades(apply_qc_step):
    """Checks that organise_flags respects the Argo combinatrix hierarchy."""
    apply_qc_step.flag_store = xr.Dataset({
        "TEMP_QC": ("N_MEASUREMENTS", np.array([2, 4, 3, 1], dtype=np.int8))
    })
    
    new_flags = xr.Dataset({
        "TEMP_QC": ("N_MEASUREMENTS", np.array([4, 1, 5, 8], dtype=np.int8))
    })
    
    apply_qc_step.organise_flags(new_flags)
    
    assert list(apply_qc_step.flag_store["TEMP_QC"].values) == [4, 4, 3, 8]