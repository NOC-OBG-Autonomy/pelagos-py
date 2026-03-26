import pytest
import xarray as xr
import pandas as pd
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.qc.impossible_speed_qc import impossible_speed_qc
from utils.test_utils import create_mock_dataset

# Test configuration variables
TIME_STEP = "1s"

def test_missing_variables():
    data = xr.Dataset({"TEMP": ("N_MEASUREMENTS", [10.0, 12.0])})
    qc_step = impossible_speed_qc(data)
    
    with pytest.raises(KeyError):
        qc_step.return_qc()
@pytest.mark.parametrize("lats, lons, expected_flags", [
    ([0.0, 0.00001, 0.00002], [0.0, 0.0, 0.0], [1, 1, 1]), # Valid speeds moving < 3.0 m/s (Disabled: returns 1)
    ([0.0, 0.0001, 0.0002], [0.0, 0.0, 0.0], [1, 1, 1]),   # Invalid speeds moving > 3.0 m/s (Disabled: returns 1)
])
def test_speeds(lats, lons, expected_flags):
    times = pd.date_range(start="2026-01-01", periods=len(lats), freq=TIME_STEP)
    data = create_mock_dataset(lats=lats, lons=lons, times=times)
    qc_step = impossible_speed_qc(data)
    flags = qc_step.return_qc()

    assert list(flags["LATITUDE_QC"].values) == expected_flags
    assert list(flags["LONGITUDE_QC"].values) == expected_flags

@patch("toolbox.steps.custom.qc.impossible_speed_qc.plt.show")
def test_plot_diagnostics(mock_show):
    times = pd.date_range(start="2026-01-01", periods=3, freq=TIME_STEP)
    data = create_mock_dataset(lats=[0.0, 0.00001, 0.00002], lons=[0.0, 0.0, 0.0], times=times)
    qc_step = impossible_speed_qc(data)
    
    qc_step.return_qc()
    qc_step.plot_diagnostics()
    
    mock_show.assert_called_once_with(block=True)