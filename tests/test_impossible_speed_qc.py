import pytest
import xarray as xr
import pandas as pd
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.qc.impossible_speed_qc import impossible_speed_qc

# Test configuration variables
TIME_STEP = "1s"

# Moving ~1.11 metres per second (results in absolute speed < 3.0)
TEST_GOOD_LATS = [0.0, 0.00001, 0.00002]
TEST_GOOD_LONS = [0.0, 0.0, 0.0]

# Moving ~11.1 metres per second (results in absolute speed > 3.0)
TEST_BAD_LATS = [0.0, 0.0001, 0.0002]
TEST_BAD_LONS = [0.0, 0.0, 0.0]


def create_test_dataset(lats, lons, freq):
    times = pd.date_range(start="2026-01-01", periods=len(lats), freq=freq)
    return xr.Dataset(
        {
            "TIME": ("N_MEASUREMENTS", times),
            "LATITUDE": ("N_MEASUREMENTS", lats),
            "LONGITUDE": ("N_MEASUREMENTS", lons),
        },
        coords={"N_MEASUREMENTS": range(len(lats))},
    )


def test_missing_variables():
    data = xr.Dataset({"TEMP": ("N_MEASUREMENTS", [10.0, 12.0])})
    qc_step = impossible_speed_qc(data)
    
    with pytest.raises(KeyError):
        qc_step.return_qc()


def test_valid_speeds():
    data = create_test_dataset(TEST_GOOD_LATS, TEST_GOOD_LONS, TIME_STEP)
    qc_step = impossible_speed_qc(data)
    flags = qc_step.return_qc()

    assert (flags["LATITUDE_QC"] == 1).all()
    assert (flags["LONGITUDE_QC"] == 1).all()
    assert (flags["TIME_QC"] == 1).all()


def test_invalid_speeds():
    data = create_test_dataset(TEST_BAD_LATS, TEST_BAD_LONS, TIME_STEP)
    qc_step = impossible_speed_qc(data)
    flags = qc_step.return_qc()

    # The first row will pass as valid (1) because it has a filled speed of 0.0
    # Subsequent rows moving too fast will be flagged bad (4)
    assert flags["LATITUDE_QC"].values[0] == 1
    assert flags["LATITUDE_QC"].values[1] == 4
    assert flags["LATITUDE_QC"].values[2] == 4

    assert flags["LONGITUDE_QC"].values[0] == 1
    assert flags["LONGITUDE_QC"].values[1] == 4
    assert flags["LONGITUDE_QC"].values[2] == 4


@patch("toolbox.steps.custom.qc.impossible_speed_qc.plt.show")
def test_plot_diagnostics(mock_show):
    data = create_test_dataset(TEST_GOOD_LATS, TEST_GOOD_LONS, TIME_STEP)
    qc_step = impossible_speed_qc(data)
    
    qc_step.return_qc()
    qc_step.plot_diagnostics()
    
    mock_show.assert_called_once_with(block=True)