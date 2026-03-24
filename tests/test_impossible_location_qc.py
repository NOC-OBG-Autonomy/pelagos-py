import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.qc.impossible_location_qc import impossible_location_qc

TEST_GOOD_LATS = [45.0, -89.9]
TEST_GOOD_LONS = [179.9, -179.9]

TEST_BAD_LATS = [95.0, -100.0] 
TEST_BAD_LONS = [185.0, -185.0]

TEST_NAN_LATS = [np.nan, 45.0]
TEST_NAN_LONS = [10.0, np.nan]

TEST_ZERO_LATS = [0.0, 0.0]
TEST_ZERO_LONS = [0.0, 0.0]

def create_test_dataset(lats, lons):
    return xr.Dataset(
        {
            "LATITUDE": ("N_MEASUREMENTS", lats),
            "LONGITUDE": ("N_MEASUREMENTS", lons),
        },
        coords={"N_MEASUREMENTS": range(len(lats))},
    )

def test_missing_variables():
    data = xr.Dataset({"TEMP": ("N_MEASUREMENTS", [10.0, 12.0])})
    qc_step = impossible_location_qc(data)
    
    with pytest.raises(KeyError):
        qc_step.return_qc()

def test_valid_locations():
    data = create_test_dataset(TEST_GOOD_LATS, TEST_GOOD_LONS)
    qc_step = impossible_location_qc(data)
    flags = qc_step.return_qc()

    assert (flags["LATITUDE_QC"] == 1).all()
    assert (flags["LONGITUDE_QC"] == 1).all()

def test_zero_locations():
    data = create_test_dataset(TEST_ZERO_LATS, TEST_ZERO_LONS)
    qc_step = impossible_location_qc(data)
    flags = qc_step.return_qc()

    assert (flags["LATITUDE_QC"] == 1).all()
    assert (flags["LONGITUDE_QC"] == 1).all()

def test_invalid_locations():
    data = create_test_dataset(TEST_BAD_LATS, TEST_BAD_LONS)
    qc_step = impossible_location_qc(data)
    flags = qc_step.return_qc()

    assert flags["LATITUDE_QC"].values[0] == 4
    assert flags["LATITUDE_QC"].values[1] == 4

    assert flags["LONGITUDE_QC"].values[0] == 4
    assert flags["LONGITUDE_QC"].values[1] == 4

def test_nan_locations():
    data = create_test_dataset(TEST_NAN_LATS, TEST_NAN_LONS)
    qc_step = impossible_location_qc(data)
    flags = qc_step.return_qc()

    assert flags["LATITUDE_QC"].values[0] == 9
    assert flags["LATITUDE_QC"].values[1] == 1

    assert flags["LONGITUDE_QC"].values[0] == 1
    assert flags["LONGITUDE_QC"].values[1] == 9

@patch("toolbox.steps.custom.qc.impossible_location_qc.plt.show")
def test_plot_diagnostics(mock_show):
    data = create_test_dataset(TEST_GOOD_LATS, TEST_GOOD_LONS)
    qc_step = impossible_location_qc(data)
    
    qc_step.return_qc()
    qc_step.plot_diagnostics()
    
    mock_show.assert_called_once_with(block=True)