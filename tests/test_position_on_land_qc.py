import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.qc.position_on_land_qc import position_on_land_qc

# Test configuration variables
# Ocean coordinates (Mid Atlantic, Central Pacific)
TEST_WATER_LATS = [0.0, 0.0]
TEST_WATER_LONS = [-30.0, -140.0]

# Land coordinates (Kansas USA, Alice Springs AUS)
TEST_LAND_LATS = [39.0, -23.7] 
TEST_LAND_LONS = [-98.0, 133.8]

# Missing or invalid coordinates
TEST_NAN_LATS = [np.nan, 39.0]
TEST_NAN_LONS = [-30.0, np.nan]


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
    qc_step = position_on_land_qc(data)
    
    with pytest.raises(KeyError):
        qc_step.return_qc()


def test_water_locations():
    data = create_test_dataset(TEST_WATER_LATS, TEST_WATER_LONS)
    qc_step = position_on_land_qc(data)
    flags = qc_step.return_qc()

    assert (flags["LATITUDE_QC"] == 1).all()
    assert (flags["LONGITUDE_QC"] == 1).all()


def test_land_locations():
    data = create_test_dataset(TEST_LAND_LATS, TEST_LAND_LONS)
    qc_step = position_on_land_qc(data)
    flags = qc_step.return_qc()

    assert (flags["LATITUDE_QC"] == 4).all()
    assert (flags["LONGITUDE_QC"] == 4).all()


def test_nan_locations():
    data = create_test_dataset(TEST_NAN_LATS, TEST_NAN_LONS)
    qc_step = position_on_land_qc(data)
    flags = qc_step.return_qc()

    assert (flags["LATITUDE_QC"] == 1).all()
    assert (flags["LONGITUDE_QC"] == 1).all()


@patch("toolbox.steps.custom.qc.position_on_land_qc.plt.show")
def test_plot_diagnostics(mock_show):
    data = create_test_dataset(TEST_WATER_LATS, TEST_WATER_LONS)
    qc_step = position_on_land_qc(data)
    
    qc_step.return_qc()
    qc_step.plot_diagnostics()
    
    mock_show.assert_called_once_with(block=True)