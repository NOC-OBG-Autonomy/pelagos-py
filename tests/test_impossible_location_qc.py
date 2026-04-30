import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.qc.impossible_location_qc import impossible_location_qc
from utils.test_utils import create_mock_dataset


def test_missing_variables(caplog):
    data = xr.Dataset({"TEMP": ("N_MEASUREMENTS", [10.0, 12.0])})
    qc_step = impossible_location_qc(data)
    
    flags = qc_step.return_qc()
    
    assert "LATITUDE or LONGITUDE missing" in caplog.text
    assert "LATITUDE_QC" not in flags
    assert "LONGITUDE_QC" not in flags


@pytest.mark.parametrize("lats,lons,lat_flags,lon_flags", [
    ([45.0, -89.9], [179.9, -179.9], [1, 1],  [1, 1]),  # valid
    ([0.0, 0.0],    [0.0, 0.0],      [1, 1],   [1, 1]),  # zero
    ([95.0, -100.0],[185.0, -185.0], [4, 4],   [4, 4]),  # invalid
    ([np.nan, 45.0],[10.0, np.nan],  [9, 1],   [1, 9]),  # nan
])
def test_locations(lats, lons, lat_flags, lon_flags):
    data = create_mock_dataset(lats=lats, lons=lons)
    qc_step = impossible_location_qc(data)
    flags = qc_step.return_qc()

    assert list(flags["LATITUDE_QC"].values) == lat_flags
    assert list(flags["LONGITUDE_QC"].values) == lon_flags


@patch("toolbox.steps.custom.qc.impossible_location_qc.plt.show")
@patch("toolbox.steps.custom.qc.impossible_location_qc.matplotlib.use")
def test_plot_diagnostics(mock_use, mock_show):
    data = create_mock_dataset(lats=[45.0, -89.9], lons=[179.9, -179.9])
    qc_step = impossible_location_qc(data)
    
    qc_step.return_qc()
    qc_step.plot_diagnostics()
    
    mock_use.assert_called_once_with("tkagg")
    mock_show.assert_called_once_with(block=True)