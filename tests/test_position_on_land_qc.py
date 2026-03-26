import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.qc.position_on_land_qc import position_on_land_qc
from utils.test_utils import create_mock_dataset


def test_missing_variables():
    data = xr.Dataset({"TEMP": ("N_MEASUREMENTS", [10.0, 12.0])})
    qc_step = position_on_land_qc(data)
    
    with pytest.raises(KeyError):
        qc_step.return_qc()


@pytest.mark.parametrize(
    "lats, lons, expected_flags", 
    [
        ([0.0, 0.0],          [-30.0, -140.0], [1, 1]),  # Ocean coordinates (Mid Atlantic, Central Pacific)
        ([39.0, -23.7],       [-98.0, 133.8],  [4, 4]),  # Land coordinates (Kansas USA, Alice Springs AUS)
        ([np.nan, 39.0],      [-30.0, np.nan], [1, 1]),  # Missing or invalid coordinates
    ],
    ids=["water", "land", "nan_coords"],
)
def test_locations(lats, lons, expected_flags):
    data = create_mock_dataset(lats=lats, lons=lons)
    qc_step = position_on_land_qc(data)
    flags = qc_step.return_qc()

    assert list(flags["LATITUDE_QC"].values) == expected_flags
    assert list(flags["LONGITUDE_QC"].values) == expected_flags


@patch("toolbox.steps.custom.qc.position_on_land_qc.plt.show")
def test_plot_diagnostics(mock_show):
    data = create_mock_dataset(lats=[0.0, -23.7], lons=[-30.0, 133.8])
    qc_step = position_on_land_qc(data)
    
    qc_step.return_qc()
    qc_step.plot_diagnostics()
    
    mock_show.assert_called_once_with(block=True)
