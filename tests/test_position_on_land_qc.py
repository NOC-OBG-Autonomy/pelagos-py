import logging
import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.qc.position_on_land_qc import position_on_land_qc
from utils.test_utils import create_mock_dataset

def test_missing_variables(caplog):
    pipeline_logger = logging.getLogger("toolbox.pipeline")
    orig_propagate = pipeline_logger.propagate
    pipeline_logger.propagate = True
    try:
        with caplog.at_level("WARNING"):
            data = xr.Dataset({"TEMP": ("N_MEASUREMENTS", [10.0, 12.0])})
            qc_step = position_on_land_qc(data)
            flags = qc_step.return_qc()
    finally:
        pipeline_logger.propagate = orig_propagate

    assert "LATITUDE or LONGITUDE missing" in caplog.text
    assert "LATITUDE_QC" not in flags
    assert "LONGITUDE_QC" not in flags

@pytest.mark.parametrize(
    "lats, lons, expected_flags", 
    [
        ([0.0, 0.0],          [-30.0, -140.0], [1, 1]),
        ([39.0, -23.7],       [-98.0, 133.8],  [4, 4]),
        ([np.nan, 39.0],      [-30.0, np.nan], [1, 1]),
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
@patch("toolbox.steps.custom.qc.position_on_land_qc.matplotlib.use")
def test_plot_diagnostics(mock_use, mock_show):
    data = create_mock_dataset(lats=[0.0, -23.7], lons=[-30.0, 133.8])
    qc_step = position_on_land_qc(data)
    
    qc_step.return_qc()
    qc_step.plot_diagnostics()
    
    mock_use.assert_called_once_with("tkagg")
    mock_show.assert_called_once_with(block=True)