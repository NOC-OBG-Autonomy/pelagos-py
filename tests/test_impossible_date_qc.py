import pytest
import xarray as xr
import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime

from toolbox.steps.custom.qc.impossible_date_qc import impossible_date_qc
from utils.test_utils import create_mock_dataset


@pytest.mark.parametrize("times, expected_flags", [
    (pd.date_range(start="2000-01-01", periods=10, freq="D"), [1] * 10),
    (pd.date_range(start="1900-01-01", periods=10, freq="D"), [4] * 10),
    (pd.to_datetime(["1677-12-07", "1989-11-09", "1978-09-21", "2020-01-01", "1975-11-10"]), [4, 1, 4, 1, 4]),
])
def test_dates(times, expected_flags):
    data = create_mock_dataset(times=times)
    test = impossible_date_qc(data)
    flags = test.return_qc()
    
    assert list(flags["TIME_QC"].values) == expected_flags
