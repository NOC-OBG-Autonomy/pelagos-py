import pytest
import xarray as xr
import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime
from toolbox.steps.custom.qc.impossible_date_qc import impossible_date_qc

# Helper function to create a test xarray Dataset
def create_test_dataset(times):
    return xr.Dataset(
        {
            "TIME": ("N_MEASUREMENTS", times),
        },
        coords={"N_MEASUREMENTS": range(len(times))},
    )

def test_all_valid_dates():
    times_good = pd.date_range(start="2000-01-01", periods=10, freq="D")
    data = create_test_dataset(times_good)
    test = impossible_date_qc(data)
    flags = test.return_qc()
    assert (flags["TIME_QC"] == 1).all()

    bad_times = pd.date_range(start="1900-01-01", periods=10, freq="D")
    data = create_test_dataset(bad_times)
    test = impossible_date_qc(data)
    flags = test.return_qc()
    assert (flags["TIME_QC"] == 4).all()

    others = pd.to_datetime(
        ["1677-12-07", "1989-11-09", "1978-09-21", "2020-01-01", "1975-11-10"]
    )
    expected_flags = np.array([4, 1, 4, 1, 4])
    data = create_test_dataset(others)
    test = impossible_date_qc(data)
    flags = test.return_qc()
    assert (flags["TIME_QC"] == expected_flags).all()

# test_plot_diagnostics

