import numpy as np
import xarray as xr
from unittest.mock import patch

from pelagos_py.steps.quality_control.impossible_speed_qc import impossible_speed_qc
from utils.test_utils import create_mock_dataset


def _hourly(n):
    return np.datetime64("2024-01-01") + np.arange(n) * np.timedelta64(1, "h")


def test_speed_in_metres_per_second():
    # 0.1 deg latitude per hour is ~3.1 m/s: bad; 0.05 deg per hour is ~1.5 m/s: good
    data = create_mock_dataset(lats=[60.0, 60.1, 60.15], lons=[0.0, 0.0, 0.0], times=_hourly(3))
    flags = impossible_speed_qc(data).return_qc()
    assert list(flags["LATITUDE_QC"].values) == [1, 4, 1]
    assert list(flags["TIME_QC"].values) == [1, 4, 1]


def test_longitude_scaled_by_latitude():
    # 0.1 deg longitude at 60N is half the distance it is at the equator
    data = create_mock_dataset(lats=[60.0, 60.0], lons=[0.0, 0.1], times=_hourly(2))
    assert list(impossible_speed_qc(data).return_qc()["LONGITUDE_QC"].values) == [1, 1]
    data = create_mock_dataset(lats=[0.0, 0.0], lons=[0.0, 0.1], times=_hourly(2))
    assert list(impossible_speed_qc(data).return_qc()["LONGITUDE_QC"].values) == [1, 4]


def test_gps_jitter_within_min_interval_is_ignored():
    # Two fixes 1 s apart, 20 m apart (20 m/s) -- jitter, not motion.
    times = np.datetime64("2024-01-01") + np.array([0, 1, 3600]) * np.timedelta64(1, "s")
    data = create_mock_dataset(lats=[60.0, 60.00018, 60.01], lons=[0.0, 0.0, 0.0], times=times)
    assert list(impossible_speed_qc(data).return_qc()["LATITUDE_QC"].values) == [1, 1, 1]


def test_single_bad_fix_does_not_flag_the_next():
    # A 1-degree jump out and back: only the outlier is bad, the return is
    # judged against the last good fix.
    data = create_mock_dataset(lats=[60.0, 61.0, 60.02, 60.04], lons=[0.0] * 4, times=_hourly(4))
    flags = impossible_speed_qc(data).return_qc()
    assert list(flags["LATITUDE_QC"].values) == [1, 4, 1, 1]


def test_speed_skips_samples_without_a_fix():
    data = create_mock_dataset(
        lats=[60.0, np.nan, np.nan, 60.05], lons=[0.0, np.nan, np.nan, 0.0], times=_hourly(4)
    )
    flags = impossible_speed_qc(data).return_qc()
    assert list(flags["LATITUDE_QC"].values) == [1, 9, 9, 1]
    assert list(flags["TIME_QC"].values) == [1, 1, 1, 1]


@patch("pelagos_py.steps.quality_control.impossible_speed_qc.plt.show")
@patch("pelagos_py.steps.quality_control.impossible_speed_qc.matplotlib.use")
def test_plot_diagnostics(mock_use, mock_show):
    data = create_mock_dataset(lats=[60.0, 60.1], lons=[0.0, 0.0], times=_hourly(2))
    qc = impossible_speed_qc(data)
    qc.return_qc()
    qc.plot_diagnostics()
    mock_show.assert_called_once_with(block=True)
