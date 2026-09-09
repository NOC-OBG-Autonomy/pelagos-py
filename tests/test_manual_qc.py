import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pelagos_py.steps.quality_control.manual_qc import manual_qc


def make_data():
    t = pd.date_range("2024-05-01", periods=6, freq="h").values
    return xr.Dataset(
        {
            "PRES": ("N_MEASUREMENTS", np.array([1.0, 5.0, 10.0, 20.0, np.nan, 30.0])),
            "TEMP": ("N_MEASUREMENTS", np.arange(6, dtype=float)),
        },
        coords={"N_MEASUREMENTS": np.arange(6), "TIME": ("N_MEASUREMENTS", t)},
    )


def test_inside_box_flags_y_variable_by_default():
    qc = manual_qc(make_data(), boxes=[
        {"x": ["2024-05-01T00:30", "2024-05-01T03:30"], "y": [0, 12], "flag": 4},
    ])
    assert qc.required_variables == ["TIME", "PRES"]
    assert qc.qc_outputs == ["PRES_QC"]
    assert list(qc.return_qc()["PRES_QC"].values) == [0, 4, 4, 0, 0, 0]


def test_outside_box_and_extra_variables():
    qc = manual_qc(make_data(), boxes=[
        {"x": ["2024-05-01", "2024-05-02"], "y": [0, 25], "flag": 3,
         "mode": "outside", "variables": ["PRES", "TEMP"]},
    ])
    flags = qc.return_qc()
    # NaN PRES is never flagged; the 30 dbar sample is outside the box.
    assert list(flags["PRES_QC"].values) == [0, 0, 0, 0, 0, 3]
    assert list(flags["TEMP_QC"].values) == [0, 0, 0, 0, 0, 3]


def test_worst_flag_wins_on_overlap():
    qc = manual_qc(make_data(), boxes=[
        {"x": ["2024-05-01", "2024-05-02"], "y": [0, 100], "flag": 2},
        {"x": ["2024-05-01T01:30", "2024-05-01T02:30"], "y": [0, 100], "flag": 4},
    ])
    assert list(qc.return_qc()["PRES_QC"].values) == [2, 2, 4, 2, 0, 2]


def test_numeric_x_axis():
    qc = manual_qc(make_data(), x_variable="TEMP", y_variable="PRES", boxes=[
        {"x": [0.5, 2.5], "y": [0, 100], "flag": 4},
    ])
    assert list(qc.return_qc()["PRES_QC"].values) == [0, 4, 4, 0, 0, 0]


def test_no_boxes_has_no_outputs():
    qc = manual_qc(None, boxes=[])
    assert qc.qc_outputs == []


@pytest.mark.parametrize("box", [
    {"x": [0, 1], "y": [0], "flag": 4},
    {"x": [0, 1], "y": [0, 1], "flag": 10},
    {"x": [0, 1], "y": [0, 1], "flag": 4, "mode": "sideways"},
])
def test_invalid_box_rejected(box):
    with pytest.raises(ValueError):
        manual_qc(None, boxes=[box])
