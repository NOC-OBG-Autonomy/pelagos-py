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
    # Untouched samples become 1 by default; the NaN sample is 9.
    assert list(qc.return_qc()["PRES_QC"].values) == [1, 4, 4, 1, 9, 1]


def test_remaining_left_zero_when_off():
    qc = manual_qc(make_data(), flag_remaining_good=False, boxes=[
        {"x": ["2024-05-01T00:30", "2024-05-01T03:30"], "y": [0, 12], "flag": 4},
    ])
    assert list(qc.return_qc()["PRES_QC"].values) == [0, 4, 4, 0, 9, 0]


def test_point_flags_nearest_sample_only():
    qc = manual_qc(make_data(), flag_remaining_good=False, boxes=[
        {"x": ["2024-05-01T02:10"], "y": [9.0], "flag": 3},
    ])
    assert list(qc.return_qc()["PRES_QC"].values) == [0, 0, 3, 0, 9, 0]


def test_outside_box_and_extra_variables():
    qc = manual_qc(make_data(), boxes=[
        {"x": ["2024-05-01", "2024-05-02"], "y": [0, 25], "flag": 3,
         "mode": "outside", "variables": ["PRES", "TEMP"]},
    ])
    flags = qc.return_qc()
    # NaN PRES is never flagged; the 30 dbar sample is outside the box.
    assert list(flags["PRES_QC"].values) == [1, 1, 1, 1, 9, 3]
    assert list(flags["TEMP_QC"].values) == [1, 1, 1, 1, 1, 3]


def test_later_box_wins_on_overlap():
    qc = manual_qc(make_data(), boxes=[
        {"x": ["2024-05-01T01:30", "2024-05-01T02:30"], "y": [0, 100], "flag": 4},
        {"x": ["2024-05-01", "2024-05-02"], "y": [0, 100], "flag": 2},
    ])
    assert list(qc.return_qc()["PRES_QC"].values) == [2, 2, 2, 2, 9, 2]


def test_override_lowers_existing_flag_but_combinatrix_box_does_not():
    data = make_data()
    data["PRES_QC"] = ("N_MEASUREMENTS", np.array([4, 4, 4, 4, 9, 0]))
    box = {"x": ["2024-05-01T00:30", "2024-05-01T01:30"], "y": [0, 100], "flag": 1}
    assert list(manual_qc(data, boxes=[box]).return_qc()["PRES_QC"].values) == [4, 1, 4, 4, 9, 1]
    assert list(manual_qc(data, boxes=[{**box, "override": False}]).return_qc()["PRES_QC"].values) == [4, 4, 4, 4, 9, 1]


def test_apply_qc_store_is_the_starting_point():
    qc = manual_qc(make_data(), flag_remaining_good=False, boxes=[])
    qc.existing_flags = xr.Dataset({"PRES_QC": ("N_MEASUREMENTS", np.array([3, 3, 0, 0, 9, 0]))})
    assert list(qc.return_qc()["PRES_QC"].values) == [3, 3, 0, 0, 9, 0]


def test_numeric_x_axis():
    qc = manual_qc(make_data(), x_variable="TEMP", y_variable="PRES", boxes=[
        {"x": [0.5, 2.5], "y": [0, 100], "flag": 4},
    ])
    assert list(qc.return_qc()["PRES_QC"].values) == [1, 4, 4, 1, 9, 1]


def test_no_boxes_still_targets_y_variable():
    qc = manual_qc(None, boxes=[])
    assert qc.qc_outputs == ["PRES_QC"]


@pytest.mark.parametrize("box", [
    {"x": [0, 1], "y": [0], "flag": 4},
    {"x": [0, 1, 2], "y": [0, 1], "flag": 4},
    {"x": [0, 1], "y": [0, 1], "flag": 10},
    {"x": [0, 1], "y": [0, 1], "flag": 4, "mode": "sideways"},
])
def test_invalid_box_rejected(box):
    with pytest.raises(ValueError):
        manual_qc(None, boxes=[box])


def test_box_on_its_own_axes():
    qc = manual_qc(make_data(), flag_remaining_good=False, boxes=[
        {"x": ["2024-05-01", "2024-05-02"], "y": [2.5, 3.5], "flag": 4, "y_variable": "TEMP"},
    ])
    assert qc.required_variables == ["TIME", "PRES", "TEMP"]
    flags = qc.return_qc()
    assert list(flags["TEMP_QC"].values) == [0, 0, 0, 4, 0, 0]
    assert list(flags["PRES_QC"].values) == [0, 0, 0, 0, 9, 0]
