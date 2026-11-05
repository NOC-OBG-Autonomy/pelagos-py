from unittest.mock import patch

import numpy as np
import xarray as xr

from pelagos_py.steps.quality_control.valid_profile_qc import valid_profile_qc


def make_dataset(profile_numbers, pres):
    return xr.Dataset(
        {
            "PROFILE_NUMBER": (
                "N_MEASUREMENTS",
                np.array(profile_numbers, dtype=float),
            ),
            "PRES": ("N_MEASUREMENTS", np.array(pres, dtype=float)),
        },
        coords={"N_MEASUREMENTS": range(len(profile_numbers))},
    )


def test_flag_assignment():
    # profile 1 good, 2 too short, 3 never in depth range, last row has no profile
    profile_numbers = [1, 1, 1, 2, 3, 3, 3, np.nan]
    pres = [10, 20, 30, 10, 500, 600, 700, 5]
    flags = valid_profile_qc(
        make_dataset(profile_numbers, pres), min_length=3, depth_range=[0, 100]
    ).return_qc()
    assert list(flags["PROFILE_NUMBER_QC"].values) == [1, 1, 1, 4, 4, 4, 4, 0]


def test_custom_flag_and_no_depth_range():
    flags = valid_profile_qc(
        make_dataset([1, 1, 2, 2, 2], [5000, 5000, 20, 25, 30]), min_length=3, flag=3
    ).return_qc()
    assert list(flags["PROFILE_NUMBER_QC"].values) == [3, 3, 1, 1, 1]


def test_depth_range_only_requires_depth_var():
    qc = valid_profile_qc(None, min_length=2)
    assert qc.required_variables == ["PROFILE_NUMBER"]
    qc = valid_profile_qc(None, depth_range=[0, 10], depth_var="DEPTH")
    assert qc.required_variables == ["PROFILE_NUMBER", "DEPTH"]


def test_depth_range_prefers_adjusted_pressure():
    data = make_dataset([1, 1, 1], [500, 600, 700])
    data["PRES_ADJUSTED"] = ("N_MEASUREMENTS", np.array([10.0, 20.0, 30.0]))
    qc = valid_profile_qc(data, min_length=3, depth_range=[0, 100])
    assert qc.required_variables == ["PROFILE_NUMBER", "PRES_ADJUSTED"]
    assert list(qc.return_qc()["PROFILE_NUMBER_QC"].values) == [1, 1, 1]


@patch("pelagos_py.steps.quality_control.valid_profile_qc.plt.show")
@patch("pelagos_py.steps.quality_control.valid_profile_qc.matplotlib.use")
def test_plot_diagnostics_without_depth_variable(mock_use, mock_show):
    data = make_dataset([1, 1, 2, 2], [10, 20, 10, 20]).drop_vars("PRES")
    qc = valid_profile_qc(data, min_length=2)
    qc.return_qc()
    qc.plot_diagnostics()
    mock_show.assert_called_once_with(block=True)
