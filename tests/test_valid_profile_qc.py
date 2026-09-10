import numpy as np
import xarray as xr

from pelagos_py.steps.quality_control.valid_profile_qc import valid_profile_qc


def make_dataset(profile_numbers, pres):
    return xr.Dataset(
        {
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.array(profile_numbers, dtype=float)),
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
