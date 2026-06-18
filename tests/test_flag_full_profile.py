import numpy as np
import pytest
import xarray as xr

from pelagos_py.steps.quality_control.flag_full_profile import flag_full_profile


def make_dataset(profile_numbers, pres_qc):
    n = len(profile_numbers)
    return xr.Dataset(
        {
            "PRES": ("N_MEASUREMENTS", np.zeros(n)),
            "PRES_QC": ("N_MEASUREMENTS", np.array(pres_qc)),
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.array(profile_numbers, dtype=float)),
        },
        coords={"N_MEASUREMENTS": range(n)},
    )


def test_profile_over_threshold_is_fully_flagged():
    """A profile hitting the bad-flag threshold has all its points flagged bad (4);
    a profile below the threshold is left untouched."""
    profile_numbers = [1, 1, 1, 2, 2, 2]
    pres_qc = [4, 4, 1, 4, 1, 1]


    qc = flag_full_profile(make_dataset(profile_numbers, pres_qc), check_vars={"PRES": 2})
    flags = qc.return_qc()

    assert list(flags["PRES_QC"].values) == [4, 4, 4, 4, 1, 1]


def test_missing_check_vars_raises():
    """check_vars is required; omitting it raises an error."""
    with pytest.raises(ValueError, match="check_vars"):
        flag_full_profile(make_dataset([1, 1], [1, 1]))
