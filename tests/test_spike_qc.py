import numpy as np
import xarray as xr

from pelagos_py.steps.quality_control.spike_qc import spike_qc


def make_data(values, profiles):
    return xr.Dataset(
        {
            "CHLA": ("N_MEASUREMENTS", np.asarray(values, dtype=float)),
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.asarray(profiles, dtype=float)),
        },
        coords={"N_MEASUREMENTS": np.arange(len(values))},
    )


def test_profile_shorter_than_window_is_left_unchecked_and_counted(caplog):
    data = make_data([1.0, 1.1, 1.0], [0, 0, 0])
    qc = spike_qc(data, variables={"CHLA": 3}, window_size=10)

    flags = qc.return_qc()["CHLA_QC"].values
    assert list(flags) == [0, 0, 0]
    assert qc._untested["CHLA"] == (1, 1)


def test_spike_in_long_profile_is_flagged():
    values = [1.0] * 15
    values[7] = 50.0
    qc = spike_qc(make_data(values, [0] * 15), variables={"CHLA": 2}, window_size=5)

    flags = qc.return_qc()["CHLA_QC"].values
    assert flags[7] == 4 and set(np.delete(flags, 7)) == {1}


def test_also_flag_does_not_propagate_missing():
    # CNDC present where PRES is NaN: CNDC must not inherit flag 9
    n = 30
    pres = np.linspace(0, 100, n)
    pres[5] = np.nan
    data = xr.Dataset(
        {
            "PRES": ("N_MEASUREMENTS", pres),
            "CNDC": ("N_MEASUREMENTS", np.full(n, 4.0)),
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.zeros(n)),
        },
        coords={"N_MEASUREMENTS": np.arange(n)},
    )
    qc = spike_qc(data, variables={"PRES": 3}, also_flag={"PRES": ["CNDC"]}, window_size=5)
    flags = qc.return_qc()
    assert flags["PRES_QC"].values[5] == 9
    assert flags["CNDC_QC"].values[5] == 0
    assert (flags["CNDC_QC"].values[[0, 10]] == flags["PRES_QC"].values[[0, 10]]).all()
