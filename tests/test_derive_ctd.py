from pelagos_py.steps.processing.derive_ctd import DeriveCTDVariables

import gsw
import numpy as np
import xarray as xr

ALL = ["DEPTH", "PSAL", "ASAL", "CONS_TEMP", "DENSITY"]


def make_context(**extra):
    n = 50
    ds = xr.Dataset(
        {
            "TIME": (
                "N_MEASUREMENTS",
                np.datetime64("2024-01-01") + np.arange(n).astype("timedelta64[s]"),
            ),
            "LATITUDE": ("N_MEASUREMENTS", np.full(n, 50.0)),
            "LONGITUDE": ("N_MEASUREMENTS", np.full(n, -10.0)),
            "PRES": ("N_MEASUREMENTS", np.linspace(0, 200, n)),
            "TEMP": ("N_MEASUREMENTS", np.linspace(15, 7, n)),
            "CNDC": ("N_MEASUREMENTS", np.linspace(4.2, 3.6, n)),
            **{name: ("N_MEASUREMENTS", values) for name, values in extra.items()},
        }
    )
    ds["CNDC"].attrs["units"] = "S/m"
    for var in list(ds.data_vars):
        ds[f"{var}_QC"] = ("N_MEASUREMENTS", np.ones(n, dtype=int))
    return {"data": ds, "global_parameters": {}}


def run_step(context, to_derive=ALL):
    step = DeriveCTDVariables(
        name="Derive CTD",
        parameters={"to_derive": to_derive},
        diagnostics=False,
        context=context,
    )
    return step.run()["data"]


def test_writes_adjusted_variables_with_qc():
    data = run_step(make_context())

    for var in ALL:
        assert f"{var}_ADJUSTED" in data
        assert f"{var}_ADJUSTED_QC" in data
        assert var not in data
    assert (
        data["PSAL_ADJUSTED"].attrs["standard_name"] == "sea_water_practical_salinity"
    )


def test_raw_psal_is_left_untouched():
    raw_psal = np.full(50, 30.0)
    data = run_step(make_context(PSAL=raw_psal))

    assert np.array_equal(data["PSAL"].values, raw_psal)
    assert not np.allclose(data["PSAL_ADJUSTED"].values, raw_psal)


def test_existing_psal_adjusted_is_not_overwritten_and_feeds_asal():
    psal_adjusted = np.full(50, 34.0)
    data = run_step(make_context(PSAL_ADJUSTED=psal_adjusted))

    assert np.array_equal(data["PSAL_ADJUSTED"].values, psal_adjusted)
    expected = gsw.SA_from_SP(psal_adjusted, data["PRES"].values, -10.0, 50.0)
    assert np.allclose(data["ASAL_ADJUSTED"].values, expected)


def test_output_variables_follow_to_derive():
    outputs = DeriveCTDVariables.output_variables({"to_derive": ["DEPTH", "PSAL"]})

    assert outputs == ["DEPTH_ADJUSTED", "PSAL_ADJUSTED"]
