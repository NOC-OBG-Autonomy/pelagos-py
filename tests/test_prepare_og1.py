"""Tests the step 'Prepare OG1' (src/pelagos_py/steps/input_output/prepare_og1.py)."""

from pelagos_py.steps.input_output.prepare_og1 import PrepareOG1

import numpy as np
import xarray as xr


def make_context(**variables):
    ds = xr.Dataset()
    for name, (values, units) in variables.items():
        ds[name] = ("N_MEASUREMENTS", np.asarray(values, dtype=float))
        if units:
            ds[name].attrs["units"] = units
    return {"data": ds, "global_parameters": {}}


def run(context, **parameters):
    return PrepareOG1(name="prepare", parameters=parameters, context=context).run()["data"]


def test_renames_gps_coordinates_when_canonical_missing():
    ctx = make_context(LATITUDE_GPS=([50.0, 51.0], "degree_north"), LONGITUDE_GPS=([-5.0, -6.0], "degree_east"))
    ctx["data"]["LATITUDE_GPS_QC"] = ("N_MEASUREMENTS", np.array([1, 1], dtype=np.int8))

    out = run(ctx)

    assert "LATITUDE" in out and "LATITUDE_GPS" not in out
    assert "LATITUDE_QC" in out
    assert "Renamed from LATITUDE_GPS" in out["LATITUDE"].attrs["comment"]
    assert "LONGITUDE" in out


def test_bodc_codes_used_when_gps_names_absent():
    ctx = make_context(ALATPT01=([50.0], None), ALONPT01=([-5.0], None))
    out = run(ctx)
    assert "LATITUDE" in out and "LONGITUDE" in out


def test_all_nan_alternative_does_not_count():
    # An empty LATITUDE_GPS beside a real LATITUDE must leave LATITUDE alone.
    ctx = make_context(LATITUDE=([50.0], None), LATITUDE_GPS=([np.nan], None))
    out = run(ctx)
    assert np.allclose(out["LATITUDE"].values, 50.0)
    assert "LATITUDE_GPS" in out


def test_cndc_mislabelled_as_s_per_m_is_rescaled():
    ctx = make_context(CNDC=([35.0, 36.0], "S m-1"))
    out = run(ctx)
    assert np.allclose(out["CNDC"].values, [3.5, 3.6])
    assert out["CNDC"].attrs["units"] == "S/m"


def test_cndc_mislabelled_as_ms_per_cm_is_relabelled():
    ctx = make_context(CNDC=([3.5, 3.6], "mS/cm"))
    out = run(ctx)
    assert np.allclose(out["CNDC"].values, [3.5, 3.6])
    assert out["CNDC"].attrs["units"] == "S/m"


def test_cndc_genuine_units_untouched():
    for values, units in (([3.5], "mhos/m"), ([35.0], "mS/cm")):
        out = run(make_context(CNDC=(values, units)))
        assert np.allclose(out["CNDC"].values, values)
        assert out["CNDC"].attrs["units"] == units


def test_bbp700_treated_as_beta_by_default():
    ctx = make_context(BBP700=([1e-4], "m-1"))
    out = run(ctx)
    assert "BETA_BACKSCATTERING700" in out and "BBP700" not in out


def test_bbp700_kept_when_beta_present_or_flag_off():
    ctx = make_context(BBP700=([1e-4], "m-1"), BETA_BACKSCATTERING700=([1e-4], "m-1.sr-1"))
    assert "BBP700" in run(ctx)
    ctx = make_context(BBP700=([1e-4], "m-1"))
    assert "BBP700" in run(ctx, bbp700_is_beta=False)


def test_renames_for_matches_run():
    names = {"LATITUDE_GPS", "LONGITUDE", "BBP700", "DOXY"}
    assert PrepareOG1.renames_for(names) == {
        "LATITUDE_GPS": "LATITUDE", "BBP700": "BETA_BACKSCATTERING700", "DOXY": "MOLAR_DOXY",
    }
    assert "BBP700" not in PrepareOG1.renames_for(names, {"bbp700_is_beta": False})
