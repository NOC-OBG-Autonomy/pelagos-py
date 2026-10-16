"""Tests the steps 'Derive Oxygen Concentration' and 'Derive DOXY' (src/pelagos_py/steps/processing/oxygen.py)."""

from pelagos_py.steps.processing import oxygen

import numpy as np
import xarray as xr

DeriveOxygenConcentration = oxygen.DeriveOxygenConcentration
DeriveDOXY = oxygen.DeriveDOXY

SVU_COEFFICIENTS = [
    2.67915e-03,
    1.12428e-04,
    2.29021e-06,
    1.43603e02,
    -2.08012e-01,
    -3.59303e01,
    2.82344e00,
]


def make_context(temp, phase):
    n = len(temp)
    ds = xr.Dataset(
        {
            "TEMP_DOXY": ("N_MEASUREMENTS", np.asarray(temp, dtype=float)),
            "TEMP_DOXY_QC": ("N_MEASUREMENTS", np.zeros(n, dtype=int)),
            "CAL_PHASE_DOXY": ("N_MEASUREMENTS", np.asarray(phase, dtype=float)),
            "CAL_PHASE_DOXY_QC": ("N_MEASUREMENTS", np.zeros(n, dtype=int)),
        }
    )
    return {"data": ds, "global_parameters": {}}


def make_step(parameters, context, diagnostics=False):
    return DeriveOxygenConcentration(
        name="Derive Oxygen Concentration",
        parameters=parameters,
        diagnostics=diagnostics,
        context=context,
    )


def test_svu_matches_stern_volmer_uchida_equation():
    c0, c1, c2, c3, c4, c5, c6 = SVU_COEFFICIENTS
    temp = [10.0, 15.0, 20.0]
    phase = [25.0, 27.0, 29.0]

    ctx = make_context(temp, phase)
    step = make_step(
        {
            "method": "SVU",
            "temperature_name": "TEMP_DOXY",
            "svu_coefficients": SVU_COEFFICIENTS,
        },
        ctx,
    )
    out = step.run()

    T = np.asarray(temp)
    P = np.asarray(phase)
    K_SV = c0 + c1 * T + c2 * T**2
    P0 = c3 + c4 * T
    Pc = c5 + c6 * P
    expected = (P0 / Pc - 1.0) / K_SV

    assert np.allclose(out["data"]["MOLAR_DOXY_ADJUSTED"].values, expected)
    assert "MOLAR_DOXY_ADJUSTED_QC" in out["data"]
    assert "MOLAR_DOXY_ADJUSTED" in out["intermediate_variables"]


def test_shipped_molar_doxy_is_left_untouched():
    ctx = make_context([10.0, 15.0], [25.0, 27.0])
    shipped = np.array([250.0, 260.0])
    ctx["data"]["MOLAR_DOXY"] = ("N_MEASUREMENTS", shipped.copy())
    step = make_step(
        {
            "method": "SVU",
            "temperature_name": "TEMP_DOXY",
            "svu_coefficients": SVU_COEFFICIENTS,
        },
        ctx,
    )
    out = step.run()

    assert np.array_equal(out["data"]["MOLAR_DOXY"].values, shipped)
    assert not np.allclose(out["data"]["MOLAR_DOXY_ADJUSTED"].values, shipped)


def make_doxy_context(molar_name, molar, psal=35.0, temp=10.0, pres=0.0):
    n = len(molar)
    zeros = np.zeros(n, dtype=int)
    ds = xr.Dataset(
        {
            molar_name: ("N_MEASUREMENTS", np.asarray(molar, dtype=float)),
            f"{molar_name}_QC": ("N_MEASUREMENTS", zeros.copy()),
            "PSAL": ("N_MEASUREMENTS", np.full(n, psal)),
            "PSAL_QC": ("N_MEASUREMENTS", zeros.copy()),
            "TEMP": ("N_MEASUREMENTS", np.full(n, temp)),
            "TEMP_QC": ("N_MEASUREMENTS", zeros.copy()),
            "PRES": ("N_MEASUREMENTS", np.full(n, pres)),
            "PRES_QC": ("N_MEASUREMENTS", zeros.copy()),
            "LATITUDE": ("N_MEASUREMENTS", np.zeros(n)),
            "LONGITUDE": ("N_MEASUREMENTS", np.zeros(n)),
        }
    )
    return {"data": ds, "global_parameters": {}}


def run_derive_doxy(parameters, context):
    step = DeriveDOXY(
        name="Derive DOXY", parameters=parameters, diagnostics=False, context=context
    )
    return step.run()["data"]


def test_derive_doxy_divides_by_potential_density_at_zero_pressure():
    # 35 PSU, 10 degC, 0 dbar: rho0 = 1026.95 kg/m3
    ctx = make_doxy_context("MOLAR_DOXY_PSAL_PRES", [300.0])
    data = run_derive_doxy({}, ctx)

    assert np.isclose(
        data["DOXY_ADJUSTED"].values[0], 300.0 * 1000 / 1026.95, rtol=1e-5
    )
    assert data["DOXY_ADJUSTED"].attrs["units"] == "umol/kg"
    assert data["DOXY_ADJUSTED_QC"].values[0] == 0


def test_derive_doxy_uses_potential_not_in_situ_density():
    ctx = make_doxy_context("MOLAR_DOXY_PSAL_PRES", [300.0], pres=1000.0)
    data = run_derive_doxy({}, ctx)

    assert np.isclose(
        data["DOXY_ADJUSTED"].values[0], 300.0 * 1000 / 1026.98, rtol=1e-5
    )


def test_derive_doxy_from_shipped_molar_doxy_reads_adjusted_inputs():
    ctx = make_doxy_context("MOLAR_DOXY", [300.0, 300.0], psal=0.0)
    ctx["data"]["PSAL_ADJUSTED"] = ("N_MEASUREMENTS", np.full(2, 35.0))
    ctx["data"]["PSAL_ADJUSTED_QC"] = ("N_MEASUREMENTS", np.array([0, 4]))
    data = run_derive_doxy({"apply_to": "MOLAR_DOXY"}, ctx)

    assert np.allclose(data["DOXY_ADJUSTED"].values, 300.0 * 1000 / 1026.95, rtol=1e-5)
    assert list(data["DOXY_ADJUSTED_QC"].values) == [0, 4]
    assert np.array_equal(data["MOLAR_DOXY"].values, [300.0, 300.0])
