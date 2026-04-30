import pytest
import logging
import numpy as np
import xarray as xr
from unittest.mock import patch

from toolbox.steps.custom.qc.ctd_qc import ctd_qc


def create_ctd_dataset(pres=None, temp=None, cndc=None, cndc_units="S/m"):
    """Helper to create a minimal CTD dataset for testing."""
    n = len(pres)
    return xr.Dataset(
        {
            "PRES": ("N_MEASUREMENTS", np.array(pres, dtype=float)),
            "TEMP": ("N_MEASUREMENTS", np.array(temp, dtype=float)),
            "CNDC": ("N_MEASUREMENTS", np.array(cndc, dtype=float), {"units": cndc_units}),
            "TIME": ("N_MEASUREMENTS", np.arange(n, dtype=float)),
        },
        coords={"N_MEASUREMENTS": np.arange(n)},
    )


# --- Zero Flagging ---

def test_zero_values_flagged_as_9():
    """Exact 0.000 values across all three CTD variables should be flagged as 9."""
    data = create_ctd_dataset(
        pres=[0.0, 10.0],
        temp=[0.0, 8.5],
        cndc=[0.0, 3.5],
    )
    # ADD apply_cndc_range=False here so 3.5 isn't flagged as an outlier
    flags = ctd_qc(data, auto_scale=False, apply_cndc_range=False).return_qc()

    assert flags["PRES_QC"].values[0] == 9
    assert flags["TEMP_QC"].values[0] == 9
    assert flags["CNDC_QC"].values[0] == 9

    assert flags["PRES_QC"].values[1] == 0
    assert flags["TEMP_QC"].values[1] == 0


def test_nonzero_values_not_flagged():
    """Valid non-zero values should receive flag 0 (no issue raised)."""
    data = create_ctd_dataset(
        pres=[5.0, 10.0, 15.0],
        temp=[8.0, 8.5, 9.0],
        # CHANGE these to valid mS/cm values (> 20.0) so they aren't caught by the outlier check
        cndc=[34.0, 35.0, 36.0], 
        cndc_units="mS/cm",
    )
    flags = ctd_qc(data, auto_scale=False).return_qc()

    assert all(flags["PRES_QC"].values == 0)
    assert all(flags["TEMP_QC"].values == 0)
    assert all(flags["CNDC_QC"].values == 0)


# --- CNDC Unit Scaling ---
def test_cndc_scaled_when_in_sm(caplog):
    """CNDC in S/m (max < 10) should be scaled x10 to mS/cm when auto_scale=True."""
    # Tell caplog to capture INFO level messages
    caplog.set_level(logging.INFO)
    
    data = create_ctd_dataset(
        pres=[5.0, 10.0],
        temp=[8.0, 8.5],
        cndc=[3.5, 3.6],
        cndc_units="S/m",
    )
    step = ctd_qc(data, auto_scale=True)
    step.return_qc()

    assert step.scaled is True
    assert "Converting CNDC from S/m to mS/cm" in caplog.text
    assert data["CNDC"].attrs["units"] == "mS/cm"
    np.testing.assert_allclose(data["CNDC"].values, [35.0, 36.0])

def test_cndc_not_scaled_when_already_mscm():
    """CNDC already in mS/cm should not be scaled again."""
    data = create_ctd_dataset(
        pres=[5.0, 10.0],
        temp=[8.0, 8.5],
        cndc=[35.0, 36.0],
        cndc_units="mS/cm",
    )
    step = ctd_qc(data, auto_scale=True)
    step.return_qc()

    assert step.scaled is False
    np.testing.assert_allclose(data["CNDC"].values, [35.0, 36.0])


def test_cndc_not_scaled_when_auto_scale_disabled():
    """CNDC should never be scaled when auto_scale=False, regardless of magnitude."""
    data = create_ctd_dataset(
        pres=[5.0, 10.0],
        temp=[8.0, 8.5],
        cndc=[3.5, 3.6],
        cndc_units="S/m",
    )
    step = ctd_qc(data, auto_scale=False)
    step.return_qc()

    assert step.scaled is False
    np.testing.assert_allclose(data["CNDC"].values, [3.5, 3.6])


def test_cndc_not_scaled_when_units_metadata_says_mscm():
    """If metadata explicitly says mS/cm, skip scaling even if values look like S/m."""
    for unit_str in ["mS/cm", "ms/cm", "ms cm-1", "millisiemens/cm"]:
        data = create_ctd_dataset(
            pres=[5.0],
            temp=[8.0],
            cndc=[3.5],
            cndc_units=unit_str,
        )
        step = ctd_qc(data, auto_scale=True)
        step.return_qc()

        assert step.scaled is False, f"Should not scale when units='{unit_str}'"


# --- Sigma Outlier Detection ---

def test_cndc_outlier_cross_flags_triad():
    """A gross CNDC outlier should flag PRES, TEMP, and CNDC all as 4."""
    normal = [35.0] * 100
    data = create_ctd_dataset(
        pres=[10.0] * 101,
        temp=[8.5] * 101,
        cndc=normal + [9999.0],
        cndc_units="mS/cm",
    )
    flags = ctd_qc(data, auto_scale=False).return_qc()

    assert flags["CNDC_QC"].values[-1] == 4
    assert flags["PRES_QC"].values[-1] == 4
    assert flags["TEMP_QC"].values[-1] == 4

    assert all(flags["CNDC_QC"].values[:-1] == 0)
def test_zero_not_reclassified_as_outlier():
    """A zero value (flagged 9) should not be overwritten by the outlier (4) logic."""
    normal = [35.0] * 20
    data = create_ctd_dataset(
        pres=[10.0] * 20 + [10.0],
        temp=[8.5] * 20 + [8.5],
        cndc=normal + [0.0],
        cndc_units="mS/cm",
    )
    flags = ctd_qc(data, auto_scale=False).return_qc()

    assert flags["CNDC_QC"].values[-1] == 9


# --- Diagnostics ---

@patch("toolbox.steps.custom.qc.ctd_qc.plt.show")
@patch("toolbox.steps.custom.qc.ctd_qc.matplotlib.use")
def test_plot_diagnostics(mock_use, mock_show):
    data = create_ctd_dataset(
        pres=[5.0, 10.0],
        temp=[8.0, 8.5],
        cndc=[35.0, 36.0],
        cndc_units="mS/cm",
    )
    step = ctd_qc(data, auto_scale=False)
    step.return_qc()
    step.plot_diagnostics()

    mock_use.assert_called_once_with("tkagg")
    mock_show.assert_called_once_with(block=True)