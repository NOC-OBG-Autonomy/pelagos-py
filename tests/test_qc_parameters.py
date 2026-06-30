"""Parameter-schema behaviour for QC checks and the schema-aware pipeline validator."""

import logging

import pytest

from pelagos_py.steps.quality_control.impossible_range_qc import impossible_range_qc
from pelagos_py.steps.quality_control.spike_qc import spike_qc
from pelagos_py.steps.quality_control.gross_range_qc import gross_range_qc
from pelagos_py.steps.quality_control.impossible_date_qc import impossible_date_qc
from pelagos_py.utils.valid_config_check import check_pipeline_variables


LOGGER = logging.getLogger("test")


# --- Dynamic QC resolve their I/O from parameters --------------------------------


def test_dynamic_qc_derives_required_variables():
    qc = impossible_range_qc(
        None,
        variable_ranges={"PRES": {4: [-5, 0]}},
        also_flag={"PRES": ["CNDC"]},
        plot=[],
    )
    assert qc.required_variables == ["PRES"]
    assert set(qc.qc_outputs) == {"PRES_QC", "CNDC_QC"}


def test_optional_params_get_defaults():
    # also_flag/plot are optional; window_size defaults to 50.
    qc = spike_qc(None, variables={"PRES": 2})
    assert qc.window_size == 50
    assert qc.also_flag == {}
    assert "PROFILE_NUMBER" in qc.required_variables


def test_test_depth_range_adds_depth_requirement():
    without = impossible_range_qc(None, variable_ranges={"PRES": {4: [-5, 0]}})
    assert "DEPTH" not in without.required_variables

    with_range = impossible_range_qc(
        None, variable_ranges={"PRES": {4: [-5, 0]}}, test_depth_range=[-100, 0]
    )
    assert "DEPTH" in with_range.required_variables


def test_missing_required_param_raises():
    with pytest.raises(ValueError, match="variable_ranges"):
        gross_range_qc(None, also_flag={})


def test_unknown_param_raises():
    with pytest.raises(ValueError, match="bogus"):
        impossible_date_qc(None, bogus=1)


# --- Validator is schema-aware (steps and nested QC) -----------------------------


def test_validator_flags_missing_required_step_param():
    steps = [{"name": "Data Export", "parameters": {"export_format": "netcdf"}}]
    with pytest.raises(ValueError, match="output_path"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_flags_missing_required_qc_param():
    steps = [{"name": "Apply QC", "parameters": {"qc_settings": {"gross range qc": {}}}}]
    with pytest.raises(ValueError, match="variable_ranges"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_passes_valid_qc_config():
    steps = [
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"impossible date qc": {}}},
        }
    ]
    assert check_pipeline_variables(steps, LOGGER) is True
