import pytest

from pelagos_py.utils import parameter_spec


SCHEMA = {
    "velocity_threshold": {
        "type": float,
        "default": 0.033,
        "description": "vertical velocity",
        "min": 0.0,
        "max": 0.5,
        "unit": "m/s",
    },
    "file_path": {
        "type": str,
        "required": True,
        "description": "input path",
    },
    "is_propelled": {
        "type": [str, bool],
        "default": "auto",
        "description": "transect phase",
    },
}


def test_is_required():
    assert parameter_spec.is_required({"required": True})
    assert parameter_spec.is_required({"type": str})  # no default => required
    assert not parameter_spec.is_required({"default": 5})
    assert not parameter_spec.is_required({"default": None})  # explicit default


def test_resolve_applies_defaults_and_user_values():
    resolved = parameter_spec.resolve(
        SCHEMA, {"file_path": "x.nc", "velocity_threshold": 0.1}
    )
    assert resolved == {
        "velocity_threshold": 0.1,  # user value wins
        "file_path": "x.nc",
        "is_propelled": "auto",  # default applied
    }


def test_resolve_missing_required_raises():
    with pytest.raises(ValueError, match="file_path"):
        parameter_spec.resolve(SCHEMA, {}, label="Load OG1")


def test_resolve_rejects_unknown_params():
    # Strict: an undeclared parameter (e.g. a typo) raises.
    with pytest.raises(ValueError, match="unknown parameter"):
        parameter_spec.resolve(SCHEMA, {"file_path": "x.nc", "profile_lenght": 5})


def test_resolve_allows_whitelisted_framework_keys():
    # Framework/mixin keys are permitted via allowed_extra and not returned.
    resolved = parameter_spec.resolve(
        SCHEMA,
        {"file_path": "x.nc", "qc_handling_settings": {"a": 1}},
        allowed_extra={"qc_handling_settings"},
    )
    assert "qc_handling_settings" not in resolved
    assert resolved["file_path"] == "x.nc"


def test_describe_is_json_friendly():
    described = {d["name"]: d for d in parameter_spec.describe(SCHEMA)}

    assert described["velocity_threshold"]["type"] == "float"
    assert described["velocity_threshold"]["default"] == 0.033
    assert described["velocity_threshold"]["required"] is False
    assert described["velocity_threshold"]["unit"] == "m/s"

    assert described["file_path"]["required"] is True
    assert "default" not in described["file_path"]  # required => no default key

    # Union types render as a list of names.
    assert described["is_propelled"]["type"] == ["str", "bool"]


def test_step_describe_parameters_classmethod():
    # describe_parameters works without instantiating the step (dashboard use).
    from pelagos_py.steps.processing.find_profiles import FindProfilesStep

    described = {d["name"]: d for d in FindProfilesStep.describe_parameters()}
    assert "velocity_threshold" in described
    assert described["depth_column"]["default"] == "PRES"
