import numpy as np
import xarray as xr

from pelagos_py.steps.base_step import BaseStep


class Dummy(BaseStep):
    step_name = "Dummy"
    provided_variables = ["OUT"]
    parameter_schema = {"output_as": {"type": str, "default": "DEFAULT_OUT"}}


def make_step(data):
    step = Dummy("Dummy", context={"data": data})
    step.data = data
    return step


def make_data():
    return xr.Dataset(
        {
            "TEMP": ("N_MEASUREMENTS", np.array([10.0, 11.0, 12.0])),
            "TEMP_QC": ("N_MEASUREMENTS", np.array([1, 4, 1], dtype=np.int8)),
            "PRES": ("N_MEASUREMENTS", np.array([1.0, 2.0, 3.0])),
        }
    )


def test_resolve_prefers_adjusted():
    step = make_step(make_data())
    assert step.resolve("TEMP") == "TEMP"
    step.adjusted_copy("TEMP")
    assert step.resolve("TEMP") == "TEMP_ADJUSTED"
    assert step.resolve("TEMP_ADJUSTED") == "TEMP_ADJUSTED"
    assert step.resolve("PRES_ADJUSTED") == "PRES"


def test_adjusted_copy_keeps_raw_and_copies_qc():
    step = make_step(make_data())
    name = step.adjusted_copy("TEMP")
    step.data[name][:] = 0.0
    assert name == "TEMP_ADJUSTED"
    assert list(step.data["TEMP"].values) == [10.0, 11.0, 12.0]
    assert list(step.data["TEMP_ADJUSTED_QC"].values) == [1, 4, 1]
    # A second call reuses the existing adjusted variable.
    assert step.adjusted_copy("TEMP_ADJUSTED") == "TEMP_ADJUSTED"
    assert list(step.data["TEMP_ADJUSTED"].values) == [0.0, 0.0, 0.0]


def test_adjusted_copy_without_qc():
    step = make_step(make_data())
    step.adjusted_copy("PRES")
    assert "PRES_ADJUSTED" in step.data
    assert "PRES_ADJUSTED_QC" not in step.data


def test_mark_intermediate_accumulates():
    step = make_step(make_data())
    step.mark_intermediate("B", "A")
    step.mark_intermediate("A", "C")
    assert step.context["intermediate_variables"] == ["A", "B", "C"]


def test_output_variables_default():
    assert Dummy.output_variables({}) == ["OUT", "DEFAULT_OUT"]
    assert Dummy.output_variables({"output_as": ["X", "Y"], "qc_outputs": ["Q"]}) == [
        "OUT",
        "Q",
        "X",
        "Y",
    ]
