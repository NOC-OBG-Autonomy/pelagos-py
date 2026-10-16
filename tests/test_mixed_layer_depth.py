from pelagos_py.steps.processing.mixed_layer_depth import MixedLayerDepthStep

import numpy as np
import xarray as xr


def make_context():
    depth = np.tile(np.linspace(0, 100, 101), 2)
    # Raw CONS_TEMP is well mixed throughout; the adjusted one has a step at 40 m.
    cons_temp = np.full(depth.size, 15.0)
    cons_temp_adjusted = np.where(depth < 40, 15.0, 10.0)
    ds = xr.Dataset(
        {
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.repeat([1.0, 2.0], 101)),
            "DEPTH_ADJUSTED": ("N_MEASUREMENTS", depth),
            "ASAL_ADJUSTED": ("N_MEASUREMENTS", np.full(depth.size, 35.0)),
            "CONS_TEMP": ("N_MEASUREMENTS", cons_temp),
            "CONS_TEMP_ADJUSTED": ("N_MEASUREMENTS", cons_temp_adjusted),
        }
    )
    for var in list(ds.data_vars):
        ds[f"{var}_QC"] = ("N_MEASUREMENTS", np.ones(depth.size, dtype=int))
    return {"data": ds, "global_parameters": {}}


def test_density_method_uses_adjusted_inputs():
    step = MixedLayerDepthStep(
        name="Mixed Layer Depth",
        parameters={"method": "density"},
        diagnostics=False,
        context=make_context(),
    )
    data = step.run()["data"]

    assert np.allclose(data["MLD"].values, 40.0)
    assert set(np.unique(data["MLD_BOOL"].values)) == {0.0, 1.0}
