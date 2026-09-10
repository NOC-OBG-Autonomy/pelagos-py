"""Tests the file-specific config builder (src/pelagos_py/utils/config_builder.py)."""

import logging

import pytest
import yaml

from pelagos_py.utils import config_builder as cb
from pelagos_py.utils.valid_config_check import check_pipeline_variables

LOGGER = logging.getLogger("test_config_builder")

TEMPLATE = """\
pipeline:
  name: t
  description: template
steps:

# ===========================================================
#                       IMPORT DATA
# ===========================================================
  - name: Load OG1
    parameters:
      file_path: ""
    diagnostics: false

# Normalise names.
  - name: Prepare OG1
    parameters:
      bbp700_is_beta: true
    diagnostics: false

# ===========================================================
#                          CTD
# ===========================================================
  - name: Apply QC
    parameters:
      qc_settings:

        range qc:
          variable_ranges:
            CNDC: # S/m
              3: [0.5, 4.2, outside]
              4: [0.2, 4.5, outside]
            TEMP:
              4: [-2.5, 40, outside]
    diagnostics: false

# ===========================================================
#                      BACKSCATTER
# ===========================================================
# QC the beta.
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            BETA_BACKSCATTERING700:
              4: [-1.0e-4, 0.05, outside]
    diagnostics: false

  - name: BBP from Beta
    parameters:
      apply_to: BETA_BACKSCATTERING700
      output_as: BBP700
    diagnostics: false

# ===========================================================
#                        OXYGEN
# ===========================================================
  - name: "Derive Uncalibrated Phase"
    parameters:
      blue_phase_name: "BPHASE_DOXY"
    diagnostics: false

  - name: Correct Values
    parameters:
      target_variable: MOLAR_DOXY_PSAL_PRES
      output_as: MOLAR_DOXY_ADJUSTED
      append_description: Renamed from MOLAR_DOXY_PSAL_PRES.
    diagnostics: false

# ===========================================================
#                 PAR QC
# ===========================================================
  - name: Interpolate PAR
    parameters:
      par_var: DOWNWELLING_PAR
    diagnostics: false

# ===========================================================
#                 CHLOROPHYLL CORRECTIONS
# ===========================================================
  - name: CHLA Quenching
    parameters:
      method: thomalla2018
    diagnostics: false

  - name: "Data Export"
    parameters:
      output_path: "x.nc"
"""


def var(units="", all_nan=False, median=None):
    info = {"units": units, "numeric": True, "all_nan": all_nan}
    if median is not None:
        info["median"] = median
    return info


BASE = {
    "TIME": var(), "LATITUDE": var(), "LONGITUDE": var(), "PRES": var(), "TEMP": var(),
    "CNDC": var("mhos/m", median=3.6),
}


def steps_of(text):
    return [s["name"] for s in yaml.safe_load(text)["steps"]]


def ids(decs):
    return {d["id"]: d for d in decs}


def test_parse_render_round_trip():
    head, blocks, tail = cb._parse(TEMPLATE)
    assert cb._render(head, blocks, tail) == TEMPLATE
    assert [b.name for b in blocks][:3] == ["Load OG1", "Prepare OG1", "Apply QC"]


def test_full_file_needs_no_choices():
    probe = {**BASE, "BETA_BACKSCATTERING700": var(), "BPHASE_DOXY": var(), "DOWNWELLING_PAR": var()}
    decs = cb.decisions(probe)
    assert [d["id"] for d in decs] == ["oxygen"]
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert steps_of(text) == steps_of(TEMPLATE)
    assert "file_path: /data/g.nc" in text
    assert 'output_path: "/data/g_Processed.nc"' in text


def test_bbp700_as_beta_default_and_direct_choice():
    probe = {**BASE, "BBP700": var("m-1", median=1e-4), "BPHASE_DOXY": var(), "DOWNWELLING_PAR": var()}
    d = ids(cb.decisions(probe))["bbp"]
    assert d["default"] == "as_beta" and [o["key"] for o in d["options"]] == ["as_beta", "direct"]

    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert "BBP from Beta" in steps_of(text)

    text = cb.build(TEMPLATE, "/data/g.nc", probe, choices={"bbp": "direct"})
    assert "BBP from Beta" not in steps_of(text)
    assert "bbp700_is_beta: false" in text
    assert "BETA_BACKSCATTERING700" not in text.split("BACKSCATTER", 1)[1].split("OXYGEN")[0]


def test_no_backscatter_drops_section_and_quenching():
    probe = {**BASE, "BPHASE_DOXY": var(), "DOWNWELLING_PAR": var()}
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    names = steps_of(text)
    assert "BBP from Beta" not in names and "CHLA Quenching" not in names
    assert "BACKSCATTER" not in text


def test_oxygen_uses_first_real_phase_variable():
    probe = {**BASE, "BETA_BACKSCATTERING700": var(), "DOWNWELLING_PAR": var(),
             "BPHASE_DOXY": var(all_nan=True), "DPHASE_DOXY": var()}
    d = ids(cb.decisions(probe))["oxygen"]
    assert d["default"] == "phase" and "BPHASE_DOXY" in d["detail"]
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert 'blue_phase_name: "DPHASE_DOXY"' in text


def test_oxygen_shipped_keeps_only_rename():
    probe = {**BASE, "BETA_BACKSCATTERING700": var(), "DOWNWELLING_PAR": var(), "MOLAR_DOXY": var()}
    d = ids(cb.decisions(probe))["oxygen"]
    assert d["default"] == "shipped"
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert "Derive Uncalibrated Phase" not in steps_of(text)
    assert "target_variable: MOLAR_DOXY\n" in text
    assert "MOLAR_DOXY_ADJUSTED" in text


def test_oxygen_none_and_no_par_drop_sections():
    probe = {**BASE, "BETA_BACKSCATTERING700": var(), "FREQUENCY_DOXY": var()}
    d = ids(cb.decisions(probe))
    assert d["oxygen"]["default"] == "none" and "FREQUENCY_DOXY" in d["oxygen"]["detail"]
    assert "par" in d
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert "OXYGEN" not in text and "PAR QC" not in text
    assert steps_of(text) == ["Load OG1", "Prepare OG1", "Apply QC", "Apply QC", "BBP from Beta",
                              "CHLA Quenching", "Data Export"]


def test_cndc_states():
    mislabelled = {**BASE, "CNDC": var("S m-1", median=10.9)}
    assert ids(cb.decisions(mislabelled))["cndc"]["title"] == "CNDC units mislabelled"
    genuine = {**BASE, "CNDC": var("mS/cm", median=36.0)}
    assert ids(cb.decisions(genuine))["cndc"]["title"] == "CNDC in mS/cm"
    text = cb.build(TEMPLATE, "/data/g.nc", genuine)
    assert "3: [5.0, 42.0, outside]" in text and "4: [2.0, 45.0, outside]" in text
    assert "4: [-2.5, 40, outside]" in text  # TEMP untouched
    assert "cndc" not in ids(cb.decisions(BASE))


def test_missing_coordinates_reported():
    probe = {k: v for k, v in BASE.items() if k != "LATITUDE"}
    d = ids(cb.decisions({**probe, "ALATPT01": var()}))["coord_latitude"]
    assert "ALATPT01" in d["title"] and not d["options"]
    d = ids(cb.decisions(probe))["coord_latitude"]
    assert d["title"] == "LATITUDE missing"


def test_validator_credits_prepare_renames(monkeypatch):
    monkeypatch.setattr(
        "pelagos_py.utils.valid_config_check._read_file_variables",
        lambda *a, **k: ({"TIME", "LATITUDE_GPS", "LONGITUDE_GPS", "PRES", "TEMP", "CNDC", "BBP700"}, set()),
    )
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": "x.nc"}},
        {"name": "Prepare OG1", "parameters": {}},
        {"name": "Apply QC", "parameters": {"qc_settings": {"range qc": {
            "variable_ranges": {"BETA_BACKSCATTERING700": {4: [0, 1, "outside"]}}}}}},
    ]
    assert check_pipeline_variables(steps, LOGGER) is True
    steps[1]["parameters"] = {"bbp700_is_beta": False}
    with pytest.raises(ValueError, match="BETA_BACKSCATTERING700"):
        check_pipeline_variables(steps, LOGGER)
