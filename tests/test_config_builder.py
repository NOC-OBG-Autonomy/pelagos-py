"""Tests the file-specific config builder (src/pelagos_py/utils/config_builder.py)."""

import logging
from pathlib import Path

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
  - name: Deep Correction
    parameters:
      apply_to: CHLA
      depth_threshold: 950     # Only use data below this depth
    diagnostics: false

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
    "TIME": var(),
    "LATITUDE": var(),
    "LONGITUDE": var(),
    "PRES": var(),
    "TEMP": var(),
    "CNDC": var("mhos/m", median=3.6),
}


def steps_of(text):
    return [s["name"] for s in yaml.safe_load(text)["steps"]]


def ids(decs):
    return {d["id"]: d for d in decs}


def option_keys(decision):
    return [o["key"] for o in decision["options"]]


def test_parse_render_round_trip():
    head, blocks, tail = cb._parse(TEMPLATE)
    assert cb._render(head, blocks, tail) == TEMPLATE
    assert [b.name for b in blocks][:3] == ["Load OG1", "Prepare OG1", "Apply QC"]


def test_full_file_needs_no_choices():
    probe = {
        **BASE,
        "BETA_BACKSCATTERING700": var(),
        "BPHASE_DOXY": var(),
        "DOWNWELLING_PAR": var(),
    }
    decs = cb.decisions(probe)
    assert [d["id"] for d in decs] == ["oxygen"]
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert steps_of(text) == steps_of(TEMPLATE)
    assert 'file_path: "/data/g.nc"' in text
    assert 'output_path: "/data/g_Processed.nc"' in text


def test_bbp700_as_beta_default_and_direct_choice():
    probe = {
        **BASE,
        "BBP700": var("m-1", median=1e-4),
        "BPHASE_DOXY": var(),
        "DOWNWELLING_PAR": var(),
    }
    d = ids(cb.decisions(probe))["bbp"]
    assert d["default"] == "as_beta" and option_keys(d) == ["as_beta", "direct"]

    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert "BBP from Beta" in steps_of(text)

    text = cb.build(TEMPLATE, "/data/g.nc", probe, choices={"bbp": "direct"})
    assert "BBP from Beta" not in steps_of(text)
    assert "bbp700_is_beta: false" in text
    assert (
        "BETA_BACKSCATTERING700"
        not in text.split("BACKSCATTER", 1)[1].split("OXYGEN")[0]
    )


def test_no_backscatter_drops_section_and_quenching():
    probe = {**BASE, "BPHASE_DOXY": var(), "DOWNWELLING_PAR": var()}
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    names = steps_of(text)
    assert "BBP from Beta" not in names and "CHLA Quenching" not in names
    assert "BACKSCATTER" not in text


def test_oxygen_uses_first_real_phase_variable():
    probe = {
        **BASE,
        "BETA_BACKSCATTERING700": var(),
        "DOWNWELLING_PAR": var(),
        "BPHASE_DOXY": var(all_nan=True),
        "DPHASE_DOXY": var(),
    }
    d = ids(cb.decisions(probe))["oxygen"]
    assert d["default"] == "phase" and "BPHASE_DOXY" in d["detail"]
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert 'blue_phase_name: "DPHASE_DOXY"' in text


def test_oxygen_shipped_keeps_only_rename():
    probe = {
        **BASE,
        "BETA_BACKSCATTERING700": var(),
        "DOWNWELLING_PAR": var(),
        "MOLAR_DOXY": var(),
    }
    d = ids(cb.decisions(probe))["oxygen"]
    assert d["default"] == "shipped"
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert "Derive Uncalibrated Phase" not in steps_of(text)
    assert "target_variable: MOLAR_DOXY\n" in text
    assert "MOLAR_DOXY_ADJUSTED" in text


def test_oxygen_shipped_keeps_range_checks_from_real_template():
    template = (Path(cb.__file__).parents[1] / "default_config.yaml").read_text()
    probe = {
        **BASE,
        "BETA_BACKSCATTERING700": var(),
        "DOWNWELLING_PAR": var(),
        "MOLAR_DOXY": var(),
    }
    text = cb.build(template, "/data/g.nc", probe)
    assert "            MOLAR_DOXY:\n              4: [0, 1000, outside]" in text
    assert (
        "            MOLAR_DOXY_ADJUSTED:\n              4: [0, 1000, outside]" in text
    )
    assert "UNCAL_PHASE_DOXY_PCORR" not in text


def test_oxygen_none_and_no_par_drop_sections():
    probe = {**BASE, "BETA_BACKSCATTERING700": var(), "FREQUENCY_DOXY": var()}
    d = ids(cb.decisions(probe))
    assert (
        d["oxygen"]["default"] == "none" and "FREQUENCY_DOXY" in d["oxygen"]["detail"]
    )
    assert "par" in d
    text = cb.build(TEMPLATE, "/data/g.nc", probe)
    assert "OXYGEN" not in text and "PAR QC" not in text
    assert steps_of(text) == [
        "Load OG1",
        "Prepare OG1",
        "Apply QC",
        "Apply QC",
        "BBP from Beta",
        "Deep Correction",
        "CHLA Quenching",
        "Data Export",
    ]


def test_missing_par_can_be_renamed_from_another_variable():
    probe = {**BASE, "BETA_BACKSCATTERING700": var(), "PAR": var(), "PAR_QC": var()}
    d = ids(cb.decisions(probe))["par"]
    assert d["default"] == "remove"
    assert option_keys(d)[:2] == ["remove", "rename:BETA_BACKSCATTERING700"]
    assert "rename:PAR_QC" not in option_keys(d)
    text = cb.build(TEMPLATE, "/data/g.nc", probe, choices={"par": "rename:PAR"})
    assert "PAR QC" in text and "Interpolate PAR" in steps_of(text)
    prep = next(s for s in yaml.safe_load(text)["steps"] if s["name"] == "Prepare OG1")
    assert prep["parameters"] == {
        "bbp700_is_beta": True,
        "renames": {"DOWNWELLING_PAR": "PAR"},
    }


def test_missing_oxygen_can_be_renamed_to_molar_doxy():
    probe = {**BASE, "OXY_UMOL": var()}
    d = ids(cb.decisions(probe))["oxygen"]
    assert option_keys(d)[0] == "none"
    assert "rename:OXY_UMOL" in option_keys(d)
    assert "rename:" not in str(
        ids(cb.decisions({**BASE, "DOXY": var()}))["oxygen"]["options"]
    )
    text = cb.build(
        TEMPLATE, "/data/g.nc", probe, choices={"oxygen": "rename:OXY_UMOL"}
    )
    assert "MOLAR_DOXY: OXY_UMOL" in text and "MOLAR_DOXY_ADJUSTED" in text
    assert "Derive Uncalibrated Phase" not in steps_of(
        text
    ) and "Correct Values" in steps_of(text)


def test_missing_beta_can_be_renamed_instead_of_dropped():
    probe = {**BASE, "VSF700": var()}
    text = cb.build(TEMPLATE, "/data/g.nc", probe, choices={"bbp": "rename:VSF700"})
    assert "BBP from Beta" in steps_of(text) and "CHLA Quenching" in steps_of(text)
    assert "BETA_BACKSCATTERING700: VSF700" in text


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
    assert d["title"] == "LATITUDE missing" and d["default"] == "none"
    assert "rename:TEMP" in option_keys(d)


def test_ask_choices_takes_numbers_defaults_and_variable_names():
    probe = {**BASE, "BBP700": var(), "PAR": var()}
    answers = iter(["9", "2", "", "PAR"])  # bad number is asked again
    choices = cb.ask_choices(cb.decisions(probe), ask=lambda prompt: next(answers))
    assert choices == {"bbp": "direct", "oxygen": "none", "par": "rename:PAR"}


def test_validator_credits_prepare_renames(monkeypatch):
    monkeypatch.setattr(
        "pelagos_py.utils.valid_config_check._read_file_variables",
        lambda *a, **k: (
            {"TIME", "LATITUDE_GPS", "LONGITUDE_GPS", "PRES", "TEMP", "CNDC", "BBP700"},
            set(),
        ),
    )
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": "x.nc"}},
        {"name": "Prepare OG1", "parameters": {}},
        {
            "name": "Apply QC",
            "parameters": {
                "qc_settings": {
                    "range qc": {
                        "variable_ranges": {
                            "BETA_BACKSCATTERING700": {4: [0, 1, "outside"]}
                        }
                    }
                }
            },
        },
    ]
    assert check_pipeline_variables(steps, LOGGER) is True
    steps[1]["parameters"] = {"bbp700_is_beta": False}
    with pytest.raises(ValueError, match="BETA_BACKSCATTERING700"):
        check_pipeline_variables(steps, LOGGER)


def test_paths_with_backslashes_and_hashes_survive():
    path = r"C:\Users\me\glider #1.nc"
    config = yaml.safe_load(cb.build(TEMPLATE, path, BASE))
    load = next(s for s in config["steps"] if s["name"] == "Load OG1")
    assert load["parameters"]["file_path"] == path


def dives(*depths):
    return {**BASE, "PRES": {**var(), "dive_depths": list(depths)}}


def test_deep_threshold_follows_how_deep_enough_dives_go():
    assert ids(cb.decisions(dives(*[1000] * 20)))["deep"]["default"] == "950"
    assert ids(cb.decisions(dives(*[795] * 20)))["deep"]["default"] == "750"
    # One dive in 30 to 1000 m isn't enough; the depth a tenth of them reach sets it.
    assert ids(cb.decisions(dives(1000, *[400] * 29)))["deep"]["default"] == "350"
    assert ids(cb.decisions(dives(*[306] * 20)))["deep"]["default"] == "300"


def test_deep_threshold_applied_or_overridden():
    probe = dives(*[800] * 20)
    assert "depth_threshold: 750     # Only use data below this depth" in cb.build(
        TEMPLATE, "/g.nc", probe
    )
    text = cb.build(TEMPLATE, "/g.nc", probe, {"deep": "500"})
    assert "depth_threshold: 500" in text
    assert "Deep Correction" not in steps_of(
        cb.build(TEMPLATE, "/g.nc", probe, {"deep": "skip"})
    )


def test_shallow_dives_skip_deep_correction_by_default():
    d = ids(cb.decisions(dives(*[290] * 20)))["deep"]
    assert d["default"] == "skip"
    assert option_keys(d) == ["skip", "250", "200", "150", "100"]
    assert "Deep Correction" not in steps_of(
        cb.build(TEMPLATE, "/g.nc", dives(*[290] * 20))
    )
    assert "deep" not in ids(cb.decisions(BASE))
