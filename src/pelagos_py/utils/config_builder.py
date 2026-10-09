# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a file-specific pipeline config from the full template.

The template (``DEFAULT_CONFIG``) does everything; a given file
usually can't support all of it (no PAR, no optode phase, raw beta shipped as
BBP700...). :func:`decisions` inspects the file and lists what must change,
each with a default choice; :func:`build` applies the choices to the template
text, keeping its comments, so the result is still a readable config.
"""

import json
import re
from pathlib import Path

import yaml

from pelagos_py.utils import file_probe
from pelagos_py.utils.file_probe import present
from pelagos_py.utils.processing_utils import cndc_scale_factor
from pelagos_py.steps.input_output.prepare_og1 import (
    BBP_NAME, BETA_NAME, CNDC_MSCM_ABOVE, RENAMES, PrepareOG1,
)
from pelagos_py.steps.processing.deep_correction import MIN_DEEP_THRESHOLD

DEFAULT_CONFIG = Path(__file__).parents[1] / "default_config.yaml"
PHASE_CANDIDATES = ("BPHASE_DOXY", "DPHASE_DOXY", "TPHASE_DOXY")
_BAR = re.compile(r"^\s*#\s*[=~-]{5,}\s*$")
_TITLE = re.compile(r"^\s*#\s*(\S.*?)\s*$")
_STEP = re.compile(r"^  - name:")
_FIELD = {f: re.compile(rf"(?m)^(\s*{f}:).*$")
          for f in ("file_path", "output_path", "description", "out_directory")}

DEEP_STEP = 50  # dbar between the offered deep correction thresholds
RENAME_TARGETS = {"coord_latitude": "LATITUDE", "coord_longitude": "LONGITUDE",
                  "bbp": BETA_NAME, "oxygen": "MOLAR_DOXY", "par": "DOWNWELLING_PAR"}


# ----------------------------------------------------------------------------
# Template text model: the same "# ==== / # TITLE / # ====" banners the
# dashboard builder groups steps by, and one block per step (its explanatory
# comment lines plus the step itself) so both are dropped together.
# ----------------------------------------------------------------------------
class _Block:
    def __init__(self, section, banner, lines):
        self.section = section  # banner title, e.g. "CTD"
        self.banner = banner
        self.lines = lines
        body = "\n".join(line for line in lines if not line.startswith("#"))
        self.step = (yaml.safe_load(body) or [{}])[0]

    @property
    def name(self):
        return self.step.get("name")

    @property
    def params(self):
        return self.step.get("parameters") or {}

    def text(self):
        return "\n".join(self.lines)

    def sub(self, pattern, repl, count=1):
        self.lines = re.sub(pattern, repl, self.text(), count=count, flags=re.M).split("\n")


def _continues(lines, i):
    # Whether the step body carries on past a blank line at lines[i - 1].
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i < len(lines) and lines[i].startswith(" ") and not _STEP.match(lines[i])


def _is_banner(lines, i):
    return (i + 2 < len(lines) and _BAR.match(lines[i])
            and _TITLE.match(lines[i + 1]) and _BAR.match(lines[i + 2]))


def _parse(text):
    # (head, blocks, tail): text up to and including `steps:`, one _Block per step
    # (each carries its section's banner, emitted once before its first surviving
    # block), trailing text.
    lines = text.split("\n")
    try:
        start = next(i for i, line in enumerate(lines) if re.match(r"^steps:\s*$", line)) + 1
    except StopIteration:
        raise ValueError("Template has no 'steps:' list.")
    head = "\n".join(lines[:start])
    blocks, pending = [], []
    section, banner = None, []
    i = start
    while i < len(lines):
        line = lines[i]
        if _is_banner(lines, i):
            section = _TITLE.match(lines[i + 1]).group(1)
            banner = pending + lines[i:i + 3]
            pending = []
            i += 3
            continue
        if _STEP.match(line):
            body = [line]
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or (
                    not lines[i].strip() and _continues(lines, i + 1))):
                body.append(lines[i])
                i += 1
            blocks.append(_Block(section, banner, pending + body))
            pending = []
            continue
        pending.append(line)
        i += 1
    return head, blocks, pending


def _render(head, blocks, tail):
    parts, last_section = [head], None
    for b in blocks:
        if b.section is not None and b.section != last_section:
            parts.append("\n".join(b.banner))
            last_section = b.section
        parts.append(b.text())
    parts.append("\n".join(tail))
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------------
def _decision(id_, title, detail, options=(), default=None, section=None):
    # `section`: the template section the choice can drop (shown as skipped by the dashboard).
    return {"id": id_, "title": title, "detail": detail, "section": section,
            "options": [{"key": key, "label": label} for key, label in options], "default": default}


def _rename_options(real, expected):
    # One "rename:<var>" option per file variable that could stand in for `expected`.
    return [(f"rename:{v}", f"Use {v} as {expected}")
            for v in sorted(real) if not v.endswith("_QC")]


def _renames(choices):
    # {expected: source} from every "rename:<source>" choice on a missing-variable decision.
    return {RENAME_TARGETS[k]: v.split(":", 1)[1]
            for k, v in (choices or {}).items() if k in RENAME_TARGETS and v.startswith("rename:")}


def _oxygen_phase(probe):
    return next((v for v in PHASE_CANDIDATES if present(probe, v)), None)


def _oxygen_molar(probe):
    return next((v for v in ("MOLAR_DOXY", "DOXY") if present(probe, v)), None)


def _cndc_state(probe):
    # "ok" | "mislabelled" (mS/cm values under an S/m label; Prepare OG1 rescales)
    # | "relabel" (S/m values under an mS/cm label) | "mscm" (genuine mS/cm).
    info = (probe or {}).get("CNDC") or {}
    median = info.get("median")
    if median is None:
        return "ok"
    values_mscm = median > CNDC_MSCM_ABOVE
    labelled_mscm = cndc_scale_factor(info.get("units")) == 1.0
    if values_mscm and not labelled_mscm:
        return "mislabelled"
    if labelled_mscm and not values_mscm:
        return "relabel"
    return "mscm" if values_mscm else "ok"


def _coordinate_decisions(real):
    decs = []
    renames = PrepareOG1.renames_for(real)
    for expected in ("LATITUDE", "LONGITUDE"):
        if expected in real:
            continue
        src = next((s for s, d in renames.items() if d == expected), None)
        if src:
            decs.append(_decision(
                f"coord_{expected.lower()}", f"{expected} renamed from {src}",
                f"The file has no {expected}: {src} will be renamed to {expected}.",
            ))
        else:
            decs.append(_decision(
                f"coord_{expected.lower()}", f"{expected} missing",
                f"No {expected} or any known alternative ({', '.join(RENAMES[expected])}) "
                "in the file -- position QC and profile finding will fail unless it is "
                "held under another name.",
                [("none", "Leave missing")] + _rename_options(real, expected), "none",
            ))
    return decs


def _cndc_decision(probe):
    cndc = _cndc_state(probe)
    units = (probe or {}).get("CNDC", {}).get("units", "")
    if cndc == "mislabelled":
        return _decision(
            "cndc", "CNDC units mislabelled",
            f"CNDC is labelled '{units}' but its values are mS/cm: it will be scaled "
            "x0.1 to S/m so the range test and gsw see the right units.",
        )
    if cndc == "relabel":
        return _decision(
            "cndc", "CNDC units mislabelled",
            f"CNDC is labelled '{units}' but its values are S/m: it will be relabelled S/m.",
        )
    if cndc == "mscm":
        return _decision(
            "cndc", "CNDC in mS/cm",
            "CNDC is genuinely in mS/cm; left as is, with the CTD range test scaled to match.",
        )
    return None


def _bbp_decision(real):
    if BETA_NAME in real:
        return None
    if BBP_NAME in real:
        return _decision(
            "bbp", "BETA_BACKSCATTERING700 missing, BBP700 present",
            "Some files ship raw beta under BBP700 before it has been converted. Either "
            "treat BBP700 as beta (renamed to BETA_BACKSCATTERING700 and converted by "
            "'BBP from Beta') or trust it as already-converted BBP and skip the conversion.",
            [("as_beta", "Use BBP700 as raw beta and convert it"),
             ("direct", "Use BBP700 directly, skip conversion")],
            "as_beta", section="BACKSCATTER",
        )
    return _decision(
        "bbp", "No backscatter",
        "Neither BETA_BACKSCATTERING700 nor BBP700 is in the file: the Backscatter "
        "section and the CHLA Quenching step (which needs BBP) are removed, unless "
        "raw beta is held under another name.",
        [("remove", "Remove the Backscatter section")] + _rename_options(real, BETA_NAME),
        "remove", section="BACKSCATTER",
    )


def _oxygen_range_check(block):
    # The 0-1000 range tests on MOLAR_DOXY(_ADJUSTED), which apply to shipped oxygen too.
    ranges = block.params.get("qc_settings", {}).get("range qc", {}).get("variable_ranges", {})
    return bool(ranges) and set(ranges) <= {"MOLAR_DOXY", "MOLAR_DOXY_ADJUSTED"}


def _oxygen_decision(probe, real):
    phase, molar = _oxygen_phase(probe), _oxygen_molar(probe)
    opts, notes = [], []
    if phase:
        opts.append(("phase", f"Recompute oxygen from {phase}"))
    if molar:
        opts.append(("shipped", f"Use {molar} as shipped"))
    opts.append(("none", "No oxygen processing"))
    if not phase and not molar:
        opts += _rename_options(real, "MOLAR_DOXY")
    if "FREQUENCY_DOXY" in real:
        notes.append("FREQUENCY_DOXY (SBE43 frequency) is present but no step converts it yet.")
    empty = [v for v in PHASE_CANDIDATES if v in (probe or {}) and v not in real]
    if empty:
        notes.append(f"{', '.join(empty)} present but all-NaN.")
    if phase:
        title = f"Oxygen from {phase}"
        detail = "Full optode chain: phase -> pressure correction -> shift -> concentration."
    elif molar:
        title = "No optode phase; oxygen as shipped"
        detail = f"Only {molar} is available: the derivation steps are dropped and it is exposed as MOLAR_DOXY_ADJUSTED."
    else:
        title = "No oxygen"
        detail = ("No optode phase or oxygen concentration in the file: the Oxygen section is "
                  "removed, unless the concentration is held under another name.")
    # Known issue: "phase" is the default whenever it exists, but the template's SVU coefficients
    # are for one optode (aa4831). Revisit once it's decided how optode coefficients are supplied.
    # A lone "none" option is no choice: shown as automatic (default_choices skips it).
    return _decision("oxygen", title, " ".join([detail] + notes),
                     opts if len(opts) > 1 else (), opts[0][0], section="OXYGEN")


def _par_decision(probe, real):
    if present(probe, "DOWNWELLING_PAR"):
        return None
    extra = " (DPAR is present but in a different unit and is not used.)" if "DPAR" in real else ""
    return _decision(
        "par", "No PAR",
        f"DOWNWELLING_PAR is missing: the PAR QC section is removed, unless PAR is "
        f"held under another name.{extra}",
        [("remove", "Remove the PAR QC section")] + _rename_options(real, "DOWNWELLING_PAR"),
        "remove", section="PAR QC",
    )


def _deep_decision(probe):
    dives = ((probe or {}).get("PRES") or {}).get("dive_depths")
    if not dives:
        return None
    # Depth a tenth of the dives reach: enough profiles for Deep Correction without one-off deep dives.
    reach = sorted(dives, reverse=True)[len(dives) // 10]
    # Never suggest shallower than MIN_DEEP_THRESHOLD, but use it while some dives get past it.
    suggested = max(round(reach / DEEP_STEP) * DEEP_STEP - DEEP_STEP, MIN_DEEP_THRESHOLD)
    deepest_option = int(reach // DEEP_STEP) * DEEP_STEP
    depths = range(deepest_option, DEEP_STEP, -DEEP_STEP)
    options = [("skip", "Skip deep correction")] + [(str(d), f"Use data below {d} dbar") for d in depths]
    if reach > MIN_DEEP_THRESHOLD:
        return _decision(
            "deep", f"Deep correction below {suggested} dbar",
            f"Enough dives reach {reach:.0f} dbar, so the CHLA dark value is estimated "
            f"from data below {suggested} dbar.",
            options, str(suggested),
        )
    return _decision(
        "deep", "Dives too shallow for deep correction",
        f"Enough dives only reach {reach:.0f} dbar, too shallow (under {MIN_DEEP_THRESHOLD} dbar) "
        "to trust a CHLA dark value, so Deep Correction is removed.",
        options if len(options) > 1 else (), "skip",
    )


def decisions(probe):
    """What the template must change for this file, as a list of
    ``{id, title, detail, options, default}``; ``options`` is empty for an
    automatic fix that is only reported."""
    real = {v for v in (probe or {}) if present(probe, v)}
    decs = _coordinate_decisions(real) + [
        _cndc_decision(probe),
        _bbp_decision(real),
        _oxygen_decision(probe, real),
        _par_decision(probe, real),
        _deep_decision(probe),
    ]
    return [d for d in decs if d is not None]


def default_choices(decs):
    return {d["id"]: d["default"] for d in decs if d["options"]}


def ask_choices(decs, ask=input):
    # Terminal version of the dashboard's Build panel; Enter keeps the default.
    choices = {}
    for d in decs:
        print(f"\n{d['title']}\n  {d['detail']}")
        if not d["options"]:
            continue
        # Rename options list every file variable, too many to number; typed by name instead.
        listed = [o for o in d["options"] if not o["key"].startswith("rename:")]
        renamable = {o["key"].removeprefix("rename:") for o in d["options"] if o not in listed}
        for n, option in enumerate(listed, 1):
            default = "  (default)" if option["key"] == d["default"] else ""
            print(f"  {n}) {option['label']}{default}")
        prompt = "Choice (Enter for default"
        if renamable:
            prompt += ", or the name of a file variable to use instead"
        prompt += "): "
        while True:
            answer = ask(prompt).strip()
            if not answer:
                choices[d["id"]] = d["default"]
                break
            if answer.isdigit() and 1 <= int(answer) <= len(listed):
                choices[d["id"]] = listed[int(answer) - 1]["key"]
                break
            if answer in renamable:
                choices[d["id"]] = f"rename:{answer}"
                break
            print("  Not an option, try again.")
    return choices


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
def _times_ten(match):
    # "[0.5, 4.2, outside]" -> "[5.0, 42.0, outside]"
    items = []
    for item in match.group(1).split(","):
        item = item.strip()
        if re.fullmatch(r"-?[\d.]+", item):
            item = str(float(item) * 10)
        items.append(item)
    return "[" + ", ".join(items) + "]"


def _scale_cndc_ranges(block):
    # x10 the CNDC bands of the CTD range test (template values are S/m).
    lines, in_cndc = [], False
    for line in block.lines:
        if re.match(r"^\s*CNDC:", line):
            in_cndc = True
        elif in_cndc and re.match(r"^\s*\d+:\s*\[", line):
            line = re.sub(r"\[([^\]]*)\]", _times_ten, line, count=1)
        elif in_cndc and not re.match(r"^\s{14,}", line):
            in_cndc = False
        lines.append(line)
    block.lines = lines


def _add_renames(block, renames):
    # Insert a `renames:` mapping under `bbp700_is_beta:` in the Prepare OG1 step.
    lines = []
    for line in block.lines:
        lines.append(line)
        if line.strip().startswith("bbp700_is_beta:"):
            indent = line[:len(line) - len(line.lstrip())]
            lines.append(f"{indent}renames:  # file's name for a missing OG1 variable")
            for expected, source in renames.items():
                lines.append(f"{indent}  {expected}: {source}")
    block.lines = lines


def _yaml_string(value):
    # JSON strings are valid YAML, and quoting keeps "#", ": " and backslashes literal.
    return json.dumps(str(value))


def build(template_text, file_path, probe=None, choices=None, description=None, output_path=None):
    """The template adapted to ``file_path``: paths patched in and each
    decision's choice applied (defaults where ``choices`` doesn't say)."""
    probe = probe if probe is not None else file_probe.probe_file(file_path)
    decs = decisions(probe)
    choices = {**default_choices(decs), **(choices or {})}
    ids = {d["id"] for d in decs}
    head, blocks, tail = _parse(template_text)

    def drop(pred):
        blocks[:] = [b for b in blocks if not pred(b)]

    def set_field(value, comment=""):
        # A function, not a "\1 value" template, so backslashes in e.g. Windows paths stay literal.
        return lambda m: f"{m.group(1)} {_yaml_string(value)}{comment}"

    stem = Path(file_path).stem
    output_path = output_path or str(Path(file_path).with_name(f"{stem}_Processed.nc"))
    description = description or f"Pipeline built for {Path(file_path).name}."
    head = _FIELD["description"].sub(set_field(description), head, count=1)
    head = _FIELD["out_directory"].sub(set_field(f"{Path(file_path).parent}/"), head, count=1)
    for b in blocks:
        if b.name == "Load OG1":
            b.sub(_FIELD["file_path"].pattern, set_field(file_path, "  # Path to the input NetCDF file"))
        if b.name == "Data Export":
            b.sub(_FIELD["output_path"].pattern, set_field(output_path))

    if _cndc_state(probe) == "mscm":
        for b in blocks:
            ranges = b.params.get("qc_settings", {}).get("range qc", {}).get("variable_ranges", {})
            if b.section == "CTD" and "CNDC" in ranges:
                _scale_cndc_ranges(b)

    renames = _renames(choices)
    if renames:
        for b in blocks:
            if b.name == "Prepare OG1":
                _add_renames(b, renames)

    bbp = choices.get("bbp") if "bbp" in ids else None
    if bbp == "direct":
        drop(lambda b: b.name == "BBP from Beta")
        for b in blocks:
            if b.name == "Prepare OG1":
                b.sub(r"(?m)^(\s*bbp700_is_beta:).*$", r"\1 false")
            elif b.section == "BACKSCATTER":
                b.sub(BETA_NAME, BBP_NAME, count=0)
    elif bbp == "remove":  # no backscatter at all
        drop(lambda b: b.section == "BACKSCATTER" or b.name == "CHLA Quenching")

    oxygen = choices.get("oxygen", "none")
    if oxygen.startswith("rename:"):  # renamed to MOLAR_DOXY by Prepare OG1, then used as shipped
        oxygen = "shipped"
    phase = _oxygen_phase(probe)
    if oxygen == "phase" and phase:
        for b in blocks:
            if b.name == "Derive Uncalibrated Phase":
                b.sub(r'(?m)^(\s*blue_phase_name:).*$', rf'\1 "{phase}"')
    elif oxygen == "shipped":
        drop(lambda b: b.section == "OXYGEN" and b.name != "Correct Values" and not _oxygen_range_check(b))
        for b in blocks:
            if b.section == "OXYGEN":
                b.sub(r"(?m)^(\s*target_variable:).*$", r"\1 MOLAR_DOXY")
                b.sub(r"(?m)^(\s*append_description:).*$", r"\1 Shipped MOLAR_DOXY, renamed.")
    else:
        drop(lambda b: b.section == "OXYGEN")

    if choices.get("par") == "remove":
        drop(lambda b: b.section == "PAR QC")

    deep = choices.get("deep") if "deep" in ids else None
    if deep == "skip":
        drop(lambda b: b.name == "Deep Correction")
    elif deep:
        for b in blocks:
            if b.name == "Deep Correction":
                b.sub(r"(?m)^(\s*depth_threshold:)\s*[^\s#]+", rf"\g<1> {deep}")

    return _render(head, blocks, tail)
