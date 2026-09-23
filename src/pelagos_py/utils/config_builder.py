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

The template (``dashboard/configs/default.yaml``) does everything; a given file
usually can't support all of it (no PAR, no optode phase, raw beta shipped as
BBP700...). :func:`decisions` inspects the file and lists what must change,
each with a default choice; :func:`build` applies the choices to the template
text, keeping its comments, so the result is still a readable config.
"""

import re
from pathlib import Path

import yaml

from pelagos_py.utils import file_probe
from pelagos_py.utils.file_probe import present
from pelagos_py.utils.processing_utils import cndc_scale_factor
from pelagos_py.steps.input_output.prepare_og1 import (
    BBP_NAME, BETA_NAME, CNDC_MSCM_ABOVE, RENAMES, PrepareOG1,
)

PHASE_CANDIDATES = ("BPHASE_DOXY", "DPHASE_DOXY", "TPHASE_DOXY")
_BAR = re.compile(r"^\s*#\s*[=~-]{5,}\s*$")
_TITLE = re.compile(r"^\s*#\s*(\S.*?)\s*$")
_STEP = re.compile(r"^  - name:")
_FIELD = {f: re.compile(rf"(?m)^(\s*{f}:).*$") for f in ("file_path", "output_path", "description")}


# ----------------------------------------------------------------------------
# Template text model: the same "# ==== / # TITLE / # ====" banners the
# dashboard builder groups steps by, and one block per step (its explanatory
# comment lines plus the step itself) so both are dropped together.
# ----------------------------------------------------------------------------
class _Block:
    def __init__(self, section, lines):
        self.section = section
        self.lines = lines
        body = "\n".join(l for l in lines if not l.startswith("#"))
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


def _parse(text):
    """``(head, blocks, tail)``: text up to and including ``steps:``, one
    _Block per step (banners are kept as the block's ``section`` and emitted
    once before its first surviving block), trailing text."""
    lines = text.split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if re.match(r"^steps:\s*$", l)) + 1
    except StopIteration:
        raise ValueError("Template has no 'steps:' list.")
    head = "\n".join(lines[:start])
    blocks, pending, section = [], [], None
    i = start
    while i < len(lines):
        line = lines[i]
        if _BAR.match(line) and i + 2 < len(lines) and _BAR.match(lines[i + 2]) and _TITLE.match(lines[i + 1]):
            section = (_TITLE.match(lines[i + 1]).group(1), pending + lines[i:i + 3])
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
            blocks.append(_Block(section, pending + body))
            pending = []
            continue
        pending.append(line)
        i += 1
    return head, blocks, pending


def _render(head, blocks, tail):
    out, last_section = [head], None
    for b in blocks:
        if b.section is not None and b.section is not last_section:
            out.append("\n".join(b.section[1]))
            last_section = b.section
        out.append(b.text())
    out.append("\n".join(tail))
    return "\n".join(out)


def _section(block):
    return block.section[0] if block.section else None


# ----------------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------------
def _decision(id_, title, detail, options=(), default=None):
    return {"id": id_, "title": title, "detail": detail,
            "options": [{"key": k, "label": l} for k, l in options], "default": default}


def _rename_options(real, canonical):
    # One "rename:<var>" option per file variable that could stand in for `canonical`.
    return [(f"rename:{v}", f"Use {v} as {canonical}")
            for v in sorted(real) if not v.endswith("_QC")]


def _renames(choices):
    # {canonical: source} from every "rename:<source>" choice on a missing-variable decision.
    return {RENAME_TARGETS[k]: v.split(":", 1)[1]
            for k, v in (choices or {}).items() if k in RENAME_TARGETS and v.startswith("rename:")}


RENAME_TARGETS = {"coord_latitude": "LATITUDE", "coord_longitude": "LONGITUDE",
                  "bbp": BETA_NAME, "oxygen": "MOLAR_DOXY", "par": "DOWNWELLING_PAR"}


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


def decisions(probe):
    """What the template must change for this file, as a list of
    ``{id, title, detail, options, default}``; ``options`` is empty for an
    automatic fix that is only reported."""
    real = {v for v in (probe or {}) if present(probe, v)}
    out = []

    renames = PrepareOG1.renames_for(real)
    for canonical in ("LATITUDE", "LONGITUDE"):
        src = next((s for s, d in renames.items() if d == canonical), None)
        if canonical in real:
            continue
        if src:
            out.append(_decision(
                f"coord_{canonical.lower()}", f"{canonical} renamed from {src}",
                f"The file has no {canonical}: {src} will be renamed to {canonical}.",
            ))
        else:
            out.append(_decision(
                f"coord_{canonical.lower()}", f"{canonical} missing",
                f"No {canonical} or any known alternative ({', '.join(RENAMES[canonical])}) "
                "in the file -- position QC and profile finding will fail unless it is "
                "held under another name.",
                [("none", "Leave missing")] + _rename_options(real, canonical), "none",
            ))

    cndc = _cndc_state(probe)
    units = (probe or {}).get("CNDC", {}).get("units", "")
    if cndc == "mislabelled":
        out.append(_decision(
            "cndc", "CNDC units mislabelled",
            f"CNDC is labelled '{units}' but its values are mS/cm: it will be scaled "
            "x0.1 to S/m so the range test and gsw see the right units.",
        ))
    elif cndc == "relabel":
        out.append(_decision(
            "cndc", "CNDC units mislabelled",
            f"CNDC is labelled '{units}' but its values are S/m: it will be relabelled S/m.",
        ))
    elif cndc == "mscm":
        out.append(_decision(
            "cndc", "CNDC in mS/cm",
            "CNDC is genuinely in mS/cm; left as is, with the CTD range test scaled to match.",
        ))

    if BETA_NAME in real:
        pass
    elif BBP_NAME in real:
        out.append(_decision(
            "bbp", "BETA_BACKSCATTERING700 missing, BBP700 present",
            "Some files ship raw beta under BBP700 before it has been converted. Either "
            "treat BBP700 as beta (renamed to BETA_BACKSCATTERING700 and converted by "
            "'BBP from Beta') or trust it as already-converted BBP and skip the conversion.",
            [("as_beta", "Use BBP700 as raw beta and convert it"),
             ("direct", "Use BBP700 directly, skip conversion")],
            "as_beta",
        ))
    else:
        out.append(_decision(
            "bbp", "No backscatter",
            "Neither BETA_BACKSCATTERING700 nor BBP700 is in the file: the Backscatter "
            "section and the CHLA Quenching step (which needs BBP) are removed, unless "
            "raw beta is held under another name.",
            [("remove", "Remove the Backscatter section")] + _rename_options(real, BETA_NAME),
            "remove",
        ))

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
    # A lone "none" option is no choice: shown as automatic (default_choices skips it).
    out.append(_decision("oxygen", title, " ".join([detail] + notes),
                         opts if len(opts) > 1 else (), opts[0][0]))

    if not present(probe, "DOWNWELLING_PAR"):
        extra = " (DPAR is present but in a different unit and is not used.)" if "DPAR" in real else ""
        out.append(_decision(
            "par", "No PAR",
            f"DOWNWELLING_PAR is missing: the PAR QC section is removed, unless PAR is "
            f"held under another name.{extra}",
            [("remove", "Remove the PAR QC section")] + _rename_options(real, "DOWNWELLING_PAR"),
            "remove",
        ))
    return out


def default_choices(decs):
    return {d["id"]: d["default"] for d in decs if d["options"]}


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
def _scale_cndc_ranges(block):
    # x10 the CNDC bands of the CTD range test (template values are S/m).
    lines, in_cndc = [], False
    for l in block.lines:
        if re.match(r"^\s*CNDC:", l):
            in_cndc = True
        elif in_cndc and re.match(r"^\s*\d+:\s*\[", l):
            l = re.sub(r"\[([^\]]*)\]", lambda m: "[" + ", ".join(
                str(float(x) * 10) if re.fullmatch(r"-?[\d.]+", x.strip()) else x.strip()
                for x in m.group(1).split(",")) + "]", l, count=1)
        elif in_cndc and not re.match(r"^\s{14,}", l):
            in_cndc = False
        lines.append(l)
    block.lines = lines


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

    stem = Path(file_path).stem
    output_path = output_path or str(Path(file_path).with_name(f"{stem}_Processed.nc"))
    head = _FIELD["description"].sub(
        rf"\1 {description or f'Pipeline built for {Path(file_path).name}.'}", head, count=1)
    for b in blocks:
        if b.name == "Load OG1":
            b.sub(_FIELD["file_path"].pattern, rf"\1 {file_path}  # Path to the input NetCDF file")
        if b.name == "Data Export":
            b.sub(_FIELD["output_path"].pattern, rf'\1 "{output_path}"')

    if _cndc_state(probe) == "mscm":
        for b in blocks:
            if _section(b) == "CTD" and "CNDC" in (b.params.get("qc_settings", {}).get("range qc", {}).get("variable_ranges", {})):
                _scale_cndc_ranges(b)

    renames = _renames(choices)
    if renames:
        for b in blocks:
            if b.name == "Prepare OG1":
                b.sub(r"(?m)^(\s*)(bbp700_is_beta:.*)$", lambda m: m.group(1) + m.group(2)
                      + "\n" + m.group(1) + "renames:  # file's name for a missing OG1 variable\n"
                      + "".join(f"{m.group(1)}  {k}: {v}\n" for k, v in renames.items()).rstrip("\n"))

    bbp = choices.get("bbp") if "bbp" in ids else None
    if bbp == "direct":
        drop(lambda b: b.name == "BBP from Beta")
        for b in blocks:
            if b.name == "Prepare OG1":
                b.sub(r"(?m)^(\s*bbp700_is_beta:).*$", r"\1 false")
            elif _section(b) == "BACKSCATTER":
                b.sub(BETA_NAME, BBP_NAME, count=0)
    elif bbp == "remove":  # no backscatter at all
        drop(lambda b: _section(b) == "BACKSCATTER" or b.name == "CHLA Quenching")

    oxygen = choices.get("oxygen", "none")
    if oxygen.startswith("rename:"):  # renamed to MOLAR_DOXY by Prepare OG1, then used as shipped
        oxygen = "shipped"
    phase = _oxygen_phase(probe)
    if oxygen == "phase" and phase:
        for b in blocks:
            if b.name == "Derive Uncalibrated Phase":
                b.sub(r'(?m)^(\s*blue_phase_name:).*$', rf'\1 "{phase}"')
    elif oxygen == "shipped":
        drop(lambda b: _section(b) == "OXYGEN" and b.name != "Correct Values")
        for b in blocks:
            if _section(b) == "OXYGEN":
                b.sub(r"(?m)^(\s*target_variable:).*$", r"\1 MOLAR_DOXY")
                b.sub(r"(?m)^(\s*append_description:).*$", r"\1 Shipped MOLAR_DOXY, renamed.")
    else:
        drop(lambda b: _section(b) == "OXYGEN")

    if choices.get("par") == "remove":
        drop(lambda b: _section(b) == "PAR QC")

    return _render(head, blocks, tail)
