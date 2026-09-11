# This file is part of the NOC Autonomy pelagos_py.
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

"""Checks the format of a file against OG1/CF standards and reports the result.

A short pass/fail summary is always logged; the full report is printed when
``diagnostics`` is on, and saved to disk when ``output_type`` names a format.
"""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step

#### Custom imports ####
from compliance_checker.runner import ComplianceChecker, CheckSuite, stdout_redirector
from pathlib import Path

# The compliance checker's strictness levels map to integer score limits;
# "lenient" keeps every priority (1=high ... 3=low) in the report.
_LENIENT_LIMIT = 3

_STANDARD_LABELS = {"cf": "CF", "og": "OG1"}


def standard_label(name):
    return _STANDARD_LABELS.get(name.lower(), name.upper())


def join_labels(labels):
    return " and ".join(labels) if len(labels) <= 2 else ", ".join(labels[:-1]) + f" and {labels[-1]}"


def console_summary(checker_name, result, passed, top=3):
    """One-line summary of a checker's result: score, issue count and the worst checks."""
    scored = result.get("scored_points")
    possible = result.get("possible_points")
    label = standard_label(checker_name)
    if passed:
        return f"{label}: passed ({scored}/{possible})"
    failing = [
        (len(entry.get("msgs", [])), entry.get("name", "").rstrip(".").removeprefix("Check for ").removeprefix("Check that "))
        for entry in result.get("all_priorities", [])
        if entry.get("msgs")
    ]
    failing.sort(key=lambda item: -item[0])
    n_issues = sum(count for count, _ in failing)
    worst = ", ".join(f"{name} ({count})" for count, name in failing[:top])
    if len(failing) > top:
        worst += f", +{len(failing) - top} more"
    return (
        f"{label}: failed ({scored}/{possible}) — {n_issues} issue(s) in {len(failing)} check(s): {worst}"
    )


@register_step
class FormatCheck(BaseStep):
    """
    Run the IOOS file-format compliance checker and report the result.

    Does not run on the in-memory dataset; it re-reads the file from disk.
    A short pass/fail summary is always logged. With ``diagnostics`` on the
    full report is printed as well; JSON and/or RST report files are written
    when requested via ``output_type`` and an ``out_directory`` is set.

    Parameters
    ----------
    src : path or str, optional
        File to check. If omitted, falls back to the file loaded by a preceding
        ``Load OG1`` step.
    standards : list of str
        Standards to check, e.g. ``['cf', 'og']`` (``og`` = OG1).
    output_type : str or list of str, optional
        Report file(s) to save: ``'json'``, ``'rst'``, or a list of both. Omit
        (default) to save nothing. Requires ``out_directory`` in the config.
    proceed_on_fail : bool
        If False, halt the pipeline when the file fails the checks.
    """
    step_name = "Format Checker"

    parameter_schema = {
        "src": {
            "type": str,
            "default": None,
            "description": "File to check. If omitted, falls back to the file loaded by a preceding 'Load OG1' step.",
        },
        "standards": {
            "type": list,
            "default": ["cf", "og"],
            "description": "Standards to check, e.g. ['cf', 'og'].",
        },
        "output_type": {
            "type": [str, list],
            "default": None,
            "options": ["json", "rst"],
            "description": "Report file(s) to save: 'json', 'rst' or both. Omit to save nothing. Requires out_directory.",
        },
        "proceed_on_fail": {
            "type": bool,
            "default": True,
            "description": "If False, halt the pipeline when the file fails the checks.",
        },
    }

    def _save_formats(self):
        raw = self.parameters.get("output_type")
        if not raw:
            return []
        values = [raw] if isinstance(raw, str) else list(raw)
        return [v.lower() for v in values if isinstance(v, str) and v.lower() in ("json", "rst")]

    def run(self):
        check_suite = CheckSuite()
        check_suite.load_all_available_checkers()

        #   Fall back to the file loaded by a preceding Load OG1 step when no src is given.
        src = self.parameters.get("src") or self.context.get("global_parameters", {}).get("source_file")
        if not src:
            self.halt(
                "No file to check. Provide a 'src' path in the config, "
                "or place this step after a 'Load OG1' step so it can reuse that file."
            )

        cnames = self.parameters.get("standards")

        #   Each requested standard is served by a compliance-checker plugin; a name with
        #   no installed plugin would otherwise surface as an opaque library traceback.
        available = {name.split(":")[0] for name in check_suite.checkers}
        missing = [c for c in cnames if c not in available]
        if missing:
            self.halt(
                f"Compliance standard(s) {missing} are not installed. "
                f"Available: {', '.join(sorted(available)) or '(none)'}. "
                f"Install the matching plugin (e.g. 'pip install cc-plugin-og' for 'og')."
            )

        save_formats = self._save_formats()
        out_dir = self.context.get("global_parameters", {}).get("out_directory")
        if save_formats and not out_dir:
            self.log_warn(
                "No 'out_directory' set in the pipeline config — cannot save report file(s). "
                "Add 'out_directory', or remove 'json'/'rst' from output_type."
            )
            save_formats = []

        #   If run after loading data, the filename stem is saved in the global pipeline params.
        fname = self.context.get("global_parameters", {}).get("filename_core") or Path(src.strip("*.nc")).stem

        #   Run every requested checker once; reuse the results for the summary + files.
        ds = check_suite.load_dataset(src)
        score_groups = check_suite.run_all(ds, cnames)
        score_dict = {src: score_groups}

        overall_pass = True
        summary_lines = []
        cc_results = {}
        for checker_name, (groups, _errors) in score_groups.items():
            passed = check_suite.passtree(groups, _LENIENT_LIMIT)
            overall_pass = overall_pass and passed
            result = check_suite.dict_output(checker_name, groups, src, _LENIENT_LIMIT)
            summary_lines.append(console_summary(checker_name, result, passed))
            cc_results[checker_name] = result

        #   Structured results let the data report render a Format Checker section
        #   regardless of whether a report file was saved.
        self.context["cc_results"] = cc_results

        saved = self._write_reports(check_suite, score_dict, out_dir, fname, save_formats)

        labels = join_labels([standard_label(c) for c in cnames])
        header = f"'{fname}' {'passed' if overall_pass else 'failed'} {labels} format checks."
        (self.log if overall_pass else self.log_warn)(header)
        for line in summary_lines:
            self.log(line)
        if ComplianceChecker.check_errors(score_groups, verbose=0):
            self.log_warn("Errors occurred while running the checker — see the full report.")
        if saved:
            self.log("  ".join(f"{fmt.upper()} report saved to: {path}" for fmt, path in saved.items()))

        if self.diagnostics:
            self._check_suite, self._score_dict = check_suite, score_dict
            self.generate_diagnostics()

        if not overall_pass and self.parameters.get("proceed_on_fail") == False:
            self.halt(
                f"'{fname}' failed the format compliance checks and 'proceed_on_fail' is False."
            )

        return self.context

    def generate_diagnostics(self):
        # Full checker report (the same text as the RST file) to stdout, so the
        # console shows it and the dashboard captures it as the step's review text.
        self.log_generating_diagnostics()
        ComplianceChecker.stdout_output(self._check_suite, self._score_dict, 1, _LENIENT_LIMIT)

    def _write_reports(self, check_suite, score_dict, out_dir, fname, save_formats):
        # Returns {format: path}; JSON is preferred as the data report's cc_file.
        if not save_formats:
            return {}

        base = out_dir + fname + "_check"
        saved = {}

        if "json" in save_formats:
            json_path = base + ".json"
            ComplianceChecker.json_output(
                check_suite, score_dict, json_path, list(score_dict), _LENIENT_LIMIT
            )
            saved["json"] = json_path

        if "rst" in save_formats:
            rst_path = base + ".rst"
            with open(rst_path, "w", encoding="utf-8") as f:
                with stdout_redirector(f):
                    ComplianceChecker.stdout_output(
                        check_suite, score_dict, 1, _LENIENT_LIMIT
                    )
            saved["rst"] = rst_path

        #   Prefer JSON for the data report (structured); fall back to RST.
        self.context["global_parameters"]["cc_file"] = saved.get("json") or saved.get("rst")
        return saved
