"""Tests the step 'Format Checker'."""

#   Test module import
from pelagos_py.steps.input_output import format_check

import pytest
from unittest.mock import patch, MagicMock

FormatCheck = format_check.FormatCheck

#   A representative slice of the OG checker's dict_output, with the two named
#   "mandatory" checks the console summary highlights plus an extra failing check.
OG_PRIORITIES = [
    {
        "name": "Check for mandatory global attributes.",
        "value": [13, 17],
        "msgs": [
            "Global attribute contributing_institutions is missing",
            "Global attribute start_date is missing",
        ],
    },
    {
        "name": "Check for mandatory variables.",
        "value": [13, 14],
        "msgs": ["Variable DEPTH is missing"],
    },
    {
        "name": "Check that all attribute names are lowercase.",
        "value": [1, 4],
        "msgs": ["attr A should be lowercase", "attr B should be lowercase", "attr C"],
    },
]


def _mock_suite(passed=True, priorities=OG_PRIORITIES, checkers=("og", "og:1.0")):
    """Build a mocked CheckSuite instance covering the run() code path."""
    suite = MagicMock()
    suite.checkers = {name: object() for name in checkers}
    suite.load_dataset.return_value = MagicMock()
    #   run_all -> {checker: (groups, errors)}; empty errors keeps check_errors quiet.
    suite.run_all.return_value = {"og": (["group"], [])}
    suite.passtree.return_value = passed
    suite.dict_output.return_value = {
        "scored_points": 27,
        "possible_points": 35,
        "all_priorities": priorities,
    }
    return suite


def _step(parameters, global_parameters):
    return FormatCheck(
        name="Format Checker",
        parameters=parameters,
        context={"global_parameters": global_parameters},
    )


# --------------------------- console_summary helper ---------------------------


def test_console_summary_ranks_failing_checks():
    result = {"scored_points": 27, "possible_points": 35, "all_priorities": OG_PRIORITIES}
    line = format_check.console_summary("og", result, passed=False, top=2)
    assert line.startswith("OG1: failed (27/35) — 6 issue(s) in 3 check(s): ")
    assert "all attribute names are lowercase (3), mandatory global attributes (2), +1 more" in line


def test_console_summary_pass():
    result = {"scored_points": 35, "possible_points": 35, "all_priorities": []}
    assert format_check.console_summary("og", result, passed=True) == "OG1: passed (35/35)"


def test_header_names_standards():
    assert format_check.join_labels([format_check.standard_label(c) for c in ["cf", "og"]]) == "CF and OG1"


def test_console_only_run_succeeds():
    with patch("pelagos_py.steps.input_output.format_check.CheckSuite") as mock_cs:
        mock_cs.return_value = _mock_suite(passed=True)
        step = _step({"src": "test_file.nc", "standards": ["og"]}, {})
        result = step.run()

    assert result is not None
    # console-only: nothing registered for the data report
    assert "cc_file" not in result["global_parameters"]


def test_json_output_saved_and_registered(tmp_path):
    with (
        patch("pelagos_py.steps.input_output.format_check.CheckSuite") as mock_cs,
        patch(
            "pelagos_py.steps.input_output.format_check.ComplianceChecker.json_output"
        ) as mock_json,
    ):
        mock_cs.return_value = _mock_suite(passed=True)
        step = _step(
            {"src": "test_file.nc", "standards": ["og"], "output_type": "json"},
            {"out_directory": str(tmp_path) + "/", "filename_core": "demo_test"},
        )
        result = step.run()

    mock_json.assert_called_once()
    output_filename = mock_json.call_args[0][2]  # (cs, score_dict, output_filename, ...)
    assert output_filename.endswith("demo_test_check.json")
    assert result["global_parameters"]["cc_file"].endswith("demo_test_check.json")


def test_output_type_without_out_directory_warns_and_skips_file():
    with (
        patch("pelagos_py.steps.input_output.format_check.CheckSuite") as mock_cs,
        patch(
            "pelagos_py.steps.input_output.format_check.ComplianceChecker.json_output"
        ) as mock_json,
    ):
        mock_cs.return_value = _mock_suite(passed=True)
        step = _step(
            {"src": "test_file.nc", "standards": ["og"], "output_type": "json"}, {}
        )
        step.log_warn = MagicMock()
        result = step.run()

    mock_json.assert_not_called()
    step.log_warn.assert_called()  # warned about console-only
    assert "cc_file" not in result["global_parameters"]


def test_failure_with_proceed_on_fail_false_halts():
    with patch("pelagos_py.steps.input_output.format_check.CheckSuite") as mock_cs:
        mock_cs.return_value = _mock_suite(passed=False)
        step = _step(
            {"src": "test_file.nc", "standards": ["og"], "proceed_on_fail": False}, {}
        )
        with pytest.raises(SystemExit):
            step.run()


def test_failure_with_proceed_on_fail_true_continues():
    with patch("pelagos_py.steps.input_output.format_check.CheckSuite") as mock_cs:
        mock_cs.return_value = _mock_suite(passed=False)
        step = _step(
            {"src": "test_file.nc", "standards": ["og"], "proceed_on_fail": True}, {}
        )
        result = step.run()  # must not raise
    assert result is not None


def test_uninstalled_standard_halts():
    with patch("pelagos_py.steps.input_output.format_check.CheckSuite") as mock_cs:
        # 'og' requested but only 'cf' available
        mock_cs.return_value = _mock_suite(checkers=("cf",))
        step = _step({"src": "test_file.nc", "standards": ["og"]}, {})
        with pytest.raises(SystemExit):
            step.run()


def test_missing_src_halts():
    with patch("pelagos_py.steps.input_output.format_check.CheckSuite") as mock_cs:
        mock_cs.return_value = _mock_suite()
        step = _step({"standards": ["og"]}, {})  # no src, no source_file
        with pytest.raises(SystemExit):
            step.run()


def test_src_falls_back_to_loaded_file():
    with patch("pelagos_py.steps.input_output.format_check.CheckSuite") as mock_cs:
        suite = _mock_suite(passed=True)
        mock_cs.return_value = suite
        step = _step(
            {"standards": ["og"]},  # no explicit src
            {"source_file": "/data/loaded.nc"},
        )
        step.run()
    # the loaded file is what gets checked
    suite.load_dataset.assert_called_once_with("/data/loaded.nc")
