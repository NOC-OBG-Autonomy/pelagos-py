"""Tests the step 'Format Checker"""

#   Test module import
from toolbox.steps.custom import format_check

import pytest
from unittest.mock import (
    patch,
    MagicMock,
)

#   Other imports

FormatCheck = format_check.FormatCheck

def test_format_checker_run_success(tmp_path):
    # Mock both the CheckSuite and run_checker from the compliance checker runner.
    with patch("toolbox.steps.custom.format_check.CheckSuite") as mock_checksuite, \
         patch("toolbox.steps.custom.format_check.ComplianceChecker.run_checker") as mock_run_checker:
        mock_suite_instance = MagicMock()
        mock_run_checker.return_value = (True, False)
        mock_checksuite.return_value = mock_suite_instance

        src = "test_file.nc"
        standards = ["og"]

        step = FormatCheck(
            name="format_check",
            parameters={
                "src": src,
                "standards": standards,
            },
            context={
                "global_parameters": {
                    "out_directory": str(tmp_path) + "/",
                    "filename_core": "demo_test"
                }
            }
        )

        result = step.run()

        mock_suite_instance.load_all_available_checkers.assert_called_once()
        mock_run_checker.assert_called_once()
        assert result is not None


def test_format_checker_run_failure_raises(tmp_path):
    with patch("toolbox.steps.custom.format_check.CheckSuite") as mock_checksuite, \
         patch("toolbox.steps.custom.format_check.ComplianceChecker.run_checker") as mock_run_checker:
        mock_suite_instance = MagicMock()
        mock_run_checker.return_value = (False, False)
        mock_checksuite.return_value = mock_suite_instance

        step = FormatCheck(
            name="format_check",
            parameters={
                "src": "test_file.nc",
                "standards": ["og"],
                "proceed_on_fail": False,
            },
            context={
                "global_parameters": {
                    "out_directory": str(tmp_path) + "/",
                    "filename_core": "demo_test"
                }
            }
        )
        step.log_warn = MagicMock() # Mock these to prevent real warnings in pytest

        with pytest.raises(RuntimeError, match="Compliance check step failed"):
            step.run()
        
        step.log_warn.assert_called_once()

def test_format_checker_errors(tmp_path):
    with patch("toolbox.steps.custom.format_check.CheckSuite") as mock_checksuite, \
         patch("toolbox.steps.custom.format_check.ComplianceChecker.run_checker") as mock_run_checker:
        mock_suite_instance = MagicMock()
        mock_run_checker.return_value = (True, True)
        mock_checksuite.return_value = mock_suite_instance

        src = "test_file.nc"
        standards = ["og"]

        step = FormatCheck(
            name="format_check",
            parameters={
                "src": src,
                "standards": standards,
            },
            context={
                "global_parameters": {
                    "out_directory": str(tmp_path) + "/",
                    "filename_core": "demo_test"
                }
            }
        )
        step.log_warn = MagicMock()

        result = step.run()

        mock_suite_instance.load_all_available_checkers.assert_called_once()
        mock_run_checker.assert_called_once()
        assert result is not None

        step.log_warn.assert_called_once()