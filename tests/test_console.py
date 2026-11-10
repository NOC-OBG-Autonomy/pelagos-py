import logging

from pelagos_py.utils.console import LogFormatter
from pelagos_py.utils.log_levels import SEVERE, STOP


def _format(level, message, color=False, name="pelagos_py.pipeline", **extra):
    record = logging.LogRecord(name, level, __file__, 1, message, None, None)
    record.__dict__.update(extra)
    return LogFormatter(color=color).format(record)


def test_info_line_has_no_tag():
    line = _format(
        logging.INFO, "loaded", color=True, name="pelagos_py.pipeline.step.Load Data"
    )
    assert line.endswith("  Load Data             loaded")
    assert "\033[" not in line


def test_warning_colours_only_the_tag():
    line = _format(
        logging.WARNING,
        "odd value",
        color=True,
        name="pelagos_py.pipeline.step.Load Data",
    )
    assert line.endswith("Load Data             \033[1;33mWARN\033[0m  odd value")


def test_plain_mode_keeps_tag_without_escapes():
    for level, tag in [
        (logging.WARNING, "WARN"),
        (SEVERE, "SEVERE"),
        (logging.ERROR, "ERROR"),
        (STOP, "STOP"),
    ]:
        line = _format(level, "something")
        assert line.endswith(f"pipeline              {tag}  something")


def test_qc_logger_name_is_shown_with_spaces():
    line = _format(logging.INFO, "flagged", name="pelagos_py.pipeline.qc.range_qc")
    assert line.endswith("  range qc              flagged")


def test_long_name_is_cut_to_the_column():
    line = _format(
        logging.INFO,
        "done",
        name="pelagos_py.pipeline.step.A Very Long Step Name Indeed",
    )
    assert line.endswith("  A Very Long Step Na…  done")


def test_raw_record_is_verbatim():
    assert _format(logging.WARNING, "as is", color=True, raw=True) == "as is"
