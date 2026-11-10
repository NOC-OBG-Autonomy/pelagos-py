import logging

from pelagos_py.utils.console import ConsoleFormatter
from pelagos_py.utils.log_levels import SEVERE, STOP


def _format(level, message, use_color, **extra):
    formatter = ConsoleFormatter()
    formatter._use_color = use_color
    record = logging.LogRecord(
        "pelagos_py.pipeline", level, __file__, 1, message, None, None
    )
    record.__dict__.update(extra)
    return formatter.format(record)


def test_info_line_has_no_tag():
    line = _format(logging.INFO, "[Load Data] loaded", use_color=True)
    assert line.endswith("Load Data             loaded")
    assert "\033[" not in line


def test_warning_colours_only_the_tag():
    line = _format(logging.WARNING, "[Load Data] odd value", use_color=True)
    assert line.endswith("Load Data             \033[1;33mWARN\033[0m  odd value")


def test_plain_mode_keeps_tag_without_escapes():
    for level, tag in [
        (logging.WARNING, "WARN"),
        (SEVERE, "SEVERE"),
        (logging.ERROR, "ERROR"),
        (STOP, "STOP"),
    ]:
        line = _format(level, "something", use_color=False)
        assert line.endswith(f"pipeline              {tag}  something")
        assert "\033[" not in line


def test_raw_record_is_verbatim():
    line = _format(logging.WARNING, "as is", use_color=True, raw=True)
    assert line == "as is"
