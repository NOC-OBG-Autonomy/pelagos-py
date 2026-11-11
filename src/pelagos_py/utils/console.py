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

"""Console presentation for pipeline runs.

Every line is ``time  step-name  detail``, on the console and in the log file;
warnings and errors add a short level tag before the detail, and countable loops
show a live progress bar. The console handler writes through :func:`tqdm.write`
so log lines and a live bar coexist; :func:`progress_bar` is the uniform bar style.
"""

import logging
import sys
import time

from tqdm import tqdm

from pelagos_py.utils.log_levels import LEVEL_STYLES, _supports_color

# Width the step/process name is padded to, so the detail columns line up.
_NAME_WIDTH = 20

# Uniform tqdm bar for every countable loop; ``{desc}`` carries the time/name prefix.
BAR_FORMAT = (
    "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
)


def _console_enabled(stream=None):
    # False when output is redirected or colour is disabled (tqdm/colour off).
    return _supports_color(stream if stream is not None else sys.stderr)


def _display_name(record):
    # Short step/process name from the logger name.
    name = record.name
    for prefix in ("pelagos_py.pipeline.step.", "pelagos_py.pipeline.qc."):
        if name.startswith(prefix):
            return name[len(prefix) :].replace("_", " ")
    if name.startswith("pelagos_py.pipeline.discovery"):
        return "discovery"
    if name == "pelagos_py.pipeline":
        return "pipeline"
    return name.rsplit(".", 1)[-1]


def _name_column(name):
    if len(name) > _NAME_WIDTH:
        name = name[: _NAME_WIDTH - 1] + "…"
    return f"{name:<{_NAME_WIDTH}}"


def _bar_prefix(step_name):
    # The ``time  name`` prefix a progress bar shares with log lines.
    stamp = time.strftime("%H:%M:%S")
    if step_name:
        return f"{stamp}  {_name_column(step_name)}  "
    return f"{stamp}  "


class LogFormatter(logging.Formatter):
    """Format a record as ``time  name  message``, for the console and the log file.

    The name is the step or process the logger belongs to, padded or cut to one
    width so the messages line up. Warning-and-above lines get a tag (``WARN``,
    ``SEVERE``, ``ERROR``, ``STOP``) before the message, coloured if ``color``.
    Pass ``extra={"raw": True}`` to emit a line verbatim.
    """

    def __init__(self, datefmt="%H:%M:%S", color=False):
        super().__init__(datefmt=datefmt)
        self.color = color

    def prefix(self, record):
        stamp = self.formatTime(record, self.datefmt)
        return f"{stamp}  {_name_column(_display_name(record))}  "

    def format(self, record):
        message = record.getMessage()
        if getattr(record, "raw", False):
            return message

        prefix = self.prefix(record)
        style = LEVEL_STYLES.get(record.levelname)
        if style is None:
            return prefix + message
        level_tag = style.tag
        if self.color:
            level_tag = f"\033[1;{style.ansi}m{style.tag}\033[0m"
        return f"{prefix}{level_tag}  {message}"


class _TqdmHandler(logging.StreamHandler):
    """Stream handler that emits through :func:`tqdm.write` so records don't
    corrupt an active progress bar."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:  # pragma: no cover - matches logging.Handler contract
            self.handleError(record)


class _ConsoleOnlyFilter(logging.Filter):
    """Drop records tagged ``extra={"console": False}`` so noisy book-keeping
    reaches the log file but not the console."""

    def filter(self, record):
        return getattr(record, "console", True)


def make_console_handler(level=logging.INFO):
    """Build the pipeline's console handler: compact, coloured, tqdm-aware."""
    handler = _TqdmHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(LogFormatter(color=_console_enabled(sys.stderr)))
    handler.addFilter(_ConsoleOnlyFilter())
    return handler


class _PhaseBar(tqdm):
    """A countable loop shown as a bar, logging a one-line summary to the file
    (only) on close so the file and report still record the phase."""

    def __init__(
        self,
        *args,
        logger=None,
        summary_unit=None,
        summary_label=None,
        **kwargs,
    ):
        self._logger = logger
        self._summary_unit = summary_unit
        self._summary_label = summary_label
        self._summary_start = time.time()
        self._summarised = False
        # Counted here, not from tqdm's ``n``, so the summary is right on a
        # redirected run where the bar is disabled and never advances.
        self._count = 0
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for obj in super().__iter__():
            self._count += 1
            yield obj

    def update(self, n=1):
        self._count += n
        return super().update(n)

    def close(self):
        if not hasattr(
            self, "disable"
        ):  # interrupted inside tqdm's __init__ (Stop during startup)
            return
        super().close()
        if self._summarised:
            return
        self._summarised = True
        if self._logger is not None and self._count:
            self._logger.info(
                "%s: %s %s (%.1fs)",
                self._summary_label or "progress",
                self._count,
                self._summary_unit or self.unit,
                time.time() - self._summary_start,
                extra={"console": False},
            )


def progress_bar(
    iterable=None,
    *,
    desc="",
    total=None,
    unit="it",
    leave=False,
    logger=None,
    step_name=None,
):
    """A tqdm bar in the pipeline's standard style for every countable loop.

    ``step_name`` adds the ``time  name`` prefix; ``logger`` also writes a
    one-line summary to the file. Auto-disables off a terminal.
    """
    display = _bar_prefix(step_name) + desc
    return _PhaseBar(
        iterable,
        desc=display,
        total=total,
        unit=unit,
        leave=leave,
        bar_format=BAR_FORMAT,
        colour="white",
        disable=not _console_enabled(),
        logger=logger,
        summary_unit=unit,
        summary_label=desc,
    )
