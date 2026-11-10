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

"""Custom logging levels shared across the pipeline."""

import logging
import os
from typing import NamedTuple

STOP = logging.ERROR + 5
"""Custom log level for a deliberate pipeline halt (e.g. invalid config).

Sits just above ERROR so it always surfaces, and is rendered as "STOP" in logs.
"""
logging.addLevelName(STOP, "STOP")

SEVERE = logging.WARNING + 5
"""Custom log level for a step that failed but was skipped (on_step_fail).

Sits between WARNING and ERROR, and its tag is amber (vs. WARNING's yellow) to
stand out from ordinary warnings.
"""
logging.addLevelName(SEVERE, "SEVERE")


class LevelStyle(NamedTuple):
    tag: str
    ansi: str  # SGR colour code for the terminal
    rgb: tuple  # colour for the PDF report's log panel


# Warning-and-above levels get a short coloured tag; INFO/DEBUG get none.
LEVEL_STYLES = {
    "WARNING": LevelStyle("WARN", "33", (240, 200, 110)),
    "SEVERE": LevelStyle("SEVERE", "38;5;202", (245, 150, 80)),
    "ERROR": LevelStyle("ERROR", "31", (240, 130, 130)),
    "STOP": LevelStyle("STOP", "31", (240, 130, 130)),
    "CRITICAL": LevelStyle("CRITICAL", "31", (240, 130, 130)),
}


def _supports_color(stream):
    """Whether ANSI colour is safe to emit on ``stream``.

    Only colour a real interactive terminal: skip when output is redirected to
    a file/pipe (``isatty()`` is False), when NO_COLOR is set (no-color.org),
    or on a dumb terminal. FORCE_COLOR overrides all of these.
    """
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()
