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

STOP = logging.ERROR + 5
"""Custom log level for a deliberate pipeline halt (e.g. invalid config).

Sits just above ERROR so it always surfaces, and is rendered as "STOP" in logs.
"""
logging.addLevelName(STOP, "STOP")

SEVERE = logging.WARNING + 5
"""Custom log level for a step that failed but was skipped (on_step_fail).

Sits between WARNING and ERROR, and is rendered amber (vs. WARNING's yellow) to
stand out from ordinary warnings.
"""
logging.addLevelName(SEVERE, "SEVERE")


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

