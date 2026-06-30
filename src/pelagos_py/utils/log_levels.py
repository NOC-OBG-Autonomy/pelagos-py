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

STOP = logging.ERROR + 5
"""Custom log level for a deliberate pipeline halt (e.g. invalid config).

Sits just above ERROR so it always surfaces, and is rendered as "STOP" in logs.
"""
logging.addLevelName(STOP, "STOP")
