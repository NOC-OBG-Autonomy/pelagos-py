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

"""pelagos_py base package."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

try:
    # Resolved from the installed distribution metadata, which setuptools-scm
    # derives from the latest git tag at build/install time.
    __version__ = _get_version("pelagos_py")
except PackageNotFoundError:
    # Running from a source checkout that was never installed/built.
    __version__ = "0.0.0+unknown"

__credits__ = "National Oceanography Centre"
