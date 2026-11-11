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
"""The pelagos_py config dashboard: ``from pelagos_py import dashboard; dashboard.run()``."""


def run(port=8791):
    """Open the config dashboard in your browser. Needs ``pip install "pelagos_py[dashboard]"``."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        raise ImportError(
            'The dashboard needs extra packages: pip install "pelagos_py[dashboard]"'
        ) from None
    from pelagos_py.dashboard.app import serve

    serve(port)
