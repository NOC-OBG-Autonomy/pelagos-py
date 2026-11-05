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

"""The demo OG1 file: :func:`get_demo_file` downloads it the first time it's used."""

from pathlib import Path

import requests
from tqdm import tqdm

DEMO_URL = (
    "https://github.com/NOC-OBG-Autonomy/pelagos-py/releases/download/"
    "demo-data-v2/demo_OG1.nc"
)
WORKSPACE_DIR = Path.home() / "Documents" / "pelagos-py"
_REPO_EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
# A source checkout keeps the demo with the examples; a pip install has no examples folder.
if _REPO_EXAMPLES.is_dir():
    DEMO_DATA_DIR = _REPO_EXAMPLES / "data" / "OG1"
else:
    DEMO_DATA_DIR = WORKSPACE_DIR / "demo_data"
DEMO_FILE = DEMO_DATA_DIR / "demo_OG1.nc"


def get_demo_file(on_progress=None):
    """
    Download the demo OG1 file into ``examples/data/OG1`` (``~/Documents/pelagos-py/demo_data``
    for a pip install), only the first time, and return its path.

    The demo is one month of NOC glider data from the North Atlantic in OG1 format,
    thinned to one reading per sensor every 2 m of depth.

    .. code-block:: python

        from pelagos_py import Pipeline, get_demo_file

        Pipeline.make_config(get_demo_file()).run()
    """
    if not DEMO_FILE.exists():
        DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _download(DEMO_URL, DEMO_FILE, on_progress)
    return DEMO_FILE


def _download(url, dest, on_progress=None):
    # the read timeout is per chunk, so a dropped connection fails after 30 s instead of hanging
    response = requests.get(url, stream=True, timeout=(15, 30))
    response.raise_for_status()
    total = int(response.headers.get("Content-Length") or 0)
    tmp = dest.with_name(dest.name + ".part")
    done = 0
    try:
        with (
            open(tmp, "wb") as f,
            tqdm(
                total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}"
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)  # left behind only if the download failed
