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

"""Demo OG1 datasets: the registry behind the dashboard's picker, and :func:`get_demo_file`.

``DEMOS`` keys are the lower-case file names, e.g. ``churchill_647`` (delayed mode) and
``churchill_647_r`` (near real time); ``MISSIONS`` groups them by campaign in picker order.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from tqdm import tqdm

_MODE_LABELS = {"nrt": "NRT", "delayed": "Full"}


@dataclass(frozen=True)
class DemoEntry:
    url: str
    filename: str
    window: tuple[str, str] | None
    label: str  # glider display name, shared by its nrt/delayed variants
    mode: str  # "nrt" or "delayed"

    @property
    def display_label(self) -> str:  # e.g. "Nelson (NRT)"
        return f"{self.label} ({_MODE_LABELS[self.mode]})"


_OG1_NRT = "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001"
_OG1_DELAYED = "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery"
_GLIDER_DATA = "https://linkedsystems.uk/erddap/files/Public_Glider_Data_0711"


def _deployment(folder, label, nrt=None, delayed=None, window=None):
    # The NRT and delayed-mode (recovered) files of one deployment, from the BODC OG1 store.
    entries = []
    if nrt:
        entries.append(DemoEntry(f"{_OG1_NRT}/{folder}/{nrt}", nrt, None, label, "nrt"))
    if delayed:
        entries.append(DemoEntry(f"{_OG1_DELAYED}/{folder}/{delayed}", delayed, window, label, "delayed"))
    return entries


# Picker groups, in order.
_MISSION_ENTRIES = {
    "Bio-Carbon": [
        *_deployment("Nelson_20240528", "Nelson", nrt="Nelson_646_R.nc", delayed="Nelson_646.nc"),
        *_deployment("Doombar_20240528", "Doombar", nrt="Doombar_648_R.nc", delayed="Doombar_648.nc"),
        *_deployment("Churchill_20240528", "Churchill", nrt="Churchill_647_R.nc",
                     delayed="Churchill_647.nc", window=("2024-08-01", "2024-09-01")),
        *_deployment("ALR_4_20240609", "ALR 4", nrt="ALR_4_649_R.nc", delayed="ALR_4_649.nc"),
        *_deployment("ALR_6_20240611", "ALR 6", nrt="ALR_6_650_R.nc", delayed="ALR_6_650.nc"),
        *_deployment("Cabot_20240528", "Cabot", nrt="Cabot_645_R.nc", delayed="Cabot_645.nc"),
    ],
    "Custard 1": [
        *_deployment("Churchill_20181204", "Churchill", nrt="Churchill_501_R.nc", delayed="Churchill_501.nc"),
        *_deployment("Pancake_20181209", "Pancake", nrt="Pancake_502_R.nc"),
        # Only hosted on the raw glider-data store, not the OG1 one.
        DemoEntry(f"{_GLIDER_DATA}/Doombar_20181204/Doombar_503_R.nc", "Doombar_503_R.nc",
                  None, "Doombar", "nrt"),
    ],
    "Custard 2": [
        *_deployment("Bellamite_20191206", "Bellamite", nrt="Bellamite_538_R.nc", delayed="Bellamite_538.nc"),
        *_deployment("Zephyr_20191206", "Zephyr", delayed="Zephyr_539.nc"),
    ],
    "ReBELS": [
        *_deployment("Zephyr_20250323", "Zephyr", nrt="Zephyr_675_R.nc", delayed="Zephyr_675.nc"),
        *_deployment("OMG-1_20250324", "OMG-1", nrt="OMG-1_676_R.nc"),
        *_deployment("9JA_20250812", "9JA", nrt="9JA_699_R.nc", delayed="9JA_699.nc"),
        *_deployment("Growler_20250323", "Growler", nrt="Growler_677_R.nc", delayed="Growler_677.nc"),
        *_deployment("Stella_20250323", "Stella", nrt="Stella_678_R.nc", delayed="Stella_678.nc"),
    ],
    "ReBELS 2": [
        *_deployment("Stella_20260403", "Stella", nrt="Stella_713_R.nc"),
    ],
    "VOTO": [
        # Hosted on VOTO's own erddap, not BODC.
        DemoEntry("https://erddap.observations.voiceoftheocean.org/erddap/files/"
                  "OG_complete_SEA063_M75/SEA063_20240724T0737_delayed.nc",
                  "SEA063_20240724T0737_delayed.nc", ("2024-07-25", "2024-08-03"), "SEA063", "delayed"),
    ],
}


def _key(entry):
    # The file name, e.g. "Churchill_647_R.nc" -> "churchill_647_r".
    return Path(entry.filename).stem.lower()


DEMOS = {_key(e): e for entries in _MISSION_ENTRIES.values() for e in entries}
MISSIONS = {mission: [_key(e) for e in entries] for mission, entries in _MISSION_ENTRIES.items()}

# Where the dashboard keeps configs and demo data; relative paths in its configs resolve from here.
WORKSPACE_DIR = Path.home() / "Documents" / "pelagos-py"
DEMO_DATA_DIR = WORKSPACE_DIR / "demo_data"


def get_demo_file(name=None, on_progress=None):
    """
    Download a demo OG1 file into ``~/Documents/pelagos-py/demo_data`` (only the first time) and return its path.

    Long deployments are cut down to a shorter time window after downloading. Call it with
    no name to list the demos.

    .. code-block:: python

        from pelagos_py import Pipeline, get_demo_file

        Pipeline.make_config(get_demo_file("nelson_646_r")).run()
    """
    if name is None:
        for mission, keys in MISSIONS.items():
            print(f"{mission}: {', '.join(keys)}")
        return None
    if name not in DEMOS:
        raise ValueError(f"Unknown demo '{name}'. Call get_demo_file() to list them.")
    entry = DEMOS[name]
    dest = DEMO_DATA_DIR / entry.filename
    if dest.exists():
        return dest
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _download(entry.url, dest, on_progress)
    if entry.window is not None:
        _cut_to_window(dest, *entry.window)
    return dest


def _download(url, dest, on_progress=None):
    # the read timeout is per chunk, so a dropped connection fails after 30 s instead of hanging
    response = requests.get(url, stream=True, timeout=(15, 30))
    response.raise_for_status()
    total = int(response.headers.get("Content-Length") or 0)
    tmp = dest.with_name(dest.name + ".part")
    done = 0
    try:
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)  # left behind only if the download failed


def _cut_to_window(path, start, end):
    # Keep measurements in [start, end) whose TIME is valid and strictly increasing
    # (the pipeline requires monotonic, non-NaT time). NaT excludes itself since its
    # comparisons are False; the running-max test drops any out-of-order sample.
    with xr.open_dataset(path) as ds:
        t = ds["TIME"].values
        idx = np.flatnonzero((t >= np.datetime64(start)) & (t < np.datetime64(end)))
        if idx.size:
            tt = t[idx]
            increasing = np.concatenate(([True], tt[1:] > np.maximum.accumulate(tt)[:-1]))
            idx = idx[increasing]
        n_total = ds.sizes["N_MEASUREMENTS"]
        if idx.size == 0:
            print(f"  no measurements found in {start}..{end}; leaving file unchanged")
            return
        print(f"  cutting to {start}..{end}: {idx.size} of {n_total} measurements...")
        # Read the contiguous slice covering the window, then pick out the kept rows
        # in memory: a fancy-index isel straight off the compressed file stalls for
        # millions of points.
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        subset = ds.isel(N_MEASUREMENTS=slice(lo, hi)).load().isel(N_MEASUREMENTS=idx - lo)
    # Drop the source's chunk encoding (its chunksizes were sized for the full
    # dimension and stall the subset write); re-apply zlib to keep the file small.
    encoding = {}
    for v in subset.variables:
        subset[v].encoding = {}
        if subset[v].dtype.kind in "fiu":
            encoding[v] = {"zlib": True, "complevel": 2}
    # Swap the trimmed file in only once it's fully written, so a failed write never leaves a partial file.
    tmp = path.with_suffix(".cut.nc")
    try:
        subset.to_netcdf(tmp, encoding=encoding)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
