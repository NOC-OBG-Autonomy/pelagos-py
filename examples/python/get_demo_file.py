"""Fetch the demo dataset(s) used by the other example scripts.

Run this once before the other demos to download an OG1 NetCDF file into
examples/data/OG1. Same set of demo gliders as the dashboard's picker (see
pelagos_py.utils.demo_data), one entry per glider/mode: "_nrt" for the
near-real-time file, "_delayed" for the recovered/full one -- only the modes
actually hosted for that deployment are listed. Most are hosted on the BODC
deployment catalogue (see https://noc.ac.uk/projects/bio-carbon for context);
voto_og_dm (SEA063) is hosted on VOTO's erddap instead. churchill_delayed and
voto_og_dm are both cut down to a demo-sized TIME window after download (the
full download is then deleted) -- everything else is used as-is.

Usage:
  python get_demo_file.py                 # all demos (default)
  python get_demo_file.py nelson_nrt      # a single demo
  python get_demo_file.py nelson_nrt alr4_delayed   # several
  python get_demo_file.py all             # everything
"""

import sys
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from tqdm import tqdm

_OG1_NRT = "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001"
_OG1_DELAYED = "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery"
_GLIDER_DATA = "https://linkedsystems.uk/erddap/files/Public_Glider_Data_0711"

# name -> (download URL, output filename, TIME window to keep or None for whole file)
DEMOS = {
    # =========================== BODC ===========================
    # --- Bio-Carbon ---
    "nelson_nrt": (f"{_OG1_NRT}/Nelson_20240528/Nelson_646_R.nc", "Nelson_646_R.nc", None),
    "nelson_delayed": (f"{_OG1_DELAYED}/Nelson_20240528/Nelson_646.nc", "Nelson_646.nc", None),
    "doombar_nrt": (f"{_OG1_NRT}/Doombar_20240528/Doombar_648_R.nc", "Doombar_648_R.nc", None),
    "doombar_delayed": (f"{_OG1_DELAYED}/Doombar_20240528/Doombar_648.nc", "Doombar_648.nc", None),
    "churchill_nrt": (f"{_OG1_NRT}/Churchill_20240528/Churchill_647_R.nc", "Churchill_647_R.nc", None),
    "churchill_delayed": (
        f"{_OG1_DELAYED}/Churchill_20240528/Churchill_647.nc", "Churchill_647.nc",
        ("2024-08-01", "2024-09-01"),
    ),
    "alr4_nrt": (f"{_OG1_NRT}/ALR_4_20240609/ALR_4_649_R.nc", "ALR_4_649_R.nc", None),
    "alr4_delayed": (f"{_OG1_DELAYED}/ALR_4_20240609/ALR_4_649.nc", "ALR_4_649.nc", None),
    "alr6_nrt": (f"{_OG1_NRT}/ALR_6_20240611/ALR_6_650_R.nc", "ALR_6_650_R.nc", None),
    "alr6_delayed": (f"{_OG1_DELAYED}/ALR_6_20240611/ALR_6_650.nc", "ALR_6_650.nc", None),
    "cabot_nrt": (f"{_OG1_NRT}/Cabot_20240528/Cabot_645_R.nc", "Cabot_645_R.nc", None),
    "cabot_delayed": (f"{_OG1_DELAYED}/Cabot_20240528/Cabot_645.nc", "Cabot_645.nc", None),
    # --- Custard 1 ---
    "custard1_churchill_nrt": (f"{_OG1_NRT}/Churchill_20181204/Churchill_501_R.nc", "Churchill_501_R.nc", None),
    "custard1_churchill_delayed": (f"{_OG1_DELAYED}/Churchill_20181204/Churchill_501.nc", "Churchill_501.nc", None),
    "pancake_nrt": (f"{_OG1_NRT}/Pancake_20181209/Pancake_502_R.nc", "Pancake_502_R.nc", None),  # no delayed-mode file hosted
    "custard1_doombar_nrt": (  # only hosted on the raw glider-data store, not the OG1 one
        f"{_GLIDER_DATA}/Doombar_20181204/Doombar_503_R.nc", "Doombar_503_R.nc", None,
    ),
    # --- Custard 2 ---
    "bellamite_nrt": (f"{_OG1_NRT}/Bellamite_20191206/Bellamite_538_R.nc", "Bellamite_538_R.nc", None),
    "bellamite_delayed": (f"{_OG1_DELAYED}/Bellamite_20191206/Bellamite_538.nc", "Bellamite_538.nc", None),
    "custard2_zephyr_delayed": (f"{_OG1_DELAYED}/Zephyr_20191206/Zephyr_539.nc", "Zephyr_539.nc", None),  # no NRT file hosted
    # --- ReBELS ---
    "rebels_zephyr_nrt": (f"{_OG1_NRT}/Zephyr_20250323/Zephyr_675_R.nc", "Zephyr_675_R.nc", None),
    "rebels_zephyr_delayed": (f"{_OG1_DELAYED}/Zephyr_20250323/Zephyr_675.nc", "Zephyr_675.nc", None),
    "omg1_nrt": (f"{_OG1_NRT}/OMG-1_20250324/OMG-1_676_R.nc", "OMG-1_676_R.nc", None),  # no delayed-mode file hosted
    "9ja_nrt": (f"{_OG1_NRT}/9JA_20250812/9JA_699_R.nc", "9JA_699_R.nc", None),
    "9ja_delayed": (f"{_OG1_DELAYED}/9JA_20250812/9JA_699.nc", "9JA_699.nc", None),
    "growler_nrt": (f"{_OG1_NRT}/Growler_20250323/Growler_677_R.nc", "Growler_677_R.nc", None),
    "growler_delayed": (f"{_OG1_DELAYED}/Growler_20250323/Growler_677.nc", "Growler_677.nc", None),
    "rebels_stella_nrt": (f"{_OG1_NRT}/Stella_20250323/Stella_678_R.nc", "Stella_678_R.nc", None),
    "rebels_stella_delayed": (f"{_OG1_DELAYED}/Stella_20250323/Stella_678.nc", "Stella_678.nc", None),
    # --- ReBELS 2 ---
    "stella2026_nrt": (f"{_OG1_NRT}/Stella_20260403/Stella_713_R.nc", "Stella_713_R.nc", None),  # no delayed-mode file hosted (yet)

    # =========================== VOTO ===========================
    "voto_og_dm": (
        "https://erddap.observations.voiceoftheocean.org/erddap/files/"
        "OG_complete_SEA063_M75/SEA063_20240724T0737_delayed.nc",
        "SEA063_20240724T0737_delayed.nc",
        ("2024-07-25", "2024-08-03"),
    ),
}

# Work from the repo root so the relative paths below resolve the same way no
# matter where the script was started from.
_config = "examples/configs/example_config_nelson.yaml"
if not Path(_config).exists() and Path("../..", _config).exists():
    import os

    os.chdir("../..")

INPUT_DIR = Path("examples/data/OG1")


def _download(url: str, dest: Path) -> bool:
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"  download failed (HTTP {response.status_code})")
        return False
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}"
    ) as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    return True


def _cut_to_window(path: Path, start: str, end: str) -> None:
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
    tmp = path.with_suffix(".full.nc")
    path.rename(tmp)
    subset.to_netcdf(path, encoding=encoding)
    tmp.unlink()  # drop the full download, keep only the window
    print(f"  done ({idx.size} measurements kept)")


def fetch(name: str) -> None:
    url, filename, window = DEMOS[name]
    dest = INPUT_DIR / filename
    if dest.exists():
        print(f"{name}: already present at {dest.resolve()}")
        return
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _download(url, dest):
        return
    if window is not None:
        _cut_to_window(dest, *window)
    print(f"{name}: written to {dest.resolve()}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    names = list(DEMOS) if "all" in args else args
    for name in names:
        if name not in DEMOS:
            print(f"Unknown demo '{name}'. Choose from: {', '.join(DEMOS)}, all")
            continue
        fetch(name)
