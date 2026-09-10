"""Fetch the demo dataset(s) used by the other example scripts.

Downloads entries of pelagos_py.utils.demo_data.DEMOS (the dashboard's picker) into
examples/data/OG1; entries with a TIME window are cut down to it after download.

Usage:
  python get_demo_file.py                 # all demos (default)
  python get_demo_file.py nelson_nrt      # a single demo
  python get_demo_file.py nelson_nrt alr4_delayed   # several
  python get_demo_file.py all             # everything
"""

import os
import sys
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from tqdm import tqdm

from pelagos_py.utils.demo_data import DEMO_DATA_DIR, DEMOS

os.chdir(Path(__file__).resolve().parents[2])  # repo root, so relative paths match the configs
INPUT_DIR = Path(DEMO_DATA_DIR)


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
    entry = DEMOS[name]
    dest = INPUT_DIR / entry.filename
    if dest.exists():
        print(f"{name}: already present at {dest.resolve()}")
        return
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _download(entry.url, dest):
        return
    if entry.window is not None:
        _cut_to_window(dest, *entry.window)
    print(f"{name}: written to {dest.resolve()}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    names = list(DEMOS) if "all" in args else args
    for name in names:
        if name not in DEMOS:
            print(f"Unknown demo '{name}'. Choose from: {', '.join(DEMOS)}, all")
            continue
        fetch(name)
