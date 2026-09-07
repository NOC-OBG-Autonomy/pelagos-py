"""Fetch the demo dataset(s) used by the other example scripts.

Run this once before the other demos to download an OG1 NetCDF file into
examples/data/OG1. nelson/churchill/alr are hosted on the BODC deployment
catalogue (see https://noc.ac.uk/projects/bio-carbon for context);
voto_og_dm is hosted on VOTO's erddap instead.

Four demos are available:
  nelson      - Nelson (unit_397), Near-Real-Time deployment. Used as-is.
  churchill   - Churchill (unit_398), Recovered deployment. The full record
                spans May-Oct 2024; it is cut down to August 2024 to keep a
                demo-sized file, then the full download is deleted.
  alr         - ALR_4 (unit_399), Recovered deployment. Used as-is.
  voto_og_dm  - SEA063 (VOTO), cut down to 2024-07-25..2024-08-03.

Usage:
  python get_demo_file.py                 # all demos (default)
  python get_demo_file.py nelson_nrt      # a single demo
  python get_demo_file.py nelson_nrt alr4_delayed   # several
  python get_demo_file.py all             # everything
"""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

_OG1_NRT = "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001"
_OG1_DELAYED = "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery"
_GLIDER_DATA = "https://linkedsystems.uk/erddap/files/Public_Glider_Data_0711"

# name -> (download URL, output filename)
DEMOS = {
    # --- Bio-Carbon ---
    "nelson_nrt": (f"{_OG1_NRT}/Nelson_20240528/Nelson_646_R.nc", "Nelson_646_R.nc"),
    "nelson_delayed": (f"{_OG1_DELAYED}/Nelson_20240528/Nelson_646.nc", "Nelson_646.nc"),
    "doombar_nrt": (f"{_OG1_NRT}/Doombar_20240528/Doombar_648_R.nc", "Doombar_648_R.nc"),
    "doombar_delayed": (f"{_OG1_DELAYED}/Doombar_20240528/Doombar_648.nc", "Doombar_648.nc"),
    "churchill_nrt": (f"{_OG1_NRT}/Churchill_20240528/Churchill_647_R.nc", "Churchill_647_R.nc"),
    "churchill_delayed": (f"{_OG1_DELAYED}/Churchill_20240528/Churchill_647.nc", "Churchill_647.nc"),
    "alr4_nrt": (f"{_OG1_NRT}/ALR_4_20240609/ALR_4_649_R.nc", "ALR_4_649_R.nc"),
    "alr4_delayed": (f"{_OG1_DELAYED}/ALR_4_20240609/ALR_4_649.nc", "ALR_4_649.nc"),
    "alr6_nrt": (f"{_OG1_NRT}/ALR_6_20240611/ALR_6_650_R.nc", "ALR_6_650_R.nc"),
    "alr6_delayed": (f"{_OG1_DELAYED}/ALR_6_20240611/ALR_6_650.nc", "ALR_6_650.nc"),
    "cabot_nrt": (f"{_OG1_NRT}/Cabot_20240528/Cabot_645_R.nc", "Cabot_645_R.nc"),
    "cabot_delayed": (f"{_OG1_DELAYED}/Cabot_20240528/Cabot_645.nc", "Cabot_645.nc"),
    # --- Custard 1 ---
    "custard1_churchill_nrt": (f"{_OG1_NRT}/Churchill_20181204/Churchill_501_R.nc", "Churchill_501_R.nc"),
    "custard1_churchill_delayed": (f"{_OG1_DELAYED}/Churchill_20181204/Churchill_501.nc", "Churchill_501.nc"),
    "pancake_nrt": (f"{_OG1_NRT}/Pancake_20181209/Pancake_502_R.nc", "Pancake_502_R.nc"),  # no delayed-mode file hosted
    "custard1_doombar_nrt": (  # only hosted on the raw glider-data store, not the OG1 one
        f"{_GLIDER_DATA}/Doombar_20181204/Doombar_503_R.nc", "Doombar_503_R.nc",
    ),
    # --- Custard 2 ---
    "bellamite_nrt": (f"{_OG1_NRT}/Bellamite_20191206/Bellamite_538_R.nc", "Bellamite_538_R.nc"),
    "bellamite_delayed": (f"{_OG1_DELAYED}/Bellamite_20191206/Bellamite_538.nc", "Bellamite_538.nc"),
    "custard2_zephyr_delayed": (f"{_OG1_DELAYED}/Zephyr_20191206/Zephyr_539.nc", "Zephyr_539.nc"),  # no NRT file hosted
    # --- ReBELS ---
    "rebels_zephyr_nrt": (f"{_OG1_NRT}/Zephyr_20250323/Zephyr_675_R.nc", "Zephyr_675_R.nc"),
    "rebels_zephyr_delayed": (f"{_OG1_DELAYED}/Zephyr_20250323/Zephyr_675.nc", "Zephyr_675.nc"),
    "omg1_nrt": (f"{_OG1_NRT}/OMG-1_20250324/OMG-1_676_R.nc", "OMG-1_676_R.nc"),  # no delayed-mode file hosted
    "9ja_nrt": (f"{_OG1_NRT}/9JA_20250812/9JA_699_R.nc", "9JA_699_R.nc"),
    "9ja_delayed": (f"{_OG1_DELAYED}/9JA_20250812/9JA_699.nc", "9JA_699.nc"),
    "growler_nrt": (f"{_OG1_NRT}/Growler_20250323/Growler_677_R.nc", "Growler_677_R.nc"),
    "growler_delayed": (f"{_OG1_DELAYED}/Growler_20250323/Growler_677.nc", "Growler_677.nc"),
    "rebels_stella_nrt": (f"{_OG1_NRT}/Stella_20250323/Stella_678_R.nc", "Stella_678_R.nc"),
    "rebels_stella_delayed": (f"{_OG1_DELAYED}/Stella_20250323/Stella_678.nc", "Stella_678.nc"),
    # --- ReBELS 2 ---
    "stella2026_nrt": (f"{_OG1_NRT}/Stella_20260403/Stella_713_R.nc", "Stella_713_R.nc"),  # no delayed-mode file hosted (yet)
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


def fetch(name: str) -> None:
    url, filename = DEMOS[name]
    dest = INPUT_DIR / filename
    if dest.exists():
        print(f"{name}: already present at {dest.resolve()}")
        return
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _download(url, dest):
        return
    print(f"{name}: written to {dest.resolve()}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    names = list(DEMOS) if "all" in args else args
    for name in names:
        if name not in DEMOS:
            print(f"Unknown demo '{name}'. Choose from: {', '.join(DEMOS)}, all")
            continue
        fetch(name)
