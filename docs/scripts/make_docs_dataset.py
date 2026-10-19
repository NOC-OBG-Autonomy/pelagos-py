"""Cut the small OG1 file the user-guide figures are rendered from.

    python docs/scripts/make_docs_dataset.py SOURCE.nc [--start ... --end ...] [--out ...]

Keeps only the measurement variables the docs pipeline reads, over a short window.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

KEEP = [
    "TIME",
    "LATITUDE",
    "LONGITUDE",
    "PRES",
    "TEMP",
    "CNDC",
    "CHLA",
    "BETA_BACKSCATTERING700",
    "DOWNWELLING_PAR",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--start", default="2025-04-18")
    ap.add_argument("--end", default="2025-04-30")
    ap.add_argument("--out", default="examples/data/docs/Growler_677_docs.nc")
    args = ap.parse_args()

    ds = xr.open_dataset(args.source)
    t = pd.to_datetime(ds["TIME"].values)
    rows = (t >= args.start) & (t < args.end)
    keep = [v for v in ds.variables if "N_MEASUREMENTS" not in ds[v].dims or v in KEEP]
    sub = ds[keep].isel(N_MEASUREMENTS=np.flatnonzero(rows))
    sub.attrs["history"] = (
        ds.attrs.get("history", "")
        + f"docs subset {args.start} to {args.end} of {args.source}\n"
    )
    encoding = {
        v: {"zlib": True, "complevel": 4}
        for v in sub.data_vars
        if np.issubdtype(sub[v].dtype, np.number)
    }
    sub.to_netcdf(args.out, encoding=encoding)
    print(f"wrote {args.out}: {sub.sizes['N_MEASUREMENTS']} rows")


if __name__ == "__main__":
    main()
