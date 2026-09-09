"""Subprocess entry point that rasterises pickled figures (see diagnostic_capture.save_figure)."""

import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")

for line in sys.stdin:
    path = line.rstrip("\n")
    try:
        with open(path + ".fig", "rb") as fh:
            fig, kwargs = pickle.load(fh)
        os.remove(path + ".fig")
        fig.savefig(path, **kwargs)
        print("ok", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported back, the caller falls back
        print(f"err {type(exc).__name__}: {exc}", flush=True)
