"""Subprocess entry point that rasterises pickled figures (see diagnostic_capture.save_figure)."""

import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - after the backend is set

for line in sys.stdin:
    path = line.rstrip("\n")
    fig = None
    try:
        with open(path + ".fig", "rb") as fh:
            fig, kwargs = pickle.load(fh)
        fig.savefig(path, **kwargs)
        os.remove(
            path + ".fig"
        )  # keep the pickle on failure so wait_for_saves() can retry here
        print("ok", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported back, the caller falls back
        print(f"err {type(exc).__name__}: {exc}", flush=True)
    finally:
        # unpickling registers the figure with pyplot, so this long-lived worker must close it
        if fig is not None:
            plt.close(fig)
