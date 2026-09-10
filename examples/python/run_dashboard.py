"""Launch the pelagos_py config dashboard (same as ``python dashboard/app.py``, from anywhere)."""

import runpy
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])  # repo root: app.py resolves paths from there
runpy.run_path("dashboard/app.py", run_name="__main__")
