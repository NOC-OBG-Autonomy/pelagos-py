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
"""FastAPI backend for the pelagos_py config dashboard.

The dashboard is a *standalone* helper for authoring, validating and running
pipeline YAML configs. It never modifies the pipeline: it only introspects the
live step/QC registries (so newly-added steps appear automatically) and reuses
the pipeline's own ``parameter_spec`` validation, so what the dashboard accepts
is exactly what the pipeline accepts.

Run with::

    python dashboard/app.py            # then open http://localhost:8791
"""

from __future__ import annotations

import codecs
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# HDF5's C library prints its own error stack to stderr on a failed open (e.g. a
# truncated live NRT file); that's already raised as an exception, so silence it.
h5py._errors.silence_errors()

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _clean_ansi(text: str) -> str:
    # Keep SGR colour codes (the console renders them); drop cursor/clear-line ones.
    return _ANSI_RE.sub(lambda m: m.group(0) if m.group(0).endswith("m") else "", text)


# --- Locate the repo and make pelagos_py importable -------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parent
SRC_DIR = REPO_ROOT / "src"
STATIC_DIR = DASHBOARD_DIR / "static"
RUN_BOOTSTRAP = DASHBOARD_DIR / "run_bootstrap.py"
# Figures captured from the current run (see run_bootstrap.py); cleared per run.
FIG_DIR = DASHBOARD_DIR / "_run_figures"
FIG_DIR.mkdir(exist_ok=True)
# Float64 traces for exact click-lookups (run_figpoint), lazily loaded per run.
_FIGDATA_CACHE: dict[str, dict] = {}
# Configs authored in the dashboard live here by default.
CONFIG_DIR = DASHBOARD_DIR / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Importing the package runs discover_steps(), which populates the registries.
from pelagos_py.steps import STEP_CLASSES, QC_CLASSES, resolve_step_name  # noqa: E402
from pelagos_py.utils import parameter_spec  # noqa: E402
from pelagos_py.utils.qc_handling import QC_COMBINATRIX  # noqa: E402
from pelagos_py.utils.demo_data import DEMOS as DEMO_FILES, DEMO_DATA_DIR, MISSIONS  # noqa: E402
from pelagos_py.utils.valid_config_check import check_pipeline_variables  # noqa: E402
from pelagos_py.utils import config_builder, file_probe  # noqa: E402


# The top ``pipeline:`` block has no step schema, so its keys are described here.
PIPELINE_FIELDS = [
    {"name": "name", "type": "str", "required": False, "default": "",
     "description": "A short name for the pipeline."},
    {"name": "description", "type": "str", "required": False, "default": "",
     "description": "Longer description of the pipeline's purpose."},
    {"name": "out_directory", "type": "str", "required": False, "default": "./",
     "description": "Output directory for generated files (logs, reports, figures)."},
    {"name": "log_file", "type": "str", "required": False, "default": None,
     "description": "Log file name. Leave blank/null for console-only logging."},
    {"name": "on_step_fail", "type": "str", "options": ["pause", "skip", "stop"],
     "required": False, "default": "pause",
     "description": "What happens when a step fails. 'pause' shows its failure plot "
                     "so you can fix its parameters and re-run it, or skip it (outside "
                     "the dashboard this behaves like 'skip'). 'skip' moves straight "
                     "on to the next step. 'stop' ends the run."},
]


def _category(cls) -> str:
    # processing / qc / io, from the module path
    module = getattr(cls, "__module__", "")
    if ".quality_control" in module:
        return "quality_control"
    if ".processing" in module:
        return "processing"
    if ".input_output" in module:
        return "input_output"
    return "other"


def _short_doc(cls) -> str:
    doc = (cls.__doc__ or "").strip()
    if not doc:
        return ""
    # take up to the first blank line
    para = doc.split("\n\n", 1)[0]
    return " ".join(para.split())


def _describe_step(name: str, cls) -> dict:
    return {
        "name": name,
        "kind": "step",
        "category": _category(cls),
        "module": getattr(cls, "__module__", ""),
        "description": _short_doc(cls),
        # ``parameter_schema is None`` => not yet migrated to strict validation.
        "schema_declared": getattr(cls, "parameter_schema", None) is not None,
        "parameters": cls.describe_parameters(),
    }


def _describe_qc(name: str, cls) -> dict:
    return {
        "name": name,
        "kind": "qc",
        "category": "quality_control",
        "module": getattr(cls, "__module__", ""),
        "description": _short_doc(cls),
        "schema_declared": True,
        "parameters": cls.describe_parameters(),
        "required_variables": list(getattr(cls, "required_variables", []) or []),
        "qc_outputs": list(getattr(cls, "qc_outputs", []) or []),
    }


app = FastAPI(title="pelagos_py dashboard")


@app.middleware("http")
async def _no_cache(request, call_next):
    # Development convenience: never let the browser cache the UI or API.
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) or path == "/" or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# =============================== Introspection ==============================
@app.get("/api/registry")
def registry():
    """Everything the frontend needs to render the step palette and forms."""
    # The blank template scaffolds are registered but aren't real pipeline steps.
    def _is_template(cls):
        return ".templates" in getattr(cls, "__module__", "")

    steps = [
        _describe_step(n, c) for n, c in sorted(STEP_CLASSES.items())
        if not _is_template(c)
    ]
    qc = [
        _describe_qc(n, c) for n, c in sorted(QC_CLASSES.items())
        if not _is_template(c)
    ]
    return {
        "steps": steps,
        "qc": qc,
        "pipeline_fields": PIPELINE_FIELDS,
        "combinatrix": QC_COMBINATRIX.tolist(),  # Manual QC merges boxes with the same table
    }


# ================================ Validation ================================
class ValidatePayload(BaseModel):
    yaml_content: str


# check_pipeline_variables() logs its own "Validation Failed" line; validate fires
# on every keystroke and the UI shows the message anyway, so swallow it here.
_VALIDATE_LOGGER = logging.getLogger("pelagos_py.dashboard.validate")
_VALIDATE_LOGGER.addHandler(logging.NullHandler())
_VALIDATE_LOGGER.propagate = False


def _locate_variable_issue(steps, message):
    # (index, name) of the step (or Apply QC step) a check_pipeline_variables
    # message names; (None, None) renders as a pipeline-level issue.
    for index, step in enumerate(steps):
        name = step.get("name") if isinstance(step, dict) else None
        if name and f"'{name}'" in message:
            return index, name
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("name") != "Apply QC":
            continue
        qc_settings = (step.get("parameters") or {}).get("qc_settings") or {}
        for qc_name in qc_settings:
            if f"'{qc_name}'" in message:
                return index, step.get("name")
    return None, None


@app.post("/api/validate")
def validate(payload: ValidatePayload):
    """Validate a whole config with the pipeline's own ``parameter_spec``, returning
    per-step issues so the UI can point at the offending step."""
    try:
        config = yaml.safe_load(payload.yaml_content)
    except yaml.YAMLError as exc:
        return {"ok": False, "yaml_error": str(exc), "issues": []}

    if not isinstance(config, dict):
        return {"ok": False, "yaml_error": "Top-level config must be a mapping.",
                "issues": []}

    issues = []
    steps = config.get("steps") or []
    if not isinstance(steps, list):
        return {"ok": False, "yaml_error": "'steps' must be a list.", "issues": []}

    for index, step in enumerate(steps):
        if not isinstance(step, dict) or "name" not in step:
            issues.append({"index": index, "name": None,
                           "error": "Each step needs a 'name'."})
            continue
        name = step["name"]
        canonical = resolve_step_name(name)
        if canonical is None:
            issues.append({"index": index, "name": name,
                           "error": f"Unknown step '{name}'."})
            continue
        cls = STEP_CLASSES[canonical]

        schema = getattr(cls, "parameter_schema", None)
        if schema is None:
            continue  # step opted out of strict validation
        params = step.get("parameters") or {}
        try:
            parameter_spec.resolve(
                schema, params, label=name,
                allowed_extra=getattr(cls, "framework_parameters", ()),
            )
        except ValueError as exc:
            issues.append({"index": index, "name": name, "error": str(exc)})

    # Cross-step variable check (same as the pipeline's pre-run one); skipped when
    # a schema issue exists, since it would instantiate steps with bad parameters.
    if not issues:
        try:
            check_pipeline_variables(steps, _VALIDATE_LOGGER)
        except ValueError as exc:
            # Prefer the checker's own step_index: name-matching picks the first
            # step with that name, wrong when a QC test appears in several Apply QC steps.
            index = getattr(exc, "step_index", None)
            if index is not None:
                name = steps[index].get("name") if isinstance(steps[index], dict) else None
            else:
                index, name = _locate_variable_issue(steps, str(exc))
            issues.append({"index": index, "name": name, "error": str(exc)})

    return {"ok": not issues, "yaml_error": None, "issues": issues}


# ============================ Config management =============================
class SavePayload(BaseModel):
    name: str
    yaml_content: str


# Virtual, one per demo glider: no file on disk, YAML built per file by /api/build.
DEMO_CONFIGS = {f"demo_{key}.yaml" for key in DEMO_FILES}

# Read-only: the UI forks edits to custom_run_N.yaml and the API refuses to
# save/delete these, so a stale tab or hand-crafted request can't destroy them.
PROTECTED_CONFIGS = {"default.yaml"} | DEMO_CONFIGS


def _safe_config_path(name: str) -> Path:
    # Rejects path traversal out of CONFIG_DIR.
    candidate = (CONFIG_DIR / name).resolve()
    if candidate.parent != CONFIG_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid config name.")
    if candidate.suffix not in (".yaml", ".yml"):
        candidate = candidate.with_suffix(".yaml")
    return candidate


def _demo_key(config_name: str) -> str:
    return config_name[len("demo_"):-len(".yaml")]


def _demo_dest(config_name: str) -> Path | None:
    entry = DEMO_FILES.get(_demo_key(config_name))
    if entry is None:
        return None
    return REPO_ROOT / DEMO_DATA_DIR / entry.filename


@app.get("/api/configs")
def list_configs():
    files = sorted(
        p.name for p in CONFIG_DIR.iterdir()
        if p.is_file() and p.suffix in (".yaml", ".yml")
    )
    demo = sorted(DEMO_CONFIGS)  # virtual, so not filtered by `files`
    return {
        "configs": sorted(set(files) | DEMO_CONFIGS),
        "protected": sorted(PROTECTED_CONFIGS),
        "demo": demo,
        # Grouped by deployment mission, in picker display order.
        "missions": {
            mission: [f"demo_{key}.yaml" for key in keys]
            for mission, keys in MISSIONS.items()
        },
        # Glider names aren't unique across missions (nor NRT vs Full), hence labels.
        "labels": {f"demo_{key}.yaml": entry.display_label for key, entry in DEMO_FILES.items()},
        # Non-demo protected configs, shown as their own "Default" group.
        "reference": sorted((PROTECTED_CONFIGS - DEMO_CONFIGS) & set(files)),
        "downloaded": sorted(name for name in demo if _demo_dest(name).exists()),
        "sizes": {name: _demo_dest(name).stat().st_size
                  for name in demo if _demo_dest(name).exists()},
    }


def _reveal(folder: Path) -> dict:
    # Open a folder in the OS file browser (on the server machine).
    if sys.platform == "darwin":
        cmd = ["open", str(folder)]
    elif os.name == "nt":
        cmd = ["explorer", str(folder)]
    else:
        cmd = ["xdg-open", str(folder)]
    try:
        subprocess.Popen(cmd)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not open {folder}: {exc}"
        ) from exc
    return {"status": "opened", "path": str(folder)}


@app.post("/api/configs/reveal")
def reveal_configs():
    return _reveal(CONFIG_DIR)


# ================================ Demo files ================================
def _no_run_in_flight():
    if _run.is_running():
        raise HTTPException(status_code=409, detail="Stop the running pipeline first.")


def _delete_demo(config_name: str) -> bool:
    dest = _demo_dest(config_name)
    if dest is None:
        return False
    dest.with_name(dest.name + ".part").unlink(missing_ok=True)
    if not dest.exists():
        return False
    dest.unlink()
    return True


@app.delete("/api/demos/{name}")
def delete_demo(name: str):
    if name not in DEMO_CONFIGS:
        raise HTTPException(status_code=404, detail="Unknown demo.")
    _no_run_in_flight()
    return {"status": "deleted" if _delete_demo(name) else "absent", "name": name}


@app.post("/api/demos/clean")
def clean_demos():
    _no_run_in_flight()
    removed = [name for name in sorted(DEMO_CONFIGS) if _delete_demo(name)]
    return {"status": "deleted", "removed": removed}


# ================================= Outputs =================================
# Everything a run leaves behind (reports, exports, logs, kept report figures),
# so a pip-installed user can find and clear them without knowing the folder.
_OUTPUT_KINDS = {
    ".pdf": "report", ".log": "log", ".nc": "data", ".csv": "data",
    ".parquet": "data", ".h5": "data", ".hdf5": "data", ".rst": "report",
}
_DEMO_INPUTS = {entry.filename for entry in DEMO_FILES.values()}
_listed_outputs: set[Path] = set()  # only these may be served or deleted


class OutputsPayload(BaseModel):
    dirs: list[str] = []
    inputs: list[str] = []  # data files the config reads: never shown as outputs


def _output_dirs(dirs: list[str]) -> list[Path]:
    seen, out = set(), []
    for d in [DEMO_DATA_DIR, *dirs]:
        path = Path(d)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        out.append(path)
    return out


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _list_outputs(payload: OutputsPayload) -> dict:
    inputs = _DEMO_INPUTS | {Path(f).name for f in payload.inputs if f}
    dirs = _output_dirs(payload.dirs)
    files = []
    _listed_outputs.clear()
    for d in dirs:
        for entry in sorted(d.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if "_report_figures_" not in entry.name and entry.name != "_build":
                    continue
                kind, size = "figures", _dir_size(entry)
            else:
                kind = _OUTPUT_KINDS.get(entry.suffix.lower())
                if kind is None or entry.name in inputs:
                    continue
                size = entry.stat().st_size
            _listed_outputs.add(entry)
            files.append({
                "path": str(entry), "name": entry.name, "dir": str(d),
                "kind": kind, "size": size, "mtime": entry.stat().st_mtime,
            })
    files.sort(key=lambda f: -f["mtime"])
    return {"dirs": [str(d) for d in dirs], "files": files}


@app.post("/api/outputs")
def list_outputs(payload: OutputsPayload):
    return _list_outputs(payload)


def _listed_output(path: str) -> Path:
    p = Path(path)
    if p not in _listed_outputs or not p.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return p


@app.get("/api/outputs/file")
def output_file(path: str):
    p = _listed_output(path)
    if p.is_dir():
        raise HTTPException(status_code=400, detail="Not a file.")
    inline = p.suffix.lower() in (".pdf", ".log", ".rst")
    return FileResponse(
        p, media_type="application/pdf" if p.suffix.lower() == ".pdf" else None,
        headers={"Content-Disposition": f'{"inline" if inline else "attachment"}; filename="{p.name}"'},
    )


def _remove(p: Path):
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        p.unlink(missing_ok=True)


@app.delete("/api/outputs/file")
def delete_output(path: str):
    _no_run_in_flight()
    p = _listed_output(path)
    _remove(p)
    _listed_outputs.discard(p)
    return {"status": "deleted", "path": str(p)}


@app.post("/api/outputs/clean")
def clean_outputs(payload: OutputsPayload):
    _no_run_in_flight()
    listing = _list_outputs(payload)
    for f in listing["files"]:
        _remove(Path(f["path"]))
    _listed_outputs.clear()
    return {"status": "deleted", "removed": len(listing["files"])}


class RevealPayload(BaseModel):
    path: str = ""


@app.post("/api/outputs/reveal")
def reveal_outputs(payload: RevealPayload):
    dirs = _output_dirs([payload.path] if payload.path else [])
    if not dirs:
        raise HTTPException(status_code=404, detail="Folder not found.")
    return _reveal(dirs[-1] if payload.path else dirs[0])


class BrowsePayload(BaseModel):
    start: str = ""


@app.post("/api/browse")
def browse_file(payload: BrowsePayload):
    """Open the OS file picker on the server and return the chosen path
    (browsers never expose a picked file's real path)."""
    start = Path(payload.start).expanduser() if payload.start else None
    start_dir = start.parent if start and start.parent.is_dir() else Path.cwd()
    if sys.platform == "darwin":
        script = (
            'POSIX path of (choose file with prompt "Choose an input NetCDF file" '
            f'default location POSIX file "{start_dir}")'
        )
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if proc.returncode != 0:  # user cancelled
            return {"path": None}
        return {"path": proc.stdout.strip()}
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"No file dialog available: {exc}")
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askopenfilename(initialdir=str(start_dir), title="Choose an input NetCDF file")
    root.destroy()
    return {"path": chosen or None}


def _ensure_demo_file(config_name: str) -> None:
    # Fallback for get_demo_file.py not having been run; unlike it, doesn't trim churchill.
    dest = _demo_dest(config_name)
    if dest is None or dest.exists():
        return
    entry = DEMO_FILES[_demo_key(config_name)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    import requests

    # Files are 100s of MB: bound only the connect phase, not the transfer.
    response = requests.get(entry.url, stream=True, timeout=(15, None))
    response.raise_for_status()
    total = int(response.headers.get("Content-Length") or 0)
    done = 0
    _DOWNLOADS[config_name] = (0, total)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with open(tmp, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                _DOWNLOADS[config_name] = (done, total)
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)  # left behind only if the download failed
        _DOWNLOADS.pop(config_name, None)


# Demo downloads in flight: config name -> (bytes done, total or 0 if unknown).
_DOWNLOADS: dict[str, tuple[int, int]] = {}


@app.get("/api/demos/progress")
def demo_progress():
    return {name: {"done": d, "total": t} for name, (d, t) in _DOWNLOADS.items()}


class BuildPayload(BaseModel):
    file_path: str
    choices: dict | None = None
    description: str | None = None


def _template_text() -> str:
    return (CONFIG_DIR / "default.yaml").read_text()


@app.post("/api/build/decisions")
def build_decisions(payload: BuildPayload):
    """What the template must change for this file (see config_builder.decisions)."""
    path = _resolve_inspect_path(payload.file_path)
    probe = file_probe.probe_file(path)
    if probe is None:
        raise HTTPException(status_code=400, detail=f"Could not read '{path.name}'.")
    return {"path": str(path), "decisions": config_builder.decisions(probe)}


@app.post("/api/build")
def build_config(payload: BuildPayload):
    """The template adapted to this file with the given (or default) choices."""
    path = _resolve_inspect_path(payload.file_path)
    probe = file_probe.probe_file(path)
    if probe is None:
        raise HTTPException(status_code=400, detail=f"Could not read '{path.name}'.")
    # Keep demo paths repo-relative so the config reads the same on any checkout.
    file_path = payload.file_path
    if not Path(file_path).is_absolute():
        file_path = str(path.relative_to(REPO_ROOT))
    return {"yaml_content": config_builder.build(
        _template_text(), file_path, probe, payload.choices, payload.description,
    )}


@app.get("/api/configs/{name}")
def load_config(name: str):
    demo_name = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    if demo_name in DEMO_CONFIGS:
        try:
            _ensure_demo_file(demo_name)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not download demo data: {exc}"
            ) from exc
        # Demo configs are built per file (see /api/build) once the user confirms
        # the builder's decisions, so this only says which file to build for.
        entry = DEMO_FILES[_demo_key(demo_name)]
        return {
            "name": demo_name,
            "build": {
                "file_path": f"{DEMO_DATA_DIR}/{entry.filename}",
                "description": f"A demo pipeline using {entry.display_label} data.",
            },
        }
    path = _safe_config_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Config not found.")
    return {"name": path.name, "yaml_content": path.read_text()}


@app.post("/api/configs")
def save_config(payload: SavePayload):
    path = _safe_config_path(payload.name)
    if path.name in PROTECTED_CONFIGS:
        raise HTTPException(
            status_code=403,
            detail=f"'{path.name}' is a locked reference config. "
                   "Save your changes under a different name.",
        )
    path.write_text(payload.yaml_content)
    return {"status": "saved", "name": path.name}


@app.delete("/api/configs/{name}")
def delete_config(name: str):
    path = _safe_config_path(name)
    if path.name in PROTECTED_CONFIGS:
        raise HTTPException(
            status_code=403,
            detail=f"'{path.name}' is a locked reference config and cannot be deleted.",
        )
    if path.exists():
        path.unlink()
    return {"status": "deleted", "name": path.name}


# ============================== Pipeline runner =============================
class _Run:
    """Holds the single active pipeline subprocess and its captured log lines.

    Only one run at a time -- the dashboard is a single-user local tool. Output
    is pumped off the subprocess's merged stdout/stderr by a background thread.

    Progress bars (tqdm) redraw one line with a carriage return (``\\r``) rather
    than emitting a new line each tick. Reading in binary preserves those ``\\r``
    boundaries so a whole bar collapses to a single, in-place-updating line in
    the console instead of thousands of spam lines.

    State is an append-only list of committed ``lines`` plus the latest transient
    progress redraw (``live``). The SSE endpoint reads these by index rather than
    draining a queue, so any number of clients -- including one reconnecting after
    a page refresh mid-run -- each replay the full log independently, with no
    duplicated or stolen events.
    """

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []  # committed lines, append-only (replayable)
        self.live: str | None = None  # latest transient progress redraw, if any
        self.live_after = 0  # index in `lines` the current `live` follows
        self.finished = False
        self.returncode: int | None = None
        self._lock = threading.Lock()
        # Processing clock mirrored from the runner's __PELAGOS_TIME__ markers
        # (see _commit): the RAM sampler stamps each sample with active seconds
        # and stays quiet while the run is paused, so the meter's x-axis is the
        # same clock the dashboard's runtime readout shows.
        self._active = 0.0
        self._since: float | None = None

    def _sample_mem(self, proc):
        # Poll the child's RSS from here rather than inside the run: costs the
        # pipeline nothing and can't interleave with its own console output.
        try:
            import psutil
            child = psutil.Process(proc.pid)
        except Exception:  # noqa: BLE001 - the meter is a bonus, never fatal
            return
        while proc.poll() is None:
            try:
                rss = child.memory_info().rss / 1024 ** 2
            except Exception:  # noqa: BLE001 - child just exited
                return
            with self._lock:
                if self._since is not None:
                    x = self._active + time.time() - self._since
                    self.lines.append(f"__PELAGOS_SAMPLE__ {x:.1f}\t{rss:.1f}")
            time.sleep(0.5)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, config_path: Path):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Output goes to a pipe, not a terminal, so the pipeline would otherwise
        # drop its colour and disable its tqdm bars (see utils/log_levels.py
        # _supports_color, which FORCE_COLOR overrides). The browser console
        # renders both, so ask for them. COLUMNS gives the bars a sane width,
        # since there is no terminal to measure.
        env["FORCE_COLOR"] = "1"
        env.pop("NO_COLOR", None)
        env["COLUMNS"] = "110"
        # HDF5 file locking can transiently fail (Errno -101) when a file was
        # just opened and closed by another process (e.g. the variable-check
        # subprocess in valid_config_check.py) -- disable it for the run.
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # Ensure the subprocess can import pelagos_py from src/.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SRC_DIR), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        # Fresh figure dir per run so the Plots tab only shows this run's plots.
        # .json/.f32 are the interactive plot spec and its float32 data,
        # _full.npz the float64 copy for exact point lookups, beside each .png.
        for pattern in ("*.png", "*.json", "*.f32", "*_full.npz"):
            for old in FIG_DIR.glob(pattern):
                old.unlink(missing_ok=True)
        _FIGDATA_CACHE.clear()
        # run_bootstrap.py redirects plt.show to save diagnostic figures into
        # FIG_DIR (and catches the Stop-button SIGINT) -- see that file.
        self.proc = subprocess.Popen(
            [sys.executable, str(RUN_BOOTSTRAP), str(config_path), str(FIG_DIR)],
            cwd=str(REPO_ROOT),  # relative paths in configs resolve from repo root
            stdin=subprocess.PIPE,  # control channel for pause/continue/rerun
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=-1,  # buffered binary: gives a BufferedReader (has read1);
            # \r is preserved either way, and read1 returns as soon as any
            # bytes arrive so progress-bar ticks still stream promptly.
        )
        self.lines = []
        self.live = None
        self.live_after = 0
        self.finished = False
        self.returncode = None
        self._active, self._since = 0.0, None
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._sample_mem, args=(self.proc,), daemon=True).start()

    def _commit(self, text: str):
        """A finished (newline-terminated) line: append it and clear live progress."""
        with self._lock:
            self.lines.append(text)
            self.live = None
            self.live_after = len(self.lines)
            if text.startswith("__PELAGOS_TIME__ "):
                try:
                    active, paused, _ = text.split(" ", 1)[1].split("\t")
                    self._active = float(active)
                    self._since = None if paused.strip() == "1" else time.time()
                except ValueError:
                    pass

    def _progress(self, text: str):
        """A transient in-place redraw (bar tick): update live, don't accumulate."""
        with self._lock:
            self.live = text
            self.live_after = len(self.lines)

    def _pump(self):
        assert self.proc is not None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stream = self.proc.stdout  # binary BufferedReader
        seg = ""  # chars seen since the last \r or \n
        pending_cr = False  # saw a \r; a \n next means it was a \r\n line ending
        while True:
            chunk = stream.read1(4096)  # type: ignore[union-attr]
            if not chunk:
                break
            for ch in decoder.decode(chunk):
                if ch == "\n":
                    # A \r right before this \n is a Windows (\r\n) line ending,
                    # not a redraw -- treat the whole pair as one newline.
                    pending_cr = False
                    self._commit(_clean_ansi(seg))
                    seg = ""
                    continue
                if pending_cr:
                    # The earlier \r had no \n after it: a real in-place redraw
                    # (tqdm bar tick). Emit what was drawn, then start fresh.
                    cleaned = _clean_ansi(seg)
                    if cleaned:
                        self._progress(cleaned)
                    seg = ""
                    pending_cr = False
                if ch == "\r":
                    pending_cr = True
                else:
                    seg += ch
        if seg:  # trailing text with no final newline
            if pending_cr:
                cleaned = _clean_ansi(seg)
                if cleaned:
                    self._progress(cleaned)
            else:
                self._commit(_clean_ansi(seg))
        self.returncode = self.proc.wait()
        self.finished = True

    def stop(self):
        if self.is_running():
            # SIGINT (not SIGTERM) so the child raises KeyboardInterrupt and runs
            # its Python cleanup -- atexit/finally, multiprocessing pool shutdown --
            # releasing pool semaphores instead of leaking them on abrupt exit.
            # Escalate to SIGTERM then SIGKILL if it doesn't stop promptly.
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

    def send(self, command: str):
        """Write a control line to the paused run's stdin.

        Drives the interactive pause protocol in run_bootstrap.py:
        ``continue`` / ``rerun <json>`` / ``stop``. No-op (swallowed) if the
        process has already exited or its stdin is gone.
        """
        if self.proc is not None and self.proc.stdin is not None and self.is_running():
            try:
                self.proc.stdin.write((command + "\n").encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass


_run = _Run()


class RunPayload(BaseModel):
    yaml_content: str


@app.post("/api/run")
def run_pipeline(payload: RunPayload):
    if _run.is_running():
        raise HTTPException(status_code=409, detail="A pipeline is already running.")
    # Persist the exact YAML being run so the subprocess (and the user) can see it.
    run_path = CONFIG_DIR / "_last_run.yaml"
    run_path.write_text(payload.yaml_content)
    _run.start(run_path)
    return {"status": "started"}


@app.post("/api/run/stop")
def stop_pipeline():
    _run.stop()
    return {"status": "stopping"}


class RerunPayload(BaseModel):
    parameters: dict = {}


@app.post("/api/run/continue")
def continue_run():
    """Resume a run paused at a diagnostics step (interactive stepping)."""
    _run.send("continue")
    return {"status": "continued"}


@app.post("/api/run/rerun")
def rerun_step(payload: RerunPayload):
    """Re-run the currently paused step with edited parameters, then re-pause."""
    _run.send("rerun " + json.dumps(payload.parameters))
    return {"status": "rerunning"}


@app.get("/api/run/status")
def run_status():
    return {
        "running": _run.is_running(),
        "finished": _run.finished,
        "returncode": _run.returncode,
        "line_count": len(_run.lines),
    }


@app.get("/api/run/stream")
def stream_logs():
    """Server-Sent Events stream of the current run's log lines.

    Reads the run's append-only state by index, so it replays everything already
    captured (a client connecting late -- e.g. after a mid-run page refresh --
    sees the whole run) and then tails new output until the process exits. Each
    connection keeps its own cursors, so reconnecting never steals or duplicates
    events.
    """
    def frame(kind: str, text: str) -> str:
        # 'line' -> default SSE event (message); 'progress' -> named event.
        prefix = "" if kind == "line" else f"event: {kind}\n"
        return f"{prefix}data: {text}\n\n"

    def event_gen():
        cursor = 0  # next committed-line index to emit
        last_live = None  # last progress text emitted, to avoid repeats
        idle = 0.0
        while True:
            with _run._lock:
                new_lines = _run.lines[cursor:]
                cursor = len(_run.lines)
                live = _run.live
                finished = _run.finished
                returncode = _run.returncode
            for line in new_lines:
                last_live = None  # a committed line supersedes any live redraw
                yield frame("line", line)
            if live is not None and live != last_live:
                last_live = live
                yield frame("progress", live)
            if finished and cursor >= len(_run.lines):
                yield f"event: end\ndata: {returncode}\n\n"
                return
            if not new_lines:
                idle += 0.1
                if idle >= 15.0:  # periodic comment frame keeps the connection open
                    idle = 0.0
                    yield ": keep-alive\n\n"
            else:
                idle = 0.0
            time.sleep(0.1)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# Files run_bootstrap.py writes per captured figure: PNG, plot spec, float32 traces.
_FIG_FILES = {
    ".png": ("image/png", "Figure"),
    ".json": ("application/json", "Plot spec"),
    ".f32": ("application/octet-stream", "Plot data"),
}


def _fig_file(name: str, suffix: str) -> FileResponse:
    media_type, label = _FIG_FILES[suffix]
    path = FIG_DIR / Path(name).name  # .name strips directories so a crafted name can't escape
    if path.suffix != suffix or not path.is_file():
        raise HTTPException(status_code=404, detail=f"{label} not found.")
    return FileResponse(path, media_type=media_type)


@app.get("/api/run/figure/{name}")
def run_figure(name: str):
    return _fig_file(name, ".png")


@app.get("/api/run/figspec/{name}")
def run_figspec(name: str):
    return _fig_file(name, ".json")


@app.get("/api/run/figbin/{name}")
def run_figbin(name: str):
    # A plain file gives the browser a Content-Length for its progress bar.
    return _fig_file(name, ".f32")


@app.get("/api/run/figpoint/{name}")
def run_figpoint(name: str, panel: int, trace: int, index: int):
    """Exact float64 ``x``/``y`` of one plotted point (dates as ISO strings),
    for the viewer's click tooltip; the drawn data is float32."""
    stem = Path(Path(name).name).stem
    if stem not in _FIGDATA_CACHE:
        path = FIG_DIR / (stem + "_full.npz")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No point data for this figure.")
        with np.load(path) as npz:
            _FIGDATA_CACHE[stem] = {k: npz[k] for k in npz.files}
    data = _FIGDATA_CACHE[stem]
    key = f"{panel}_{trace}"
    if f"{key}_x" not in data or not 0 <= index < len(data[f"{key}_x"]):
        raise HTTPException(status_code=404, detail="No such point.")
    spec = json.loads((FIG_DIR / (stem + ".json")).read_text())["panels"][panel]

    def fmt(v, is_date):
        if not np.isfinite(v):
            return None
        if is_date:
            return pd.Timestamp(v, unit="ms").isoformat(timespec="milliseconds")
        return float(v)

    return {"x": fmt(data[f"{key}_x"][index], spec["xdate"]),
            "y": fmt(data[f"{key}_y"][index], spec["ydate"])}


@app.get("/api/run/report")
def run_report(path: str):
    """Serve a PDF report produced by the current run (path from its
    ``__PELAGOS_REPORT__`` marker), inline so the browser previews it."""
    p = Path(path)
    if p.suffix.lower() != ".pdf" or not p.is_file():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(
        p, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{p.name}"'},
    )


# ================================== Inspect ==================================
def _resolve_inspect_path(file_path):
    # Relative paths resolve against the repo root, as in the pipeline itself.
    if not file_path:
        raise HTTPException(status_code=400, detail="No file_path given.")
    path = Path(file_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return path


@app.get("/api/inspect")
def inspect_file(file_path: str):
    """Variables, global attributes and sensors of a NetCDF file, for the Inspect tab."""
    path = _resolve_inspect_path(file_path)
    try:
        with xr.open_dataset(path) as ds:
            variables = [
                {
                    "name": name,
                    "units": var.attrs.get("units", ""),
                    "description": var.attrs.get("long_name") or var.attrs.get("comment") or "",
                    "dtype": str(var.dtype),
                }
                for name, var in ds.variables.items()
            ]
            global_attrs = {k: str(v) for k, v in ds.attrs.items()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open '{path.name}': {exc}")

    # 'instrument' global attribute may be "a, b, c" or a Python-list-style string.
    instr_key = next((k for k in global_attrs if k.lower() == "instrument"), None)
    raw = global_attrs.get(instr_key, "") if instr_key else ""
    sensors = [
        s.strip().strip("'\"")
        for s in raw.strip().lstrip("[").rstrip("]").split(",")
        if s.strip()
    ]

    return {
        "path": str(path),
        "variables": variables,
        "global_attributes": global_attrs,
        "sensors": sensors,
    }


_INSPECT_PLOT_MAX = 20_000
_inspect_plot_lock = threading.Lock()  # pyplot is not thread-safe


@app.get("/api/inspect/plot")
def inspect_plot(file_path: str, var: str):
    """PNG of ``var`` against TIME (coloured by ``{var}_QC`` if present) plus
    point counts, for a clicked variable in the Inspect tab. Evenly subsampled
    to at most _INSPECT_PLOT_MAX non-NaN points so it renders in well under a second."""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pelagos_py.utils import fig_spec

    path = _resolve_inspect_path(file_path)
    try:
        with xr.open_dataset(path) as ds:
            if var not in ds.variables:
                raise HTTPException(status_code=404, detail=f"'{var}' not in file.")
            da = ds[var]
            numeric = np.issubdtype(da.dtype, np.number) or np.issubdtype(da.dtype, np.datetime64)
            if da.ndim == 0:
                return {"value": str(da.values)}
            if da.ndim == 1 and not numeric and da.size <= 200:
                return {"value": ", ".join(str(v) for v in da.values)}
            if da.ndim != 1 or not numeric:
                shape = " x ".join(f"{d}={n}" for d, n in zip(da.dims, da.shape))
                raise HTTPException(status_code=400, detail=f"{da.dtype} ({shape}), not plotted.")
            y = da.values
            units = da.attrs.get("units", "")
            x_is_time = var != "TIME" and "TIME" in ds.variables and ds["TIME"].dims == da.dims
            x = ds["TIME"].values if x_is_time else np.arange(y.size)
            qc_name = f"{var}_QC"
            flags = ds[qc_name].values if qc_name in ds.variables and ds[qc_name].dims == da.dims else None
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read '{var}': {exc}")

    valid = ~pd.isnull(y)
    n_total, n_valid = int(y.size), int(valid.sum())
    idx = np.flatnonzero(valid)
    if idx.size > _INSPECT_PLOT_MAX:
        idx = idx[np.linspace(0, idx.size - 1, _INSPECT_PLOT_MAX).astype(int)]

    with _inspect_plot_lock:
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        if flags is not None:
            fig_spec.flag_points(ax, x[idx], y[idx], flags[idx])
            ax.legend(fontsize=fig_spec.FS_LEGEND, loc="best", frameon=False, markerscale=2)
        else:
            fig_spec.points(ax, x[idx], y[idx], color=fig_spec.CATEGORY[1])
        fig_spec.style_axes(
            ax, xlabel="TIME" if x_is_time else "index", ylabel=fig_spec.axis_label(var, units)
        )
        if x_is_time:
            fig_spec.date_axis(ax, index=x)
        if var.startswith(("PRES", "DEPTH")):
            ax.invert_yaxis()
        # Page-element look: transparent, no grid or box, fixed margins so every plot is the same size.
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.12)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)

    return {
        "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
        "n_total": n_total,
        "n_valid": n_valid,
        "n_shown": int(idx.size),
        "qc": flags is not None,
    }


# ================================ Static site ===============================
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import webbrowser

    import uvicorn

    # Bind to the loopback IP but show the friendlier hostname in the URL.
    print("pelagos_py dashboard -> http://localhost:8791")
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:8791")).start()
    uvicorn.run(app, host="127.0.0.1", port=8791, log_level="warning")
