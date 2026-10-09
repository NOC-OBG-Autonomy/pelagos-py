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
"""FastAPI backend for the config dashboard: build, validate and run pipeline configs.

Start it with ``pelagos-py dashboard``.
"""

from __future__ import annotations

import atexit
import codecs
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

# Importing the package runs discover_steps(), which populates the registries.
from pelagos_py.steps import STEP_CLASSES, QC_CLASSES, resolve_step_name
from pelagos_py.utils import parameter_spec
from pelagos_py.utils.qc_handling import QC_COMBINATRIX
from pelagos_py.utils.demo_data import DEMOS as DEMO_FILES, DEMO_DATA_DIR, MISSIONS, WORKSPACE_DIR, get_demo_file
from pelagos_py.utils.valid_config_check import check_pipeline_variables
from pelagos_py.utils import config_builder, file_probe

# HDF5 prints its own error stack on a failed open, which is already raised as an exception
h5py._errors.silence_errors()

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _clean_ansi(text: str) -> str:
    # keep colour codes, drop cursor/clear-line ones
    return _ANSI_RE.sub(lambda m: m.group(0) if m.group(0).endswith("m") else "", text)


DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"
RUN_BOOTSTRAP = DASHBOARD_DIR / "run_bootstrap.py"
# one folder per dashboard process, so two dashboards can't delete each other's figures
FIG_DIR = Path(tempfile.mkdtemp(prefix="pelagos-py-figures-"))
atexit.register(shutil.rmtree, FIG_DIR, ignore_errors=True)
_FIGDATA_CACHE: dict[str, dict] = {}
CONFIG_DIR = WORKSPACE_DIR / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


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
    para = doc.split("\n\n", 1)[0]
    return " ".join(para.split())


def _describe_step(name: str, cls) -> dict:
    return {
        "name": name,
        "kind": "step",
        "category": _category(cls),
        "module": getattr(cls, "__module__", ""),
        "description": _short_doc(cls),
        "beta": cls.beta,
        "schema_declared": getattr(cls, "parameter_schema", None) is not None,
        "parameters": cls.describe_parameters(),
        "more_diagnostics": any(
            level is False for _, level in (getattr(cls, "diagnostic_figures", None) or {}).values()
        ),
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
# blocks DNS-rebinding pages from driving the local API
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])


@app.middleware("http")
async def _no_cache(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) or path == "/" or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# =============================== Introspection ==============================
@app.get("/api/registry")
def registry():
    """Everything the frontend needs to render the step palette and forms."""
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
        "combinatrix": QC_COMBINATRIX.tolist(),
    }


# ================================ Validation ================================
class ValidatePayload(BaseModel):
    yaml_content: str


# validate runs on every keystroke and the UI shows the error, so mute the checker's log line
_VALIDATE_LOGGER = logging.getLogger("pelagos_py.dashboard.validate")
_VALIDATE_LOGGER.addHandler(logging.NullHandler())
_VALIDATE_LOGGER.propagate = False


def _locate_variable_issue(steps, message):
    # (None, None) shows as a pipeline-level issue
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


def _data_file_error(file_path):
    if not file_path or not str(file_path).strip():
        return None
    path = Path(str(file_path)).expanduser()
    if path.suffix.lower() != ".nc":
        return f"Data file '{file_path}' is not a NetCDF (.nc) file."
    if not path.is_file():
        return f"Could not find data file '{file_path}'."
    return None


@app.post("/api/validate")
def validate(payload: ValidatePayload):
    """Validate a config with the pipeline's own checks and return per-step issues."""
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
            continue
        params = step.get("parameters") or {}
        try:
            parameter_spec.resolve(
                schema, params, label=name,
                allowed_extra=getattr(cls, "framework_parameters", ()),
            )
        except ValueError as exc:
            issues.append({"index": index, "name": name, "error": str(exc)})
        if canonical == "Load OG1":
            file_error = _data_file_error(params.get("file_path"))
            if file_error:
                issues.append({"index": index, "name": name, "error": file_error})

    # the variable check instantiates steps, so only run it once the parameters are valid
    if not issues:
        try:
            check_pipeline_variables(steps, _VALIDATE_LOGGER)
        except ValueError as exc:
            # name-matching is wrong when a QC test appears in several Apply QC steps
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


# virtual: no file on disk, the YAML is built per file by /api/build
DEMO_CONFIGS = {f"demo_{key}.yaml" for key in DEMO_FILES}

DEFAULT_CONFIG_NAME = "default.yaml"

# the UI forks edits to these into custom_run_N.yaml
PROTECTED_CONFIGS = {DEFAULT_CONFIG_NAME} | DEMO_CONFIGS


def _safe_config_path(name: str) -> Path:
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
    return DEMO_DATA_DIR / entry.filename


@app.get("/api/configs")
def list_configs():
    files = sorted(
        p.name for p in CONFIG_DIR.iterdir()
        if p.is_file() and p.suffix in (".yaml", ".yml")
    )
    demo = sorted(DEMO_CONFIGS)
    return {
        "configs": sorted(set(files) | DEMO_CONFIGS | {DEFAULT_CONFIG_NAME}),
        "protected": sorted(PROTECTED_CONFIGS),
        "demo": demo,
        "missions": {
            mission: [f"demo_{key}.yaml" for key in keys]
            for mission, keys in MISSIONS.items()
        },
        # glider names repeat across missions and NRT/full, hence labels
        "labels": {f"demo_{key}.yaml": entry.display_label for key, entry in DEMO_FILES.items()},
        "gliders": {f"demo_{key}.yaml": entry.label for key, entry in DEMO_FILES.items()},
        "modes": {f"demo_{key}.yaml": entry.mode for key, entry in DEMO_FILES.items()},
        "reference": sorted((PROTECTED_CONFIGS - DEMO_CONFIGS) & (set(files) | {DEFAULT_CONFIG_NAME})),
        "downloaded": sorted(name for name in demo if _demo_dest(name).exists()),
        "sizes": {name: _demo_dest(name).stat().st_size
                  for name in demo if _demo_dest(name).exists()},
    }


def _reveal(folder: Path) -> dict:
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


@app.post("/api/demos/{name}/download")
def download_demo(name: str):
    if name not in DEMO_CONFIGS:
        raise HTTPException(status_code=404, detail="Unknown demo.")
    try:
        _ensure_demo_file(name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not download demo data: {exc}") from exc
    return {"status": "downloaded", "name": name}


@app.post("/api/demos/clean")
def clean_demos():
    _no_run_in_flight()
    removed = [name for name in sorted(DEMO_CONFIGS) if _delete_demo(name)]
    return {"status": "deleted", "removed": removed}


# ================================= Outputs =================================
_OUTPUT_KINDS = {
    ".pdf": "report", ".log": "log", ".nc": "data", ".csv": "data",
    ".parquet": "data", ".h5": "data", ".hdf5": "data", ".rst": "report",
}
_listed_outputs: set[Path] = set()  # only listed files may be served or deleted


class OutputsPayload(BaseModel):
    dirs: list[str] = []
    exports: list[str] = []


def _output_dirs(dirs: list[str]) -> list[Path]:
    seen, out = set(), []
    for d in [DEMO_DATA_DIR, *dirs]:
        path = Path(d)
        if not path.is_absolute():
            path = WORKSPACE_DIR / path
        path = path.resolve()
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        out.append(path)
    return out


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _list_outputs(payload: OutputsPayload) -> dict:
    exports = {Path(f).name for f in payload.exports if f}
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
                if kind is None or (kind == "data" and entry.name not in exports):
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


class PathsPayload(BaseModel):
    paths: list[str] = []


@app.post("/api/files/info")
def files_info(payload: PathsPayload):
    out = {}
    for p in payload.paths:
        f = Path(p).expanduser()
        out[p] = {"exists": f.is_file(), "size": f.stat().st_size if f.is_file() else 0}
    return out


@app.post("/api/files/reveal")
def reveal_file(payload: RevealPayload):
    f = Path(payload.path).expanduser()
    if not f.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return _reveal(f.parent)


class BrowsePayload(BaseModel):
    start: str = ""


@app.post("/api/files/pick")
def pick_files(payload: BrowsePayload):
    # a picked folder adds all its .nc files
    start = Path(payload.start).expanduser() if payload.start else None
    start_dir = start.parent if start and start.parent.is_dir() else Path.cwd()
    if sys.platform == "darwin":
        script = (
            'on run argv\n'
            'set fs to choose file of type {"public.folder", "public.data"} with prompt '
            '"Choose OG1 NetCDF files or a folder" with multiple selections allowed default location POSIX file (item 1 of argv)\n'
            'if class of fs is not list then set fs to {fs}\n'
            'set out to ""\nrepeat with f in fs\nset out to out & POSIX path of f & linefeed\nend repeat\nreturn out\n'
            'end run'
        )
        proc = subprocess.run(["osascript", "-e", script, str(start_dir)], capture_output=True, text=True)
        chosen = proc.stdout.splitlines() if proc.returncode == 0 else []
    else:
        try:
            import tkinter
            from tkinter import filedialog
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=f"No file dialog available: {exc}")
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = list(filedialog.askopenfilenames(initialdir=str(start_dir), filetypes=[("NetCDF", "*.nc")]))
        root.destroy()
    paths = []
    for c in chosen:
        if not c:
            continue
        p = Path(c)
        paths.extend(sorted(str(f) for f in p.rglob("*.nc")) if p.is_dir() else [str(p)])
    return {"paths": [p for p in paths if p.endswith(".nc")]}


@app.post("/api/browse")
def browse_file(payload: BrowsePayload):
    """Open the OS file picker; browsers never expose a picked file's real path."""
    start = Path(payload.start).expanduser() if payload.start else None
    start_dir = start.parent if start and start.parent.is_dir() else Path.cwd()
    if sys.platform == "darwin":
        script = (
            'on run argv\n'
            'POSIX path of (choose file with prompt "Choose an input NetCDF file" '
            'default location POSIX file (item 1 of argv))\n'
            'end run'
        )
        proc = subprocess.run(["osascript", "-e", script, str(start_dir)], capture_output=True, text=True)
        if proc.returncode != 0:
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
    dest = _demo_dest(config_name)
    if dest is None or dest.exists():
        return

    def progress(done, total):
        _DOWNLOADS[config_name] = (done, total)

    progress(0, 0)
    try:
        get_demo_file(_demo_key(config_name), on_progress=progress)
    finally:
        _DOWNLOADS.pop(config_name, None)


# config name -> (bytes done, total or 0 if unknown)
_DOWNLOADS: dict[str, tuple[int, int]] = {}


@app.get("/api/demos/progress")
def demo_progress():
    return {name: {"done": d, "total": t} for name, (d, t) in _DOWNLOADS.items()}


class BuildPayload(BaseModel):
    file_path: str
    choices: dict | None = None
    description: str | None = None


def _template_text() -> str:
    return config_builder.DEFAULT_CONFIG.read_text()


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
    # keep demo paths workspace-relative so the config works on any machine
    file_path = payload.file_path
    if not Path(file_path).is_absolute():
        file_path = str(path.relative_to(WORKSPACE_DIR))
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
        # built by /api/build once the user confirms the decisions; this only names the file
        entry = DEMO_FILES[_demo_key(demo_name)]
        return {
            "name": demo_name,
            "build": {
                "file_path": str((DEMO_DATA_DIR / entry.filename).relative_to(WORKSPACE_DIR)),
                "description": f"A demo pipeline using {entry.display_label} data.",
            },
        }
    if demo_name == DEFAULT_CONFIG_NAME:
        return {"name": DEFAULT_CONFIG_NAME, "yaml_content": _template_text()}
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
    # The one pipeline subprocess. Lines are append-only so a reconnecting client can replay them.

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self.live: str | None = None  # latest progress-bar redraw
        self.live_after = 0
        self.finished = False
        self.returncode: int | None = None
        self.reports: set[str] = set()  # only announced PDFs are served
        self._lock = threading.Lock()
        # processing clock from __PELAGOS_TIME__, so RAM samples share the runtime readout's x-axis
        self._active = 0.0
        self._since: float | None = None
        self._columns = threading.Condition()
        self._columns_done: set[int] = set()
        self._columns_id = 0

    def _sample_mem(self, proc):
        # polled from here so it can't interleave with the run's console output
        try:
            import psutil
            child = psutil.Process(proc.pid)
        except Exception:  # noqa: BLE001
            return
        while proc.poll() is None:
            try:
                rss = child.memory_info().rss / 1024 ** 2
            except Exception:  # noqa: BLE001
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
        # a pipe isn't a terminal, so force colour and tqdm bars (the browser console renders both)
        env["FORCE_COLOR"] = "1"
        env.pop("NO_COLOR", None)
        env["COLUMNS"] = "110"
        # locking can fail (Errno -101) right after another process (the variable check) closed the file
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        for pattern in ("*.png", "*.json", "*.f32", "*_full.npz", "cols_*.bin"):
            for old in FIG_DIR.glob(pattern):
                old.unlink(missing_ok=True)
        _FIGDATA_CACHE.clear()
        self.proc = subprocess.Popen(
            [sys.executable, str(RUN_BOOTSTRAP), str(config_path), str(FIG_DIR)],
            cwd=str(WORKSPACE_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=-1,  # binary BufferedReader: keeps \r and has read1, so bar ticks stream promptly
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
        if text.startswith("__PELAGOS_DATA__ "):  # a reply to columns(), not a log line
            with self._columns:
                self._columns_done.add(int(text.split(" ", 1)[1]))
                self._columns.notify_all()
            return
        if text.startswith("__PELAGOS_REPORT__ "):
            self.reports.add(text.split(" ", 1)[1].split("\t")[0].strip())
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
        with self._lock:
            self.live = text
            self.live_after = len(self.lines)

    def _pump(self):
        assert self.proc is not None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stream = self.proc.stdout
        seg = ""  # chars since the last \r or \n
        pending_cr = False
        while True:
            chunk = stream.read1(4096)  # type: ignore[union-attr]
            if not chunk:
                break
            for ch in decoder.decode(chunk):
                if ch == "\n":
                    # \r\n is a line ending, not a redraw
                    pending_cr = False
                    self._commit(_clean_ansi(seg))
                    seg = ""
                    continue
                if pending_cr:
                    # a lone \r is a progress-bar redraw
                    cleaned = _clean_ansi(seg)
                    if cleaned:
                        self._progress(cleaned)
                    seg = ""
                    pending_cr = False
                if ch == "\r":
                    pending_cr = True
                else:
                    seg += ch
        if seg:
            if pending_cr:
                cleaned = _clean_ansi(seg)
                if cleaned:
                    self._progress(cleaned)
            else:
                self._commit(_clean_ansi(seg))
        self.returncode = self.proc.wait()
        self.proc.stdin.close()
        self.finished = True

    def stop(self):
        if self.is_running() and sys.platform == "win32":
            self.proc.terminate()  # Windows Popen can't send SIGINT (raises ValueError)
        elif self.is_running():
            # SIGINT first so the child runs its cleanup and doesn't leak pool semaphores
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
        # a control line for run_bootstrap.py's pause loop; ignored once the run has exited
        if self.proc is not None and self.proc.stdin is not None and self.is_running():
            try:
                self.proc.stdin.write((command + "\n").encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass


    def columns(self, names: list[str], timeout: float = 60.0) -> bytes | None:
        # packed columns from the paused runner, or None if it didn't answer
        with self._columns:
            self._columns_id += 1
            request_id = self._columns_id
        self.send("data " + json.dumps({"id": request_id, "names": names}))
        with self._columns:
            answered = self._columns.wait_for(
                lambda: request_id in self._columns_done or not self.is_running(), timeout
            )
            self._columns_done.discard(request_id)
        path = FIG_DIR / f"cols_{request_id}.bin"
        if not answered or not path.is_file():
            return None
        blob = path.read_bytes()
        path.unlink(missing_ok=True)
        return blob


_run = _Run()
atexit.register(_run.stop)


class RunPayload(BaseModel):
    yaml_content: str


@app.post("/api/run")
def run_pipeline(payload: RunPayload):
    if _run.is_running():
        raise HTTPException(status_code=409, detail="A pipeline is already running.")
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
    yaml_content: str = ""


@app.post("/api/run/continue")
def continue_run():
    """Resume a paused run."""
    _run.send("continue")
    return {"status": "continued"}


@app.post("/api/run/skip")
def skip_step():
    """Resume a paused run without the paused step's result."""
    _run.send("skip")
    return {"status": "skipped"}


@app.post("/api/run/rerun")
def rerun_step(payload: RerunPayload):
    """Re-run the currently paused step with edited parameters, then re-pause."""
    if payload.yaml_content:
        # a page reload reopens this file, so keep it in step with the edits
        (CONFIG_DIR / "_last_run.yaml").write_text(payload.yaml_content)
    _run.send("rerun " + json.dumps(payload.parameters))
    return {"status": "rerunning"}


@app.get("/api/run/columns")
def run_columns(names: str):
    """Raw columns from the paused run, for the Manual QC views."""
    blob = _run.columns([n for n in names.split(",") if n])
    if blob is None:
        raise HTTPException(status_code=409, detail="The run is not paused, or did not answer.")
    return Response(blob, media_type="application/octet-stream")


@app.get("/api/sigma0")
def sigma0_lines(smin: float, smax: float, tmin: float, tmax: float):
    """Sigma0 contours over a T-S window, for the Manual QC T-S view."""
    # practical salinity and in-situ temperature stand in for SA and CT; close enough to draw
    import contourpy
    import gsw

    s = np.linspace(smin, smax, 120)
    t = np.linspace(tmin, tmax, 120)
    sigma = gsw.sigma0(*np.meshgrid(s, t))
    lo, hi = float(np.nanmin(sigma)), float(np.nanmax(sigma))
    step = next((st for st in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0) if (hi - lo) / st <= 12), 10.0)
    lines = contourpy.contour_generator(s, t, sigma)
    levels = np.arange(np.ceil(lo / step) * step, hi, step)
    return [{"level": round(float(level), 2), "lines": [seg.tolist() for seg in lines.lines(level)]} for level in levels]


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
    """Server-Sent Events stream of the run's log, replayed from the start for each client."""
    def frame(kind: str, text: str) -> str:
        prefix = "" if kind == "line" else f"event: {kind}\n"
        return f"{prefix}data: {text}\n\n"

    def event_gen():
        cursor = 0
        last_live = None
        idle = 0.0
        while True:
            with _run._lock:
                new_lines = _run.lines[cursor:]
                cursor = len(_run.lines)
                live = _run.live
                finished = _run.finished
                returncode = _run.returncode
            for line in new_lines:
                last_live = None
                yield frame("line", line)
            if live is not None and live != last_live:
                last_live = live
                yield frame("progress", live)
            if finished and cursor >= len(_run.lines):
                yield f"event: end\ndata: {returncode}\n\n"
                return
            if not new_lines:
                idle += 0.1
                if idle >= 15.0:  # keep-alive
                    idle = 0.0
                    yield ": keep-alive\n\n"
            else:
                idle = 0.0
            time.sleep(0.1)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


_FIG_FILES = {
    ".png": ("image/png", "Figure"),
    ".json": ("application/json", "Plot spec"),
    ".f32": ("application/octet-stream", "Plot data"),
}


def _fig_file(name: str, suffix: str) -> FileResponse:
    media_type, label = _FIG_FILES[suffix]
    path = FIG_DIR / Path(name).name  # no path traversal
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
    # a plain file gives the browser a Content-Length for its progress bar
    return _fig_file(name, ".f32")


@app.get("/api/run/figpoint/{name}")
def run_figpoint(name: str, panel: int, trace: int, index: int):
    """Exact x/y of one plotted point, for the viewer's click tooltip."""
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
    """A PDF report the current run wrote, inline so the browser previews it."""
    p = Path(path)
    if path not in _run.reports or not p.is_file():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(
        p, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{p.name}"'},
    )


# ================================== Inspect ==================================
def _resolve_inspect_path(file_path):
    if not file_path:
        raise HTTPException(status_code=400, detail="No file_path given.")
    path = Path(file_path)
    if not path.is_absolute():
        path = WORKSPACE_DIR / path
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

    # 'instrument' may be "a, b, c" or a Python-list-style string
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
    """PNG of a variable against TIME (coloured by its QC flags), subsampled for speed."""
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
        # fixed margins so every plot is the same size
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


def serve(port=8791):
    import webbrowser

    import uvicorn

    os.chdir(WORKSPACE_DIR)  # relative config paths resolve here, as in the pipeline run
    url = f"http://localhost:{port}"
    print(f"pelagos_py dashboard -> {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    # one Ctrl+C stops the server, even with open streams
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=2)
