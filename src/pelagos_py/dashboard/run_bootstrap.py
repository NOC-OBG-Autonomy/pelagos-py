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
"""Subprocess the dashboard runs a pipeline in: ``python run_bootstrap.py <config> <figure_dir>``.

Forces the Agg backend, saves each ``plt.show()`` figure (plus a plot spec) to
``figure_dir``, pauses after diagnostics steps for review, and reports progress
to the dashboard as ``__PELAGOS_*__`` lines on stdout (see the README).
"""

import base64
import contextlib
import copy
import io
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# The imports below take seconds; let Stop during them exit without a traceback.
_START = time.time()
print(f"__PELAGOS_TIME__ 0.000\t0\t{_START:.3f}", flush=True)
signal.signal(signal.SIGINT, lambda *args: sys.exit(130))

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend setup)

plt.switch_backend("Agg")  # initialise Agg now, before the switches below become no-ops
# No display here: steps asking for a GUI backend (matplotlib.use("tkagg")) must stay on Agg
matplotlib.use = lambda *args, **kwargs: None
plt.switch_backend = lambda *args, **kwargs: None

import numpy as np  # noqa: E402
from pelagos_py.dashboard import fig_spec  # noqa: E402
from pelagos_py.utils.console import ConsoleFormatter  # noqa: E402
import pelagos_py.pipeline as pipeline_module  # noqa: E402
from pelagos_py.pipeline import (  # noqa: E402
    REPORT_STEP_NAME,
    SEVERE,
    STOP,
    Pipeline,
    resolve_on_step_fail,
)  # noqa: E402

signal.signal(
    signal.SIGINT, signal.default_int_handler
)  # imports done: Stop is a KeyboardInterrupt again

FIG_DIR = sys.argv[2]
_saved = {"n": 0}
_orig_show = plt.show

# Processing time, stopped while paused on stdin
_clock = {"active": 0.0, "since": _START}


# RAM itself is sampled by the dashboard server; this adds the dataset size after each step
def _emit_data_size(context):
    nbytes = getattr((context or {}).get("data"), "nbytes", None)
    if nbytes is not None:
        print(f"__PELAGOS_MEM__ {nbytes / 1e6:.1f}", flush=True)


def _caption(fig) -> str:
    # suptitle, else the first axes title
    try:
        text = ""
        if fig._suptitle and fig._suptitle.get_text():
            text = fig._suptitle.get_text()
        else:
            for ax in fig.axes:
                if ax.get_title():
                    text = ax.get_title()
                    break
        # the marker is one line, so a multi-line title must be flattened
        return " ".join(text.split())
    except Exception:
        return ""


def _capture_spec(fig, stem: str):
    # Returns (spec filename, reason); on any failure the figure stays PNG-only
    try:
        spec, reason, blob, full = fig_spec.serialise(fig)
        if spec is None:
            return "", reason
        name = stem + ".json"
        with open(os.path.join(FIG_DIR, name), "w") as handle:
            json.dump(spec, handle)
        with open(os.path.join(FIG_DIR, stem + ".f32"), "wb") as handle:
            handle.write(blob)
        np.savez(os.path.join(FIG_DIR, stem + "_full.npz"), **full)
        return name, ""
    except Exception as exc:
        return "", f"{type(exc).__name__}"


def _capture_show(*args, **kwargs):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        _saved["n"] += 1
        stem = f"fig_{_saved['n']:03d}"
        fname = stem + ".png"
        try:
            fig.savefig(os.path.join(FIG_DIR, fname), dpi=130, bbox_inches="tight")
            spec, reason = _capture_spec(fig, stem)
            print(
                f"__PELAGOS_FIG__ {fname}\t{_caption(fig)}\t{spec}\t{reason}",
                flush=True,
            )
        except Exception:
            pass
        finally:
            plt.close(fig)


plt.show = _capture_show


class _Tee:
    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, s):
        self._primary.write(s)
        self._secondary.write(s)
        return len(s)

    def flush(self):
        self._primary.flush()

    def __getattr__(self, name):
        return getattr(self._primary, name)


# Printed diagnostics of a pausable step that drew no figure; None when not capturing
_diag_capture = {"chunks": None}


def _patch_diagnostics_capture():
    # Hooks BaseStep's per-instance diagnostics wrapper so a step that only prints shows as a log
    from pelagos_py.steps.base_step import BaseStep

    orig_wrap = BaseStep._wrap_diagnostics_timing

    def wrap_with_capture(self):
        orig_wrap(self)
        for attr in ("generate_diagnostics", "plot_diagnostics"):
            method = getattr(self, attr, None)
            if not callable(method):
                continue

            def captured(*args, _method=method, **kwargs):
                if _diag_capture["chunks"] is None:
                    return _method(*args, **kwargs)
                fig_before = _saved["n"]
                buf = io.StringIO()
                with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
                    result = _method(*args, **kwargs)
                if _saved["n"] == fig_before and buf.getvalue().strip():
                    _diag_capture["chunks"].append(buf.getvalue())
                return result

            setattr(self, attr, captured)

    BaseStep._wrap_diagnostics_timing = wrap_with_capture


def _begin_diag_capture(pausable):
    _diag_capture["chunks"] = [] if pausable else None


def _emit_fail(idx, name, test, exc):
    text = getattr(exc, "halt_message", None) or f"{type(exc).__name__}: {exc}"
    payload = base64.b64encode(text.encode()).decode()
    print(f"__PELAGOS_FAIL__ {idx}\t{name}\t{test or ''}\t{payload}", flush=True)


def _emit_diag_log(idx, name, test):
    chunks = _diag_capture["chunks"]
    _diag_capture["chunks"] = None
    if not chunks:
        return
    text = "\n".join(chunks).strip()
    if not text:
        return
    payload = base64.b64encode(text.encode()).decode()
    print(f"__PELAGOS_LOG__ {idx}\t{name}\t{test or ''}\t{payload}", flush=True)


def _emit_report(context, since):
    # Newest PDF under out_directory since the step began, so any report step works
    try:
        gp = (context or {}).get("global_parameters") or {}
        base = Path(gp.get("out_directory") or "./")
        if not base.is_absolute():
            base = Path.cwd() / base
        newest, newest_mtime = None, since
        for pdf in base.rglob("*.pdf"):
            try:
                mtime = pdf.stat().st_mtime
            except OSError:
                continue
            if mtime >= newest_mtime:
                newest, newest_mtime = pdf, mtime
        if newest is not None:
            print(f"__PELAGOS_REPORT__ {newest.resolve()}\t{newest.name}", flush=True)
    except Exception:
        pass


def _emit_log_file(pipeline):
    # the Outputs list only offers to delete files a run wrote
    for handler in pipeline.logger.handlers:
        if isinstance(handler, logging.FileHandler):
            print(f"__PELAGOS_OUTPUT__ {handler.baseFilename}", flush=True)


def _snapshot(context):
    # steps mutate the dataset in place, so a re-run needs a deep copy
    if context is None:
        return None
    snap = dict(context)
    if snap.get("data") is not None:
        snap["data"] = snap["data"].copy(deep=True)
    # Apply QC appends to these lists in place
    if "qc_history" in snap:
        snap["qc_history"] = copy.deepcopy(snap["qc_history"])
    return snap


def _qc_tests(step_config):
    settings = (step_config.get("parameters") or {}).get("qc_settings")
    return settings if isinstance(settings, dict) and settings else None


def _pausable(step_config, test):
    step_diag = bool(step_config.get("diagnostics"))
    # an unsplit QC step only exists when no test is pausable (see _expand)
    if test is None:
        return step_diag
    return bool(
        ((_qc_tests(step_config) or {}).get(test) or {}).get("diagnostics", step_diag)
    )


def _expand(step_config):
    # One unit per test of a pausable QC step, so each test pauses and re-runs on its own
    tests = _qc_tests(step_config)
    if not tests:
        return [(step_config, None)]
    if not any(_pausable(step_config, name) for name in tests):
        return [(step_config, None)]
    units = []
    for name, settings in tests.items():
        sub = dict(step_config)
        sub["parameters"] = dict(
            step_config["parameters"], qc_settings={name: settings}
        )
        units.append((sub, name))
    return units


def _qc_flags(context):
    data = context["data"]
    return data[[name for name in data.data_vars if name.endswith("_QC")]].copy(
        deep=True
    )


def _unit_context(context, test, flags_before):
    # Each test of a split Apply QC reads the flags from before the step, as in one unsplit step
    if test is None:
        return context
    return dict(context, qc_flags_before=flags_before)


def _accept_parameters(pipeline, idx, test, parameters):
    # later steps (e.g. the report) read the run's parameters from pipeline.steps
    step = pipeline.steps[idx]
    if test is not None:
        qc_settings = dict(_qc_tests(step))
        qc_settings[test] = parameters["qc_settings"][test]
        parameters = dict(step["parameters"], qc_settings=qc_settings)
    pipeline.steps[idx] = dict(step, parameters=parameters)


def _has_data(values):
    # all-NaN/NaT or all-zero columns go last in Visual QC's colour picker
    if np.issubdtype(values.dtype, np.datetime64):
        return bool((~np.isnat(values)).any())
    if not np.issubdtype(values.dtype, np.number):
        return True
    return bool((np.isfinite(values) & (values != 0)).any())


def _emit_vars(context):
    try:
        data = (context or {}).get("data")
        if data is None:
            return
        names = [
            name
            for name in data.variables
            if data[name].dims == ("N_MEASUREMENTS",)
            and not name.endswith("_QC")
            and name != "N_MEASUREMENTS"
        ]
        empty = [name for name in names if not _has_data(data[name].values)]
        print(
            f"__PELAGOS_VARS__ {json.dumps({'names': names, 'empty': empty})}",
            flush=True,
        )
    except Exception:
        pass


def _emit_columns(context, snapshot, request):
    # _QC columns are the flags from before this step, so the browser can replay the boxes
    request_id = int(request.get("id", 0))
    try:
        from pelagos_py.utils import palettes
        from pelagos_py.utils.fig_spec import categories

        data = context["data"]
        before = (snapshot or context)["data"]
        header, arrays = [], []
        for name in request.get("names", []):
            base = name[:-3] if name.endswith("_QC") else None
            if base and name in before:
                values = before[name].fillna(9).values.astype("<f4")
            elif base and base in data:
                values = np.where(
                    np.isfinite(data[base].values.astype(float)), 0, 9
                ).astype("<f4")
            elif name in data.variables and data[name].dims == ("N_MEASUREMENTS",):
                values = data[name].values
            else:
                continue
            entry = {"name": name, "dtype": "f4", "n": len(values)}
            if np.issubdtype(values.dtype, np.datetime64):
                values = values.astype("datetime64[ms]").astype("<f8")
                entry["dtype"] = "f8"
            else:
                values = values.astype("<f4")
                cmap = palettes.cmap_for_variable(name, default=plt.get_cmap("viridis"))
                entry["stops"] = [
                    matplotlib.colors.to_hex(cmap(t)) for t in np.linspace(0, 1, 32)
                ]
                if name in data.variables:
                    entry["categories"] = categories(data[name])
            header.append(entry)
            arrays.append(values)
        with open(os.path.join(FIG_DIR, f"cols_{request_id}.bin"), "wb") as handle:
            handle.write(fig_spec._pack({"columns": header}, arrays))
    except Exception as exc:
        print(f"Could not send columns to the dashboard: {exc}", flush=True)
    print(f"__PELAGOS_DATA__ {request_id}", flush=True)


def _drop_captures(pipeline, start, end=None):
    figs = getattr(pipeline, "_captured_figures", None)
    if not figs:
        return
    for entry in figs[start:end]:
        for path in entry.get("images", []):
            try:
                os.remove(path)
            except OSError:
                pass
    del figs[start:end]


def _capture_count(pipeline):
    return len(getattr(pipeline, "_captured_figures", None) or [])


def _emit_time(paused):
    now = time.time()
    if _clock["since"] is not None:
        _clock["active"] += now - _clock["since"]
    _clock["since"] = None if paused else now
    print(
        f"__PELAGOS_TIME__ {_clock['active']:.3f}\t{int(paused)}\t{now:.3f}", flush=True
    )


def _read_command():
    # "continue", "skip", "rerun <json>" or "data <json>"; EOF continues so the run never hangs
    line = sys.stdin.readline()
    if not line:
        return ("continue", None)
    line = line.rstrip("\n")
    if line == "skip":
        return ("skip", None)
    for action in ("rerun", "data"):
        if line.startswith(action + " "):
            try:
                return (action, json.loads(line[len(action) + 1 :]))
            except Exception:
                return ("continue", None)
    return ("continue", None)


class _LevelFormatter(ConsoleFormatter):
    # warning+ lines carry their level, so the browser tags them instead of reading colour codes
    def __init__(self):
        super().__init__()
        self._use_color = False

    def format(self, record):
        line = super().format(record)
        if record.levelno < logging.WARNING or getattr(record, "raw", False):
            return line
        return f"__PELAGOS_LEVEL__ {record.levelname}\t{line}"


def _send_levels():
    for name in list(logging.Logger.manager.loggerDict):
        for handler in logging.getLogger(name).handlers:
            if isinstance(handler.formatter, ConsoleFormatter):
                handler.setFormatter(_LevelFormatter())
    # Pipeline() makes a fresh console handler, and logs while doing so
    make_handler = pipeline_module.make_console_handler

    def make_level_handler(*args, **kwargs):
        handler = make_handler(*args, **kwargs)
        handler.setFormatter(_LevelFormatter())
        return handler

    pipeline_module.make_console_handler = make_level_handler


def main():
    config_path = sys.argv[1]
    try:
        _patch_diagnostics_capture()
        _emit_time(paused=False)
        _send_levels()
        pipeline = Pipeline(config_path=config_path)
        _emit_log_file(pipeline)
        pipeline._diagnose_failures = True
        _run(pipeline)
    except KeyboardInterrupt:
        # Stop sends SIGINT; the dashboard logs its own "stopped" line
        sys.exit(130)
    finally:
        _emit_time(paused=True)


def _run(pipeline):
    # Drives execute_step itself so the run can pause after a step for review and re-run
    with pipeline.run_context():
        context = pipeline._context
        # idx stays the config index for split QC units, to match the builder card
        units = [
            (idx, sub, test)
            for idx, step_config in enumerate(pipeline.steps)
            for sub, test in _expand(step_config)
        ]
        fail_mode = resolve_on_step_fail(pipeline.global_parameters.get("on_step_fail"))
        flags_idx, flags_before = None, None
        for idx, step_config, test in units:
            name = step_config.get("name", "")
            if test is not None and idx != flags_idx:
                flags_idx, flags_before = idx, _qc_flags(context)
            pausable = _pausable(step_config, test)
            # a failed step may have mutated context in place, so keep a clean copy
            can_pause = pausable or fail_mode == "pause"
            snapshot = _snapshot(context) if can_pause else None
            captured_mark = _capture_count(pipeline)
            label = name + (f"\t{test}" if test else "")
            # the pipeline's own "Executing:" line is file-only, so announce the step here
            print(f"__PELAGOS_STEP__ {idx}\t{label}", flush=True)
            report_since = time.time()
            _begin_diag_capture(pausable)
            failed = False
            # True once any attempt succeeded, so a later failed re-run still keeps that result
            has_result = False
            try:
                context = pipeline.execute_step(
                    step_config, _unit_context(context, test, flags_before)
                )
                has_result = True
            except (RuntimeError, ValueError, SystemExit) as exc:
                _diag_capture["chunks"] = None
                # same on_step_fail handling as Pipeline.run()
                if fail_mode == "stop":
                    pipeline.logger.log(STOP, "Pipeline stopped at step '%s'.", label)
                    sys.exit(1)
                if fail_mode == "skip":
                    pipeline.logger.log(
                        SEVERE, "Step '%s' failed and was skipped.", label
                    )
                    continue
                failed = True
                _emit_fail(idx, name, test, exc)
            if not failed:
                _emit_diag_log(idx, name, test)
                _emit_data_size(context)
                if name == REPORT_STEP_NAME:
                    _emit_report(context, report_since - 2)
            if not pausable and not failed:
                continue
            _emit_vars(context)
            while True:
                print(f"__PELAGOS_PAUSE__ {idx}\t{label}", flush=True)
                _emit_time(paused=True)
                action, params = _read_command()
                while action == "data":
                    _emit_columns(context, snapshot, params)
                    action, params = _read_command()
                _emit_time(paused=False)
                if action == "skip":
                    context = snapshot
                    _drop_captures(pipeline, captured_mark)
                    pipeline.logger.log(SEVERE, "Step '%s' was skipped.", label)
                    break
                if action == "continue":
                    if failed and not has_result:
                        context = snapshot
                        _drop_captures(pipeline, captured_mark)
                        pipeline.logger.log(
                            SEVERE, "Step '%s' failed and was skipped.", label
                        )
                    break
                if action == "rerun":
                    print(f"__PELAGOS_RERUN__ {idx}", flush=True)
                    print(f"__PELAGOS_STEP__ {idx}\t{label}", flush=True)
                    rerun_config = dict(step_config)
                    # a failed step can pause with diagnostics off; plot its re-run anyway
                    rerun_config["diagnostics"] = step_config.get("diagnostics") or True
                    if params is not None:
                        # a split QC unit only re-runs its own test
                        if test is not None and isinstance(
                            params.get("qc_settings"), dict
                        ):
                            params = dict(
                                params,
                                qc_settings={
                                    test: params["qc_settings"].get(
                                        test,
                                        (_qc_tests(step_config) or {}).get(test, {}),
                                    )
                                },
                            )
                        rerun_config["parameters"] = params
                    print(
                        f"Re-running with parameters: {rerun_config.get('parameters')}",
                        flush=True,
                    )
                    _begin_diag_capture(True)
                    retry_mark = _capture_count(pipeline)
                    retry_context = _unit_context(
                        _snapshot(snapshot), test, flags_before
                    )
                    try:
                        context = pipeline.execute_step(rerun_config, retry_context)
                    except (RuntimeError, ValueError, SystemExit) as exc:
                        _diag_capture["chunks"] = None
                        _drop_captures(pipeline, retry_mark)
                        failed = True
                        _emit_fail(idx, name, test, exc)
                        continue
                    # the re-run's figures replace the earlier result's
                    _drop_captures(pipeline, captured_mark, retry_mark)
                    if params is not None:
                        step_config = dict(step_config, parameters=params)
                        _accept_parameters(pipeline, idx, test, params)
                    failed = False
                    has_result = True
                    _emit_diag_log(idx, name, test)
                    _emit_data_size(context)
        pipeline._context = context


if __name__ == "__main__":
    main()
