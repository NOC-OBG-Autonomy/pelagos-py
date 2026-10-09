"""Subprocess the dashboard runs a pipeline in: ``python run_bootstrap.py <config> <figure_dir>``.

Forces the Agg backend, saves each ``plt.show()`` figure (plus a plot spec) to
``figure_dir``, pauses after diagnostics steps for review, and reports progress
to the dashboard as ``__PELAGOS_*__`` lines on stdout (see the README).
"""

import base64
import contextlib
import io
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

# The imports below take seconds; let Stop during them exit without a traceback.
_START = time.time()
print(f"__PELAGOS_TIME__ 0.000\t0\t{_START:.3f}", flush=True)
signal.signal(signal.SIGINT, lambda *args: sys.exit(130))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend setup)

plt.switch_backend("Agg")  # initialise Agg now, before the switches below become no-ops
# No display here: steps asking for a GUI backend (matplotlib.use("tkagg")) must stay on Agg
matplotlib.use = lambda *args, **kwargs: None
plt.switch_backend = lambda *args, **kwargs: None

import numpy as np  # noqa: E402
from pelagos_py.dashboard import fig_spec  # noqa: E402
from pelagos_py.pipeline import (
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

# RAM meter: a step's memory spike is usually freed before it returns, so a thread polls RSS during it.
_mem = {
    "label": "startup",
    "step_start": 0.0,
    "step_peak": 0.0,
    "run_peak": 0.0,
    "run_peak_label": "",
}
_mem_lock = threading.Lock()

# Processing time, stopped while paused on stdin
_clock = {"active": 0.0, "since": _START}


def _rss_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:  # noqa: BLE001
        return None


def _mem_sampler():
    while True:
        rss = _rss_mb()
        if rss is not None:
            with _mem_lock:
                if rss > _mem["step_peak"]:
                    _mem["step_peak"] = rss
                if rss > _mem["run_peak"]:
                    _mem["run_peak"] = rss
                    _mem["run_peak_label"] = _mem["label"]
        time.sleep(0.4)


def _mem_begin(label):
    rss = _rss_mb() or 0.0
    with _mem_lock:
        _mem["label"] = label
        _mem["step_start"] = rss
        _mem["step_peak"] = rss


def _emit_mem(context):
    rss_mb = _rss_mb()
    if rss_mb is None:
        return
    with _mem_lock:
        step_peak = max(_mem["step_peak"], rss_mb)
        step_start = _mem["step_start"]
        if rss_mb > _mem["run_peak"]:
            _mem["run_peak"] = rss_mb
            _mem["run_peak_label"] = _mem["label"]
        run_peak = _mem["run_peak"]
        run_peak_label = _mem["run_peak_label"]
        label = _mem["label"]
    data_mb = ""
    try:
        data = (context or {}).get("data")
        nbytes = getattr(data, "nbytes", None)
        if nbytes is not None:
            data_mb = f"{nbytes / 1024**2:.1f}"
    except Exception:  # noqa: BLE001
        data_mb = ""
    active = _clock["active"] + (
        time.time() - _clock["since"] if _clock["since"] else 0
    )
    print(
        f"__PELAGOS_MEM__ {rss_mb:.1f}\t{run_peak:.1f}\t{data_mb}\t{label}"
        f"\t{step_peak:.1f}\t{run_peak_label}\t{step_start:.1f}\t{active:.1f}",
        flush=True,
    )


if _rss_mb() is not None:
    threading.Thread(target=_mem_sampler, daemon=True).start()


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
    except Exception:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        pass


def _snapshot(context):
    # steps mutate the dataset in place, so a re-run needs a deep copy
    if context is None:
        return None
    snap = dict(context)
    if snap.get("data") is not None:
        snap["data"] = snap["data"].copy(deep=True)
    return snap


def _qc_tests(step_config):
    settings = (step_config.get("parameters") or {}).get("qc_settings")
    return settings if isinstance(settings, dict) and settings else None


def _pausable(step_config, test):
    step_diag = bool(step_config.get("diagnostics"))
    if (
        test is None
    ):  # an unsplit QC step only exists when no test is pausable (see _expand)
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


def _has_data(values):
    # all-NaN/NaT or all-zero columns go last in Manual QC's colour picker
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
    except Exception:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
        print(f"Could not send columns to the dashboard: {exc}", flush=True)
    print(f"__PELAGOS_DATA__ {request_id}", flush=True)


def _drop_captures(pipeline, mark):
    figs = getattr(pipeline, "_captured_figures", None)
    if not figs:
        return
    for entry in figs[mark:]:
        for path in entry.get("images", []):
            try:
                os.remove(path)
            except OSError:
                pass
    del figs[mark:]


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
            except Exception:  # noqa: BLE001
                return ("continue", None)
    return ("continue", None)


def main():
    config_path = sys.argv[1]
    try:
        _patch_diagnostics_capture()
        _emit_time(paused=False)
        pipeline = Pipeline(config_path=config_path)
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
        for idx, step_config, test in units:
            name = step_config.get("name", "")
            pausable = _pausable(step_config, test)
            snapshot = _snapshot(context) if pausable else None
            pre_context = context
            # a re-run replaces this unit's report figures
            captured_mark = len(getattr(pipeline, "_captured_figures", None) or [])
            label = name + (f"\t{test}" if test else "")
            # the pipeline's own "Executing:" line is file-only, so announce the step here
            print(f"__PELAGOS_STEP__ {idx}\t{label}", flush=True)
            mem_label = name + (f" · {test}" if test else "")
            _mem_begin(mem_label)
            report_since = time.time()
            _begin_diag_capture(pausable)
            failed = False
            # True once any attempt succeeded, so a later failed re-run still keeps that result
            has_result = False
            try:
                context = pipeline.execute_step(step_config, context)
                has_result = True
            except (RuntimeError, SystemExit) as exc:
                _diag_capture["chunks"] = None
                # same on_step_fail handling as Pipeline.run()
                fail_mode = resolve_on_step_fail(
                    pipeline.global_parameters.get("on_step_fail")
                )
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
                _emit_mem(context)
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
                    context = snapshot if snapshot is not None else pre_context
                    pipeline.logger.log(SEVERE, "Step '%s' was skipped.", label)
                    break
                if action == "continue":
                    if failed and not has_result:
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
                    _mem_begin(mem_label)
                    _begin_diag_capture(True)
                    _drop_captures(pipeline, captured_mark)
                    if snapshot is None:
                        # keep pre_context clean so Skip still means "before this step"
                        snapshot = _snapshot(pre_context)
                    retry_context = (
                        _snapshot(snapshot) if snapshot is not None else pre_context
                    )
                    try:
                        context = pipeline.execute_step(rerun_config, retry_context)
                    except (RuntimeError, SystemExit) as exc:
                        _diag_capture["chunks"] = None
                        failed = True
                        _emit_fail(idx, name, test, exc)
                        continue
                    failed = False
                    has_result = True
                    _emit_diag_log(idx, name, test)
                    _emit_mem(context)
        pipeline._context = context


if __name__ == "__main__":
    main()
