"""Subprocess entry point the dashboard uses to run a pipeline.

Run as::

    python run_bootstrap.py <config_path> <figure_dir>

The dashboard runs the pipeline in a headless subprocess, so a step's
``plt.show()`` diagnostic popup is a silent no-op and the user never sees the
plot. This launcher redirects ``plt.show`` to *save* every open figure into
``figure_dir`` and print a marker line the dashboard picks off the log stream::

    __PELAGOS_FIG__ <filename>\t<caption>\t<spec filename or "">

so the browser can display the plot inline (see run.js). Alongside the PNG it
also tries to write a JSON plot spec (see fig_spec.py) holding the figure's
underlying x/y data, which the browser redraws with WebGL so the plot can be
zoomed and panned. Figures that cannot be serialised faithfully just get an
empty spec field and stay PNG-only. It also forces the Agg
backend and neutralises backend switches, so no step can grab a GUI backend
(e.g. ``matplotlib.use("tkagg")``) that would crash in this displayless process.

This is deliberately dashboard-only glue: it changes nothing in pelagos_py, it
just wraps how the plots are surfaced. Only steps that actually call
``plt.show`` (i.e. ``diagnostics: true``) produce plots here.

A step whose diagnostics are log-only (e.g. Load Data, Export — they print a
summary instead of plotting) draws no figure, so it gets a ``__PELAGOS_LOG__``
marker instead::

    __PELAGOS_LOG__ <step index>\t<step name>\t<QC test or "">\t<base64 text>

so the dashboard can show that text where a plot would otherwise go. See
``_patch_diagnostics_capture``.

A step that raises, with ``on_step_fail: pause`` (the default), pauses the same
way a diagnostics step does rather than being skipped or halting the run, so
the user can fix its parameters and re-run it, or Skip it (its failure plot, if
it draws one, is captured like any other figure)::

    __PELAGOS_FAIL__ <step index>\t<step name>\t<QC test or "">\t<base64 text>

See ``_emit_fail`` and ``pelagos_py.pipeline.resolve_on_step_fail``.
"""

import base64
import contextlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend setup)

plt.switch_backend("Agg")  # initialise Agg now, before the switches below become no-ops
# No display here: steps asking for a GUI backend (matplotlib.use("tkagg")) must stay on Agg
matplotlib.use = lambda *args, **kwargs: None
plt.switch_backend = lambda *args, **kwargs: None

import numpy as np  # noqa: E402
import fig_spec  # noqa: E402  (dashboard-local; this script's directory is on sys.path)
from pelagos_py.pipeline import REPORT_STEP_NAME, SEVERE, STOP, Pipeline, resolve_on_step_fail  # noqa: E402

FIG_DIR = sys.argv[2]
_saved = {"n": 0}
_orig_show = plt.show

# Live memory readout for the dashboard's RAM meter. RSS is this process's
# resident set (the runner and every step share this one process). The transient
# spike a step makes usually happens *inside* run() and is freed before the step
# returns, so sampling only at step boundaries would miss it and mis-attribute
# the peak. A background thread polls RSS a few times a second, tracks the max
# *during* each step, and remembers which step the run's overall peak fell in --
# so the meter can point at the step that actually blew RAM up. ``data`` is the
# xarray dataset's own byte size, separating genuine dataset growth from
# transient per-step overhead.
_mem = {
    "label": "startup",   # step the sampler currently attributes RSS to
    "step_start": 0.0,    # RSS (MB) entering the current step
    "step_peak": 0.0,     # max RSS (MB) seen during the current step
    "run_peak": 0.0,      # max RSS (MB) over the whole run
    "run_peak_label": "", # the step run_peak occurred during
}
_mem_lock = threading.Lock()

# Processing clock for the dashboard's runtime readout. Stops while the run
# sits paused on stdin (review / manual QC), so it reads as time actually
# spent processing. ``__PELAGOS_TIME__ <active s>\t<paused 0/1>\t<epoch s>``:
# the epoch lets the browser tick on from the marker even after a reconnect.
_clock = {"active": 0.0, "since": time.time()}


def _rss_mb():
    """This process's resident set in MB, or None if psutil is unavailable."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:  # noqa: BLE001 - the meter is a bonus, never fatal
        return None


def _mem_sampler():
    """Poll RSS ~2.5x/sec, recording the per-step and whole-run peaks."""
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
    """Attribute subsequent samples to ``label``; record the entry RSS."""
    rss = _rss_mb() or 0.0
    with _mem_lock:
        _mem["label"] = label
        _mem["step_start"] = rss
        _mem["step_peak"] = rss


def _emit_mem(context):
    # __PELAGOS_MEM__ fields: settle RSS, run peak, dataset MB, step label, in-step peak,
    # peak step, step-start RSS, active seconds. No psutil -> no marker, never fatal.
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
            data_mb = f"{nbytes / 1024 ** 2:.1f}"
    except Exception:  # noqa: BLE001 - dataset size is optional detail
        data_mb = ""
    active = _clock["active"] + (time.time() - _clock["since"] if _clock["since"] else 0)
    print(
        f"__PELAGOS_MEM__ {rss_mb:.1f}\t{run_peak:.1f}\t{data_mb}\t{label}"
        f"\t{step_peak:.1f}\t{run_peak_label}\t{step_start:.1f}\t{active:.1f}",
        flush=True,
    )


if _rss_mb() is not None:
    threading.Thread(target=_mem_sampler, daemon=True).start()


def _caption(fig) -> str:
    """Best-effort human label for a figure: its suptitle, else first axes title."""
    try:
        text = ""
        if fig._suptitle and fig._suptitle.get_text():
            text = fig._suptitle.get_text()
        else:
            for ax in fig.axes:
                if ax.get_title():
                    text = ax.get_title()
                    break
        # The __PELAGOS_FIG__ record is one tab-separated line; a multi-line
        # title (e.g. a correction formula on its own line) would otherwise
        # split the record and hide the spec field, forcing PNG-only.
        return " ".join(text.split())
    except Exception:  # noqa: BLE001 - a caption is cosmetic, never fatal
        return ""


def _capture_spec(fig, stem: str):
    """Write the figure's interactive spec, float32 blob and float64 sidecar:
    ``(spec filename, reason)``. Best-effort: anything fig_spec cannot represent
    (or anything unexpected) leaves the dashboard with the PNG it already has."""
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
    except Exception as exc:  # noqa: BLE001 - a bonus feature, never fatal
        return "", f"{type(exc).__name__}"


def _capture_show(*args, **kwargs):
    """Save every open figure to FIG_DIR and announce it, instead of displaying."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        _saved["n"] += 1
        stem = f"fig_{_saved['n']:03d}"
        fname = stem + ".png"
        try:
            fig.savefig(os.path.join(FIG_DIR, fname), dpi=130, bbox_inches="tight")
            spec, reason = _capture_spec(fig, stem)
            # Tab-separated so the dashboard can split the fields apart; a
            # caption may contain spaces but not tabs/newlines.
            print(f"__PELAGOS_FIG__ {fname}\t{_caption(fig)}\t{spec}\t{reason}",
                  flush=True)
        except Exception:  # noqa: BLE001 - a capture failure must never be fatal
            pass
        finally:
            plt.close(fig)


plt.show = _capture_show


class _Tee:
    """A writable stream that mirrors everything to two underlying streams."""

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


# Text captured from a step's diagnostics call when it drew no figure, so it
# can be shown as a "log" in the dashboard instead of a plot. ``None`` means
# "not capturing" (the fast path, for every non-pausable step); a list means
# the current step is pausable and its diagnostics text is being collected.
_diag_capture = {"chunks": None}


def _patch_diagnostics_capture():
    """Make every step's diagnostics call capture its printed text.

    Piggybacks on ``BaseStep._wrap_diagnostics_timing``, which already wraps
    ``generate_diagnostics``/``plot_diagnostics`` per-instance for every step.
    When the wrapped call draws no new figure (a load/export-style step that
    only prints a summary), its stdout is stashed in ``_diag_capture`` so
    ``_emit_diag_log`` can announce it as a ``__PELAGOS_LOG__`` marker once the
    step finishes — the dashboard then shows that text in place of a plot.
    """
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
    # __PELAGOS_FAIL__: same fields as __PELAGOS_LOG__, shown by the dashboard as an error
    text = getattr(exc, "halt_message", None) or f"{type(exc).__name__}: {exc}"
    payload = base64.b64encode(text.encode()).decode()
    print(f"__PELAGOS_FAIL__ {idx}\t{name}\t{test or ''}\t{payload}", flush=True)


def _emit_diag_log(idx, name, test):
    # __PELAGOS_LOG__ (idx, name, test, base64 text) for a step that printed but drew no figure
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
    """Announce a PDF report a step just wrote, for the dashboard's Report tab.

    Prints ``__PELAGOS_REPORT__ <abspath>\t<filename>``. Best-effort: looks under
    the run's ``out_directory`` for the newest ``.pdf`` touched since the report
    step began, so it works for both the Python and Sphinx report steps without
    hardcoding their filename logic. Nothing found -> no marker.
    """
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
    except Exception:  # noqa: BLE001 - surfacing the report is a bonus, never fatal
        pass


def _snapshot(context):
    # Pre-step state for a re-run: steps mutate the dataset in place, so it is deep-copied
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
    # Whether the run pauses after this unit for the user to look at it
    step_diag = bool(step_config.get("diagnostics"))
    if test is None:  # an unsplit QC step only exists when no test is pausable (see _expand)
        return step_diag
    return bool(((_qc_tests(step_config) or {}).get(test) or {}).get("diagnostics", step_diag))


def _expand(step_config):
    # Split a pausable QC step into one Apply QC unit per test, so each can be inspected
    # and re-run on its own; a single test is still split so the pause is keyed by test name
    tests = _qc_tests(step_config)
    if not tests:
        return [(step_config, None)]
    if not any(_pausable(step_config, name) for name in tests):
        return [(step_config, None)]
    units = []
    for name, settings in tests.items():
        sub = dict(step_config)
        sub["parameters"] = dict(step_config["parameters"], qc_settings={name: settings})
        units.append((sub, name))
    return units


def _emit_vars(context):
    # __PELAGOS_VARS__: plottable variables (1-D over N_MEASUREMENTS, not _QC) for the Manual QC axes
    try:
        data = (context or {}).get("data")
        if data is None:
            return
        names = [
            name for name in data.variables
            if data[name].dims == ("N_MEASUREMENTS",) and not name.endswith("_QC")
            and name != "N_MEASUREMENTS"
        ]
        print(f"__PELAGOS_VARS__ {json.dumps(names)}", flush=True)
    except Exception:  # noqa: BLE001 - the axis list is a bonus, never fatal
        pass


def _drop_captures(pipeline, mark):
    """Discard report figures captured since ``mark`` (the previous attempt's)."""
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
    print(f"__PELAGOS_TIME__ {_clock['active']:.3f}\t{int(paused)}\t{now:.3f}", flush=True)


def _read_command():
    # One line on stdin: "continue" or "rerun <json params>"; EOF counts as continue so a
    # dropped channel never hangs the run (Stop is a SIGINT, see app.py)
    line = sys.stdin.readline()
    if not line:
        return ("continue", None)
    line = line.rstrip("\n")
    if line.startswith("rerun "):
        try:
            return ("rerun", json.loads(line[len("rerun "):]))
        except Exception:  # noqa: BLE001 - a malformed command just continues
            return ("continue", None)
    return ("continue", None)


def main():
    config_path = sys.argv[1]
    try:
        _patch_diagnostics_capture()
        _emit_time(paused=False)
        pipeline = Pipeline(config_path=config_path)
        pipeline._diagnose_failures = True  # a failed step's plot goes to the review panel
        _run(pipeline)
    except KeyboardInterrupt:
        # The Stop button sends SIGINT (works while paused on stdin too); exit
        # cleanly instead of dumping a traceback from wherever it landed. The
        # dashboard writes its own "stopped" line.
        sys.exit(130)
    finally:
        _emit_time(paused=True)  # final processing time


def _run(pipeline):
    # Drive execute_step ourselves (run_context does the rest of run()'s setup) so the
    # run can pause after a diagnostics step for the user to inspect, tweak and re-run it.
    with pipeline.run_context():
        context = pipeline._context
        # A QC step becomes several units, one per test; everything else is a
        # single unit. `idx` stays the step's index in the config either way, so
        # figures and the re-run form still line up with the builder card.
        units = [
            (idx, sub, test)
            for idx, step_config in enumerate(pipeline.steps)
            for sub, test in _expand(step_config)
        ]
        for idx, step_config, test in units:
            name = step_config.get("name", "")
            pausable = _pausable(step_config, test)
            # Snapshot the pre-step state only when we might re-run this step.
            # For a split QC step that is the state before *this test*, so a
            # re-run replays one test rather than the whole batch. When there is
            # no snapshot (a non-pausable step that goes on to fail), a re-run
            # falls back to `pre_context` below -- the pre-step reference, not a
            # copy, so a failure that never got to mutate the data still retries
            # cleanly; one that did is a known, accepted gap (see CLAUDE.md).
            snapshot = _snapshot(context) if pausable else None
            pre_context = context
            # Report captures taken so far: a re-run replaces this unit's, so the
            # report shows only the attempt that was carried forward.
            captured_mark = len(getattr(pipeline, "_captured_figures", None) or [])
            label = name + (f"\t{test}" if test else "")
            # Announce the step *before* it runs so the dashboard can attribute
            # the figures it emits. The pipeline's own "Executing:" log line is
            # file-only (extra={"console": False}), so it never reaches here.
            print(f"__PELAGOS_STEP__ {idx}\t{label}", flush=True)
            mem_label = name + (f" · {test}" if test else "")
            _mem_begin(mem_label)
            report_since = time.time()
            _begin_diag_capture(pausable)
            failed = False
            # Whether this unit has ever produced a usable context, across the
            # initial attempt and any re-runs -- so Continue only logs "failed
            # and skipped" when nothing usable exists, not when a later re-run
            # happens to fail after an earlier attempt already succeeded (in
            # that case `context` still holds that earlier good result).
            has_result = False
            try:
                context = pipeline.execute_step(step_config, context)
                has_result = True
            except (RuntimeError, SystemExit) as exc:
                _diag_capture["chunks"] = None
                # Mirror Pipeline.run()'s on_step_fail handling, which this
                # loop otherwise bypasses by driving execute_step() itself.
                fail_mode = resolve_on_step_fail(pipeline.global_parameters.get("on_step_fail"))
                if fail_mode == "stop":
                    pipeline.logger.log(STOP, "Pipeline stopped at step '%s'.", label)
                    sys.exit(1)
                if fail_mode == "skip":
                    # The fatal-error log from execute_step() already carries the
                    # detail; this just marks the step skipped.
                    pipeline.logger.log(SEVERE, "Step '%s' failed and was skipped.", label)
                    continue
                # "pause": so the user can fix the step's parameters and re-run
                # it, or accept the skip.
                failed = True
                _emit_fail(idx, name, test, exc)
            if not failed:
                _emit_diag_log(idx, name, test)
                _emit_mem(context)
                # A report step drops a PDF under out_directory; surface it so the
                # dashboard can offer to open it once the run reaches it.
                if name == REPORT_STEP_NAME:
                    _emit_report(context, report_since - 2)
            if not pausable and not failed:
                continue
            _emit_vars(context)
            while True:
                print(f"__PELAGOS_PAUSE__ {idx}\t{label}", flush=True)
                _emit_time(paused=True)
                action, params = _read_command()
                _emit_time(paused=False)
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
                    if params is not None:
                        # A split QC unit only ever re-runs its own test, even if
                        # the browser sent the whole step's settings.
                        if test is not None and isinstance(params.get("qc_settings"), dict):
                            params = dict(
                                params,
                                qc_settings={
                                    test: params["qc_settings"].get(
                                        test, (_qc_tests(step_config) or {}).get(test, {})
                                    )
                                },
                            )
                        rerun_config["parameters"] = params
                    # The other half of the round trip: what the pipeline was
                    # actually handed, next to what the browser said it sent.
                    print(
                        f"Re-running with parameters: {rerun_config.get('parameters')}",
                        flush=True,
                    )
                    # Fresh copy each re-run so repeated re-runs all start clean.
                    _mem_begin(mem_label)
                    _begin_diag_capture(True)
                    _drop_captures(pipeline, captured_mark)
                    retry_context = _snapshot(snapshot) if snapshot is not None else pre_context
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
