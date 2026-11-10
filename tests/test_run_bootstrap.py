import contextlib
import importlib
import io
import logging
import signal
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def bootstrap(tmp_path_factory):
    # importing run_bootstrap patches matplotlib, SIGINT and plt.show for the whole process
    saved = (matplotlib.use, plt.switch_backend, plt.show, sys.argv)
    saved_sigint = signal.getsignal(signal.SIGINT)
    sys.argv = ["run_bootstrap.py", "unused.yaml", str(tmp_path_factory.mktemp("figs"))]
    try:
        module = importlib.import_module("pelagos_py.dashboard.run_bootstrap")
    finally:
        matplotlib.use, plt.switch_backend, plt.show, sys.argv = saved
        signal.signal(signal.SIGINT, saved_sigint)
    return module


class FakePipeline:
    def __init__(self, steps, behaviour, on_step_fail="pause"):
        self.steps = steps
        self.global_parameters = {"on_step_fail": on_step_fail}
        self.logger = logging.getLogger("test_run_bootstrap")
        self._context = {"data": xr.Dataset({"X": ("N_MEASUREMENTS", [1.0, 2.0])})}
        self.behaviour = behaviour
        self.seen = []

    @contextlib.contextmanager
    def run_context(self):
        yield False

    def execute_step(self, step_config, context):
        name = step_config["name"]
        self.seen.append((name, context["data"]["X"].values.tolist()))
        return self.behaviour[name](step_config, context)


def corrupt_then_fail(step_config, context):
    context["data"]["X"].values[:] += 100
    raise RuntimeError("step failed")


def add_one(step_config, context):
    context["data"]["X"] = context["data"]["X"] + 1
    return context


def fail_once_then_add_one():
    calls = {"n": 0}

    def behaviour(step_config, context):
        calls["n"] += 1
        if calls["n"] == 1:
            return corrupt_then_fail(step_config, context)
        return add_one(step_config, context)

    return behaviour


def run_with_commands(bootstrap, monkeypatch, pipeline, commands):
    queue = list(commands)
    monkeypatch.setattr(bootstrap, "_read_command", lambda: queue.pop(0))
    bootstrap._run(pipeline)
    assert queue == []
    return pipeline


def two_step_pipeline(first_behaviour):
    return FakePipeline(
        [{"name": "Mutate"}, {"name": "Record"}],
        {"Mutate": first_behaviour, "Record": lambda config, context: context},
    )


@pytest.mark.parametrize(
    "command", [("skip", None), ("continue", None)], ids=["skip", "continue"]
)
def test_failed_step_is_dropped_with_clean_data(bootstrap, monkeypatch, command):
    pipeline = run_with_commands(
        bootstrap, monkeypatch, two_step_pipeline(corrupt_then_fail), [command]
    )
    assert pipeline.seen[-1] == ("Record", [1.0, 2.0])


def test_rerun_after_failure_starts_from_clean_data(bootstrap, monkeypatch):
    pipeline = run_with_commands(
        bootstrap,
        monkeypatch,
        two_step_pipeline(fail_once_then_add_one()),
        [("rerun", None), ("continue", None)],
    )
    assert pipeline.seen == [
        ("Mutate", [1.0, 2.0]),
        ("Mutate", [1.0, 2.0]),
        ("Record", [2.0, 3.0]),
    ]


def test_second_rerun_also_starts_from_clean_data(bootstrap, monkeypatch):
    pipeline = run_with_commands(
        bootstrap,
        monkeypatch,
        two_step_pipeline(corrupt_then_fail),
        [("rerun", None), ("rerun", None), ("continue", None)],
    )
    assert [values for _, values in pipeline.seen] == [[1.0, 2.0]] * 4


def test_skip_after_successful_rerun_restores_pre_step_data(bootstrap, monkeypatch):
    pipeline = run_with_commands(
        bootstrap,
        monkeypatch,
        two_step_pipeline(fail_once_then_add_one()),
        [("rerun", None), ("skip", None)],
    )
    assert pipeline.seen[-1] == ("Record", [1.0, 2.0])


def test_rerun_passes_edited_parameters(bootstrap, monkeypatch):
    configs = []

    def record_config(step_config, context):
        configs.append(step_config)
        return context

    pipeline = FakePipeline(
        [{"name": "Tune", "diagnostics": True, "parameters": {"a": 1}}],
        {"Tune": record_config},
    )
    run_with_commands(
        bootstrap, monkeypatch, pipeline, [("rerun", {"a": 2}), ("continue", None)]
    )
    assert [config["parameters"] for config in configs] == [{"a": 1}, {"a": 2}]


def test_rerun_with_rejected_parameters_pauses_again(bootstrap, monkeypatch):
    def reject_bad_parameters(step_config, context):
        if step_config.get("parameters") == {"a": "bad"}:
            raise ValueError("bad parameter")
        return add_one(step_config, context)

    pipeline = FakePipeline(
        [{"name": "Tune", "diagnostics": True}, {"name": "Record"}],
        {"Tune": reject_bad_parameters, "Record": lambda config, context: context},
    )
    run_with_commands(
        bootstrap, monkeypatch, pipeline, [("rerun", {"a": "bad"}), ("continue", None)]
    )
    assert pipeline.seen[-1] == ("Record", [2.0, 3.0])


def test_skip_mode_moves_on_without_pausing(bootstrap, monkeypatch):
    pipeline = FakePipeline(
        [{"name": "Mutate"}, {"name": "Record"}],
        {"Mutate": corrupt_then_fail, "Record": lambda config, context: context},
        on_step_fail="skip",
    )
    run_with_commands(bootstrap, monkeypatch, pipeline, [])
    assert [name for name, _ in pipeline.seen] == ["Mutate", "Record"]


def test_snapshot_is_a_deep_copy(bootstrap):
    context = {"data": xr.Dataset({"X": ("N_MEASUREMENTS", [1.0])}), "other": 1}
    snapshot = bootstrap._snapshot(context)
    context["data"]["X"].values[:] = 5
    assert snapshot["data"]["X"].values.tolist() == [1.0]
    assert snapshot["other"] == 1
    assert bootstrap._snapshot(None) is None


@pytest.mark.parametrize(
    "line, expected",
    [
        ("", ("continue", None)),
        ("continue\n", ("continue", None)),
        ("skip\n", ("skip", None)),
        ('rerun {"a": 1}\n', ("rerun", {"a": 1})),
        ('data {"id": 3}\n', ("data", {"id": 3})),
        ("rerun {not json\n", ("continue", None)),
        ("nonsense\n", ("continue", None)),
    ],
)
def test_read_command(bootstrap, monkeypatch, line, expected):
    monkeypatch.setattr(sys, "stdin", io.StringIO(line))
    assert bootstrap._read_command() == expected


def test_expand_splits_qc_step_with_a_pausable_test(bootstrap):
    step = {
        "name": "Apply QC",
        "parameters": {
            "qc_settings": {"range qc": {"diagnostics": True}, "spike qc": {}}
        },
    }
    units = bootstrap._expand(step)
    assert [test for _, test in units] == ["range qc", "spike qc"]
    assert units[1][0]["parameters"]["qc_settings"] == {"spike qc": {}}
    assert [bootstrap._pausable(config, test) for config, test in units] == [
        True,
        False,
    ]


def test_expand_keeps_qc_step_whole_when_nothing_pauses(bootstrap):
    step = {"name": "Apply QC", "parameters": {"qc_settings": {"range qc": {}}}}
    assert bootstrap._expand(step) == [(step, None)]


def test_has_data(bootstrap):
    assert bootstrap._has_data(np.array([np.nan, 1.0]))
    assert not bootstrap._has_data(np.array([np.nan, 0.0]))
    assert not bootstrap._has_data(np.array(["NaT"], dtype="datetime64[ns]"))


def test_level_formatter_tags_warnings_but_not_info(bootstrap):
    formatter = bootstrap._LevelFormatter()
    logger = logging.getLogger("pelagos_py.pipeline.step.salinity_adjustment")

    def fmt(level, msg):
        return formatter.format(
            logger.makeRecord(logger.name, level, "f", 1, msg, None, None)
        )

    assert fmt(logging.INFO, "[Salinity Adjustment] lag 0.3 s").endswith(
        "Salinity Adjustment   lag 0.3 s"
    )
    level, line = fmt(logging.WARNING, "[Salinity Adjustment] no CNDC").split("\t")
    assert level == "__PELAGOS_LEVEL__ WARNING"
    assert line.endswith("Salinity Adjustment   no CNDC")
    assert "\x1b" not in line
