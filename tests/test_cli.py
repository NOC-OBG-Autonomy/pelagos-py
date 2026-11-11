import sys

import pytest

from pelagos_py import cli


class FakePipeline:
    calls = []

    def __init__(self, source):
        self.source = source

    @classmethod
    def make_config(cls, path, ask=False, config_path=None):
        cls.calls.append(("make_config", path, ask, config_path))
        return cls(path)

    @classmethod
    def load_config(cls, path):
        cls.calls.append(("load_config", path))
        return cls(path)

    def run(self):
        self.calls.append(("run", self.source))


@pytest.fixture
def pipeline(monkeypatch):
    FakePipeline.calls = []
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    return FakePipeline


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["pelagos-py", *args])
    cli.main()


def test_no_command_prints_help(monkeypatch, capsys):
    run_cli(monkeypatch)
    assert "dashboard" in capsys.readouterr().out


def test_help_lists_every_command(monkeypatch, capsys):
    with pytest.raises(SystemExit) as info:
        run_cli(monkeypatch, "--help")
    assert info.value.code == 0
    out = capsys.readouterr().out
    for command in ("dashboard", "build", "run", "demo"):
        assert command in out


def test_unknown_command_exits_with_error(monkeypatch):
    with pytest.raises(SystemExit) as info:
        run_cli(monkeypatch, "explode")
    assert info.value.code == 2


def test_build_passes_options(monkeypatch, pipeline):
    run_cli(monkeypatch, "build", "glider.nc", "--ask", "-o", "out.yaml")
    assert pipeline.calls == [("make_config", "glider.nc", True, "out.yaml")]


def test_run_yaml_loads_the_config(monkeypatch, pipeline):
    run_cli(monkeypatch, "run", "config.yml")
    assert pipeline.calls == [("load_config", "config.yml"), ("run", "config.yml")]


def test_run_netcdf_builds_a_config_first(monkeypatch, pipeline):
    run_cli(monkeypatch, "run", "glider.nc")
    assert pipeline.calls == [
        ("make_config", "glider.nc", False, None),
        ("run", "glider.nc"),
    ]


def test_demo_prints_the_demo_path(monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_demo_file", lambda: "/data/demo_OG1.nc")
    run_cli(monkeypatch, "demo")
    assert capsys.readouterr().out.strip() == "/data/demo_OG1.nc"


def test_dashboard_missing_extra_exits_with_message(monkeypatch):
    def missing_extra():
        raise ImportError("needs the dashboard extra")

    monkeypatch.setattr(cli.dashboard_app, "run", missing_extra)
    with pytest.raises(SystemExit) as info:
        run_cli(monkeypatch, "dashboard")
    assert info.value.code == "needs the dashboard extra"
