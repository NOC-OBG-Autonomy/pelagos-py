import io
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from pelagos_py.utils import demo_data  # noqa: E402

# app.py creates WORKSPACE_DIR/configs on import, so point it away from the real workspace first
_real_workspace = demo_data.WORKSPACE_DIR
demo_data.WORKSPACE_DIR = Path(tempfile.mkdtemp(prefix="pelagos-test-workspace-"))
from pelagos_py.dashboard import app as dashboard_app  # noqa: E402
from pelagos_py.dashboard import runner  # noqa: E402

demo_data.WORKSPACE_DIR = _real_workspace


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    monkeypatch.setattr(dashboard_app, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(runner, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(dashboard_app, "CONFIG_DIR", config_dir)
    return tmp_path


@pytest.fixture
def client():
    return TestClient(dashboard_app.app, base_url="http://localhost")


@pytest.mark.parametrize("name", ["my_run", "my_run.yaml", "other.yml"])
def test_safe_config_path_accepts_plain_names(name, workspace):
    path = dashboard_app._safe_config_path(name)
    assert path.parent == (workspace / "configs").resolve()
    assert path.suffix in (".yaml", ".yml")


def test_safe_config_path_adds_yaml_suffix():
    assert dashboard_app._safe_config_path("my_run").name == "my_run.yaml"


@pytest.mark.parametrize(
    "name",
    [
        "../escape.yaml",
        "../../etc/passwd",
        "sub/dir.yaml",
        "/etc/passwd",
        "/tmp/evil.yaml",
        "..",
        "",
    ],
)
def test_safe_config_path_rejects_traversal(name):
    with pytest.raises(HTTPException) as info:
        dashboard_app._safe_config_path(name)
    assert info.value.status_code == 400


def test_save_load_delete_config_roundtrip(client, workspace):
    saved = client.post(
        "/api/configs", json={"name": "mine", "yaml_content": "steps: []\n"}
    )
    assert saved.json() == {"status": "saved", "name": "mine.yaml"}
    assert (workspace / "configs" / "mine.yaml").read_text() == "steps: []\n"
    assert client.get("/api/configs/mine.yaml").json()["yaml_content"] == "steps: []\n"
    client.delete("/api/configs/mine.yaml")
    assert not (workspace / "configs" / "mine.yaml").exists()


def test_save_refuses_protected_config(client):
    response = client.post(
        "/api/configs", json={"name": "default.yaml", "yaml_content": "x: 1\n"}
    )
    assert response.status_code == 403


def test_trusted_host_rejects_other_hosts():
    response = TestClient(dashboard_app.app).get("/api/run/status")
    assert response.status_code == 400


def test_cross_origin_post_is_blocked(client):
    response = client.post("/api/run/stop", headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_cross_site_fetch_metadata_post_is_blocked(client):
    response = client.post("/api/run/stop", headers={"Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 403


@pytest.mark.parametrize(
    "headers",
    [{}, {"Origin": "http://localhost"}, {"Sec-Fetch-Site": "same-origin"}],
)
def test_same_origin_post_is_allowed(client, headers):
    response = client.post("/api/run/stop", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "stopping"}


def test_cross_origin_get_is_allowed(client):
    response = client.get("/api/run/status", headers={"Origin": "http://evil.example"})
    assert response.status_code == 200


def test_security_headers_are_sent(client):
    headers = client.get("/api/run/status").headers
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert "frame-ancestors 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["Cache-Control"] == "no-store"


def validate(client, yaml_content):
    return client.post("/api/validate", json={"yaml_content": yaml_content}).json()


def test_validate_accepts_a_valid_config(client):
    result = validate(
        client, "steps:\n  - name: Generate Data\n  - name: Find Profiles\n"
    )
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_reports_unknown_step_on_its_index(client):
    result = validate(client, "steps:\n  - name: Generate Data\n  - name: Not A Step\n")
    assert result["ok"] is False
    assert result["issues"] == [
        {"index": 1, "name": "Not A Step", "error": "Unknown step 'Not A Step'."}
    ]


def test_validate_reports_unknown_parameter_on_its_index(client):
    result = validate(
        client,
        "steps:\n  - name: Generate Data\n  - name: Find Profiles\n"
        "    parameters: {bogus: 1}\n",
    )
    assert [issue["index"] for issue in result["issues"]] == [1]
    assert "bogus" in result["issues"][0]["error"]


def test_validate_reports_missing_variables_on_the_step(client):
    result = validate(client, "steps:\n  - name: Find Profiles\n")
    assert result["issues"][0]["index"] == 0
    assert "TIME" in result["issues"][0]["error"]


def test_validate_reports_yaml_errors(client):
    result = validate(client, "steps: [unclosed\n")
    assert result["ok"] is False
    assert result["yaml_error"]


class FakeProc:
    def __init__(self, output, returncode=0):
        self.stdout = io.BufferedReader(io.BytesIO(output))
        self.stdin = io.BytesIO()
        self.returncode = returncode

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


def test_pump_splits_lines_and_progress_redraws():
    run = runner._Run()
    proc = FakeProc(b"__PELAGOS_STEP__ 0\tLoad\r\n10%\r50%\rdone\n90%\r")
    run.proc = proc
    run._pump(proc)
    assert run.lines == ["__PELAGOS_STEP__ 0\tLoad", "done"]
    assert run.live == "90%"
    assert run.finished is True
    assert run.returncode == 0


def test_stale_pump_does_not_touch_the_current_run():
    run = runner._Run()
    stale = FakeProc(b"old line\n", returncode=130)
    run.proc = FakeProc(b"")
    run._pump(stale)
    assert run.lines == []
    assert run.finished is False
    assert run.returncode is None


def test_stopped_run_output_does_not_leak_into_next_run(tmp_path, monkeypatch):
    script = tmp_path / "fake_bootstrap.py"
    script.write_text(
        "import sys, time\n"
        "print('first' if 'first' in sys.argv[1] else 'second', flush=True)\n"
        "if 'first' in sys.argv[1]:\n"
        "    time.sleep(30)\n"
    )
    monkeypatch.setattr(runner, "RUN_BOOTSTRAP", script)
    run = runner._Run()
    try:
        run.start(tmp_path / "first.yaml")
        deadline = time.time() + 10
        while "first" not in run.lines and time.time() < deadline:
            time.sleep(0.02)
        assert "first" in run.lines
        run.stop()
        run.start(tmp_path / "second.yaml")
        deadline = time.time() + 10
        while not run.finished and time.time() < deadline:
            time.sleep(0.02)
        assert run.finished
        assert run.returncode == 0
        assert run.lines == ["second"]
    finally:
        run.stop()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT stop path is POSIX-only")
def test_stop_ends_the_subprocess(tmp_path, monkeypatch):
    script = tmp_path / "sleeper.py"
    script.write_text("import time\nprint('up', flush=True)\ntime.sleep(30)\n")
    monkeypatch.setattr(runner, "RUN_BOOTSTRAP", script)
    run = runner._Run()
    run.start(tmp_path / "config.yaml")
    deadline = time.time() + 10
    while "up" not in run.lines and time.time() < deadline:
        time.sleep(0.02)
    run.stop()
    assert not run.is_running()
