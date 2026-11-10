import io
import re
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
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
    monkeypatch.setattr(dashboard_app, "DEMO_DATA_DIR", tmp_path / "demo_data")
    monkeypatch.setattr(dashboard_app, "DEMO_FILE", tmp_path / "demo_data" / "demo.nc")

    def no_download(on_progress=None):
        raise RuntimeError("no network in tests")

    monkeypatch.setattr(dashboard_app, "get_demo_file", no_download)
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


@pytest.mark.parametrize("name", ["default.yaml", "demo.yaml"])
def test_save_refuses_protected_config(client, name):
    response = client.post(
        "/api/configs", json={"name": name, "yaml_content": "x: 1\n"}
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


def test_columns_request_returns_when_the_run_exits(tmp_path, monkeypatch):
    # the run ignores the data request and exits, so columns() must not wait out its timeout
    script = tmp_path / "exits.py"
    script.write_text(
        "import time\nprint('__PELAGOS_PAUSE__ 0\\tStep', flush=True)\ntime.sleep(0.5)\n"
    )
    monkeypatch.setattr(runner, "RUN_BOOTSTRAP", script)
    run = runner._Run()
    run.start(tmp_path / "config.yaml")
    deadline = time.time() + 10
    while not run.paused and time.time() < deadline:
        time.sleep(0.02)
    started = time.time()
    assert run.columns(["TEMP"], timeout=20) is None
    assert time.time() - started < 5


def test_start_while_running_is_refused(tmp_path, monkeypatch):
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setattr(runner, "RUN_BOOTSTRAP", script)
    run = runner._Run()
    try:
        run.start(tmp_path / "config.yaml")
        first = run.proc
        with pytest.raises(RuntimeError):
            run.start(tmp_path / "config.yaml")
        assert run.proc is first
    finally:
        run.stop()


@pytest.mark.parametrize(
    "path", ["/api/run/continue", "/api/run/skip", "/api/run/rerun"]
)
def test_commands_are_refused_when_the_run_is_not_paused(client, path):
    assert client.post(path, json={}).status_code == 409


INDEX_HTML = (dashboard_app.STATIC_DIR / "index.html").read_text()
LOCAL_ASSETS = re.findall(r'(?:src|href)="(/[^"]+)"', INDEX_HTML)


def test_index_serves_the_dashboard_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == INDEX_HTML


@pytest.mark.parametrize("asset", LOCAL_ASSETS)
def test_index_assets_are_served(client, asset):
    response = client.get(asset)
    assert response.status_code == 200
    expected = {".js": "javascript", ".css": "text/css", ".svg": "image/svg+xml"}
    assert expected[Path(asset).suffix] in response.headers["content-type"]


# /api/run/stream only returns once a run finishes, so it is covered by the end-to-end run
SKIPPED_GET_ROUTES = {"/api/run/stream"}
GET_ROUTES = [
    route
    for route in dashboard_app.app.routes
    if isinstance(route, APIRoute)
    and "GET" in route.methods
    and route.path not in SKIPPED_GET_ROUTES
]


@pytest.mark.parametrize("route", GET_ROUTES, ids=[r.path for r in GET_ROUTES])
def test_get_routes_answer_without_server_error(client, route):
    path = re.sub(r"\{[^}]+\}", "missing", route.path)
    query = {param.name: "0" for param in route.dependant.query_params}
    response = client.get(path, params=query)
    assert response.status_code < 500


def test_registry_lists_every_step_and_qc_test_with_parameters(client):
    registry = client.get("/api/registry").json()
    real_steps = {
        name
        for name, cls in dashboard_app.STEP_CLASSES.items()
        if ".templates" not in cls.__module__
    }
    real_qc = {
        name
        for name, cls in dashboard_app.QC_CLASSES.items()
        if ".templates" not in cls.__module__
    }
    assert {step["name"] for step in registry["steps"]} == real_steps
    assert {test["name"] for test in registry["qc"]} == real_qc
    for entry in registry["steps"] + registry["qc"]:
        assert isinstance(entry["parameters"], list)
    generate = next(s for s in registry["steps"] if s["name"] == "Generate Data")
    assert "sampling_info" in {p["name"] for p in generate["parameters"]}
    assert registry["pipeline_fields"] == dashboard_app.PIPELINE_FIELDS


def test_list_configs_shows_saved_and_reference_configs(client):
    client.post("/api/configs", json={"name": "mine", "yaml_content": "steps: []\n"})
    listing = client.get("/api/configs").json()
    assert listing["configs"] == ["default.yaml", "demo.yaml", "mine.yaml"]
    assert listing["protected"] == ["default.yaml", "demo.yaml"]
    assert listing["downloaded"] == []


def test_save_overwrites_an_existing_config(client):
    client.post("/api/configs", json={"name": "mine", "yaml_content": "a: 1\n"})
    client.post("/api/configs", json={"name": "mine", "yaml_content": "b: 2\n"})
    assert client.get("/api/configs/mine.yaml").json()["yaml_content"] == "b: 2\n"


def test_load_default_config_returns_the_template(client):
    loaded = client.get("/api/configs/default.yaml").json()
    assert loaded["yaml_content"] == dashboard_app._template_text()


def test_load_missing_config_is_404(client):
    assert client.get("/api/configs/nope.yaml").status_code == 404


@pytest.mark.parametrize("name", ["default.yaml", "demo.yaml"])
def test_delete_refuses_protected_config(client, name):
    assert client.delete(f"/api/configs/{name}").status_code == 403


def test_load_demo_config_downloads_the_demo_file(client, monkeypatch):
    def fake_download(on_progress=None):
        dashboard_app.DEMO_FILE.parent.mkdir(parents=True)
        dashboard_app.DEMO_FILE.write_bytes(b"demo")

    monkeypatch.setattr(dashboard_app, "get_demo_file", fake_download)
    loaded = client.get("/api/configs/demo.yaml").json()
    assert loaded["build"]["file_path"] == str(dashboard_app.DEMO_FILE)
    assert client.get("/api/configs").json()["downloaded"] == ["demo.yaml"]
    assert client.delete("/api/demos/demo.yaml").json()["status"] == "deleted"
    assert not dashboard_app.DEMO_FILE.exists()


def test_failed_demo_download_is_502(client):
    assert client.post("/api/demos/demo.yaml/download").status_code == 502


@pytest.fixture
def outputs(workspace, monkeypatch):
    out = workspace / "out"
    out.mkdir()
    for name in [
        "run.log",
        "report.pdf",
        "paper.pdf",
        "result.nc",
        "other.nc",
        "notes.txt",
    ]:
        (out / name).write_text(name)
    (out / "run_report_figures_1").mkdir()
    (out / "run_report_figures_1" / "fig.png").write_bytes(b"png")
    # as if a run had announced its log and report
    run_files = {(out / "run.log").resolve(), (out / "report.pdf").resolve()}
    monkeypatch.setattr(runner._run, "outputs", run_files)
    return out


def list_outputs(client):
    payload = {"dirs": ["out"], "exports": ["out/result.nc"]}
    return client.post("/api/outputs", json=payload).json()


def test_outputs_lists_reports_logs_figures_and_exported_data(client, outputs):
    listing = list_outputs(client)
    assert listing["dirs"] == [str(outputs.resolve())]
    kinds = {f["name"]: f["kind"] for f in listing["files"]}
    assert kinds == {
        "run.log": "log",
        "report.pdf": "report",
        "result.nc": "data",
        "run_report_figures_1": "figures",
    }


def test_output_file_serves_only_listed_files(client, outputs):
    list_outputs(client)
    listed = client.get("/api/outputs/file", params={"path": str(outputs / "run.log")})
    assert listed.status_code == 200
    assert listed.text == "run.log"
    unlisted = client.get(
        "/api/outputs/file", params={"path": str(outputs / "notes.txt")}
    )
    assert unlisted.status_code == 404


def test_delete_and_clean_outputs(client, outputs):
    list_outputs(client)
    client.delete("/api/outputs/file", params={"path": str(outputs / "run.log")})
    assert not (outputs / "run.log").exists()
    cleaned = client.post("/api/outputs/clean")
    assert cleaned.json()["removed"] == 3
    assert sorted(p.name for p in outputs.iterdir()) == [
        "notes.txt",
        "other.nc",
        "paper.pdf",
    ]


def test_files_info_reports_existence_and_size(client, outputs):
    present = str(outputs / "run.log")
    missing = str(outputs / "absent.nc")
    info = client.post("/api/files/info", json={"paths": [present, missing]}).json()
    assert info == {
        present: {"exists": True, "size": len("run.log")},
        missing: {"exists": False, "size": 0},
    }


def test_reveal_opens_the_folder_without_a_real_file_manager(
    client, outputs, monkeypatch
):
    opened = []
    monkeypatch.setattr(dashboard_app.subprocess, "Popen", opened.append)
    client.post("/api/configs/reveal")
    client.post("/api/files/reveal", json={"path": str(outputs / "run.log")})
    assert [cmd[-1] for cmd in opened] == [
        str(dashboard_app.CONFIG_DIR),
        str(outputs),
    ]


@pytest.mark.parametrize("path", ["/api/files/pick", "/api/browse"])
def test_native_dialog_routes_are_post_only(client, path, monkeypatch):
    opened = []
    monkeypatch.setattr(dashboard_app, "_native_dialog", lambda *args: opened.append(1))
    # a GET falls through to the static mount, which 404s
    assert client.get(path).status_code in (404, 405)
    assert opened == []


def test_pick_expands_folders_to_their_netcdf_files(client, outputs, monkeypatch):
    chosen = [str(outputs), str(outputs / "notes.txt")]
    monkeypatch.setattr(dashboard_app, "_native_dialog", lambda *args, **kw: chosen)
    picked = client.post("/api/files/pick", json={}).json()
    assert picked["paths"] == [str(outputs / "other.nc"), str(outputs / "result.nc")]


END_TO_END_CONFIG = """\
pipeline: {on_step_fail: stop}
steps:
  - name: Generate Data
    parameters: {sampling_info: ["2025-01-01", "2025-01-02", 600]}
  - name: Find Profiles
    diagnostics: true
"""


def wait_for(condition, timeout=30):
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        time.sleep(0.1)
    return condition()


def run_lines():
    with runner._run._lock:
        return list(runner._run.lines)


def test_end_to_end_run_through_the_api(client):
    try:
        started = client.post("/api/run", json={"yaml_content": END_TO_END_CONFIG})
        assert started.json() == {"status": "started"}
        # Find Profiles has diagnostics on, so the run pauses after it for review
        assert wait_for(
            lambda: any(line.startswith("__PELAGOS_PAUSE__ 1") for line in run_lines())
        )
        assert client.post("/api/run", json={"yaml_content": ""}).status_code == 409
        figure = next(
            line.split(" ", 1)[1].split("\t")
            for line in run_lines()
            if line.startswith("__PELAGOS_FIG__ ")
        )
        png = client.get(f"/api/run/figure/{figure[0]}")
        assert png.headers["content-type"] == "image/png"
        assert client.get(f"/api/run/figspec/{figure[2]}").json()["panels"]
        columns = client.get("/api/run/columns", params={"names": "PROFILE_NUMBER"})
        assert columns.status_code == 200
        assert columns.content
        assert client.post("/api/run/continue").status_code == 200
        # a double click must not be queued for a later pause
        assert client.post("/api/run/continue").status_code == 409
        assert wait_for(lambda: runner._run.finished)
        status = client.get("/api/run/status").json()
        assert status["running"] is False
        assert status["returncode"] == 0
        assert status["line_count"] > 0
        stream = client.get("/api/run/stream").text
        assert "Generate Data" in stream
        assert stream.endswith("event: end\ndata: 0\n\n")
        assert client.post("/api/run/stop").json() == {"status": "stopping"}
    finally:
        runner._run.stop()
