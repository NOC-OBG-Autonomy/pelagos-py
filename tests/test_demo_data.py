"""Tests the demo file downloader (src/pelagos_py/utils/demo_data.py)."""

import pytest

from pelagos_py.utils import demo_data


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.headers = {"Content-Length": str(len(content))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content


@pytest.fixture
def demo_file(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_data, "DEMO_DATA_DIR", tmp_path / "demo_data")
    monkeypatch.setattr(demo_data, "DEMO_FILE", tmp_path / "demo_data" / "demo.nc")
    return demo_data.DEMO_FILE


def test_existing_file_is_not_downloaded_again(demo_file, monkeypatch):
    demo_file.parent.mkdir()
    demo_file.write_bytes(b"already here")
    monkeypatch.setattr(
        demo_data.requests, "get", lambda *a, **k: pytest.fail("downloaded")
    )
    assert demo_data.get_demo_file() == demo_file


def test_download_reports_progress(demo_file, monkeypatch):
    monkeypatch.setattr(
        demo_data.requests, "get", lambda *a, **k: FakeResponse(b"netcdf bytes")
    )
    progress = []

    path = demo_data.get_demo_file(
        on_progress=lambda done, total: progress.append(done)
    )

    assert path.read_bytes() == b"netcdf bytes"
    assert progress == [len(b"netcdf bytes")]
    assert [p.name for p in demo_file.parent.iterdir()] == ["demo.nc"]
