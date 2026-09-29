"""Tests the demo file downloader (src/pelagos_py/utils/demo_data.py)."""

import numpy as np
import pytest
import xarray as xr

from pelagos_py.utils import demo_data
from pelagos_py.utils.demo_data import DemoEntry


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.headers = {"Content-Length": str(len(content))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content


@pytest.fixture
def demo_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_data, "DEMO_DATA_DIR", tmp_path)
    return tmp_path


def test_unknown_demo_raises():
    with pytest.raises(ValueError, match="Unknown demo"):
        demo_data.get_demo_file("not_a_demo")


def test_existing_file_is_not_downloaded_again(demo_dir, monkeypatch):
    (demo_dir / "Nelson_646_R.nc").write_bytes(b"already here")
    monkeypatch.setattr(demo_data.requests, "get", lambda *a, **k: pytest.fail("downloaded"))
    assert demo_data.get_demo_file("nelson_646_r") == demo_dir / "Nelson_646_R.nc"


def test_download_is_cut_to_window(demo_dir, tmp_path_factory, monkeypatch):
    # Out-of-window and out-of-order samples are dropped.
    times = np.array(["2024-07-31", "2024-08-02", "2024-08-01", "2024-08-03", "2024-09-05"],
                     dtype="datetime64[ns]")
    source = tmp_path_factory.mktemp("src") / "full.nc"
    xr.Dataset({"TEMP": ("N_MEASUREMENTS", np.arange(5.0))},
               coords={"TIME": ("N_MEASUREMENTS", times)}).to_netcdf(source)
    monkeypatch.setitem(demo_data.DEMOS, "tiny", DemoEntry(
        "https://example.invalid/full.nc", "tiny.nc", ("2024-08-01", "2024-09-01"), "Tiny", "delayed"))
    monkeypatch.setattr(demo_data.requests, "get", lambda *a, **k: FakeResponse(source.read_bytes()))
    progress = []

    path = demo_data.get_demo_file("tiny", on_progress=lambda done, total: progress.append(done))

    with xr.open_dataset(path) as ds:
        assert ds["TEMP"].values.tolist() == [1.0, 3.0]
    assert progress == [source.stat().st_size]
    assert sorted(p.name for p in demo_dir.iterdir()) == ["tiny.nc"]
