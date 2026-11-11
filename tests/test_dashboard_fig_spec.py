import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pelagos_py.dashboard import fig_spec  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def unpack(blob):
    header_len = int.from_bytes(blob[:4], "little")
    header = json.loads(blob[4 : 4 + header_len])
    return header, blob[4 + header_len :]


def test_line_scatter_and_refline_round_trip():
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
    top.plot([0, 1, 2], [3.0, 4.0, 5.0], "--", color="red", label="line")
    top.axhline(4.5, color="black")
    bottom.scatter([0, 1, 2], [1.0, 2.0, 3.0], label="dots")
    top.set_title("Top")
    bottom.set_xlabel("x")
    fig.suptitle("Both")

    spec, reason, blob, full = fig_spec.serialise(fig)

    assert reason == ""
    assert spec["suptitle"] == "Both"
    assert spec["points"] == 6
    first, second = spec["panels"]
    assert first["title"] == "Top"
    assert second["xlabel"] == "x"
    assert first["share_x"] == second["share_x"]
    assert first["share_y"] != second["share_y"]
    assert first["traces"][0]["mode"] == "lines"
    assert first["traces"][0]["dash"] == "dash"
    assert first["traces"][0]["color"] == "#ff0000"
    assert first["traces"][0]["label"] == "line"
    assert first["reflines"][0]["axis"] == "y"
    assert first["reflines"][0]["value"] == 4.5
    assert second["traces"][0]["mode"] == "markers"
    assert second["traces"][0]["n"] == 3
    np.testing.assert_array_equal(full["1_0_y"], [1.0, 2.0, 3.0])

    header, data = unpack(blob)
    assert [(t["panel"], t["trace"], t["n"]) for t in header["traces"]] == [
        (0, 0, 3),
        (1, 0, 3),
    ]
    arrays = np.frombuffer(data, dtype="<f4").reshape(4, 3)
    np.testing.assert_array_equal(arrays[1], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(arrays[3], [1.0, 2.0, 3.0])


def test_dates_are_relative_to_the_first_timestamp():
    fig, ax = plt.subplots()
    times = np.array(["2024-01-01T00:00", "2024-01-01T00:01"], dtype="datetime64[ns]")
    ax.plot(times, [1.0, 2.0])

    spec, _, blob, _ = fig_spec.serialise(fig)

    assert spec["panels"][0]["xdate"] is True
    assert spec["t0"] == np.datetime64("2024-01-01", "ms").astype(float)
    header, data = unpack(blob)
    np.testing.assert_array_equal(np.frombuffer(data, dtype="<f4")[:2], [0.0, 60.0])


def test_per_point_colours_ship_as_rgba():
    fig, ax = plt.subplots()
    ax.scatter([0, 1, 2], [0, 1, 2], c=[0.0, 0.5, 1.0])

    spec, _, blob, _ = fig_spec.serialise(fig)

    assert spec["panels"][0]["traces"][0]["color"] is None
    header, _ = unpack(blob)
    assert header["traces"][0]["rgba"] is True


@pytest.mark.parametrize(
    "draw, reason",
    [
        (lambda ax: ax.imshow([[0, 1], [1, 0]]), "image"),
        (lambda ax: ax.fill_between([0, 1], [0, 1]), "Collection"),
        (lambda ax: ax.hist([1, 2, 2, 3]), "patches"),
    ],
)
def test_unsupported_artists_stay_png(draw, reason):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    draw(ax)
    spec, got_reason, blob, full = fig_spec.serialise(fig)
    assert spec is None and blob is None and full is None
    assert reason in got_reason


def test_empty_figure_has_no_spec():
    fig = plt.figure()
    assert fig_spec.serialise(fig)[:2] == (None, "no axes")
    fig.add_subplot()
    assert fig_spec.serialise(fig)[:2] == (None, "empty axes")
