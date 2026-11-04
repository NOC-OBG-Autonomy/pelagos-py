"""Draw the user-guide figures that explain an algorithm rather than show a step's diagnostics.

    python docs/scripts/explainer_figures.py [--out docs/_static]

Simulated records show each case cleanly; the demo file shows real data.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from types import SimpleNamespace  # noqa: E402

from pelagos_py.steps.processing import find_profiles as fp  # noqa: E402
from pelagos_py.steps.processing.mixed_layer_depth import (  # noqa: E402
    REFERENCE_HALF_WINDOW,
    MixedLayerDepthStep,
)
from pelagos_py.utils import fig_spec  # noqa: E402
from pelagos_py.utils.demo_data import get_demo_file  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAMPLE_SECONDS = 10
TOP_PLOT_H = 3.2  # short banner plots at the top of a page


def _leg(start, end, seconds):
    # Raised cosine: real dives ease to near-zero vertical speed at each turn.
    n = max(int(seconds / SAMPLE_SECONDS), 2)
    shape = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    return start + (end - start) * shape


def _flat(depth, seconds, wobble=0.0):
    n = int(seconds / SAMPLE_SECONDS)
    return depth + wobble * np.sin(np.linspace(0, 6 * np.pi, n))


def _record(pieces, noise=0.2, seed=1, gap_after=None, gap_minutes=20):
    depth = np.concatenate(pieces)
    rng = np.random.default_rng(seed)
    depth = np.clip(depth + rng.normal(0, noise, depth.size), 0, None)
    seconds = np.arange(depth.size) * float(SAMPLE_SECONDS)
    if gap_after is not None:
        seconds[gap_after:] += gap_minutes * 60
    time = np.datetime64("2024-01-01T00:00:00", "ns") + (seconds * 1e9).astype(
        "timedelta64[ns]"
    )
    return pd.DataFrame({"TIME": time, "PRES": depth}), seconds / 60


def _phase_points(ax, x, depth, phase):
    for code, name in fp.PHASE_NAMES.items():
        mask = phase == code
        if mask.any():
            fig_spec.points(
                ax, x[mask], depth[mask], color=fp.PHASE_COLOURS[code], label=name
            )
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def _flat_steps(ax, x, y):
    # One horizontal segment per run of equal values, with no vertical joins.
    y = np.asarray(y, dtype=float)
    change = np.flatnonzero(np.diff(y) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(y)])) - 1
    ax.hlines(y[starts], x[starts], x[ends], color="black", lw=2)


def method(out):
    df, minutes = _record(
        [_flat(0.5, 300), _leg(0.5, 150, 1200), _leg(150, 0.5, 1200), _flat(0.5, 300)],
        noise=0.5,
    )
    result = fp.find_profiles(df.copy())
    seconds = minutes * 60
    chunk_id = np.zeros(len(df), dtype=np.int32)
    smoothed = (
        pd.Series(df["PRES"].to_numpy(), index=pd.DatetimeIndex(df["TIME"]))
        .rolling("30s", center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    velocity = fp._smoothed_velocity(
        df["PRES"].to_numpy(), df["TIME"].to_numpy(), seconds, chunk_id, "30s"
    )

    fig, axes = fig_spec.new_fig(nrows=3, sharex=True)
    ax1, ax2, ax3 = axes[0][0], axes[1][0], axes[2][0]
    fig_spec.points(ax1, minutes, df["PRES"], color=fp.PHASE_COLOURS[0], label="raw")
    ax1.plot(minutes, smoothed, color="black", lw=1, label="smoothed (30 s)")
    ax1.invert_yaxis()
    fig_spec.style_axes(ax1, title="1. Smooth the depth record", ylabel="PRES [dbar]")
    fig_spec.legend(ax1)

    ax2.axhspan(-0.033, 0.033, color=fp.PHASE_COLOURS[0], alpha=0.3, lw=0)
    ax2.plot(minutes, velocity, color="black", lw=1, label="vertical velocity")
    ax2.axhline(
        0.033, color=fp.PHASE_COLOURS[2], ls="--", lw=1, label="+0.033 (descent)"
    )
    ax2.axhline(
        -0.033, color=fp.PHASE_COLOURS[1], ls="--", lw=1, label="-0.033 (ascent)"
    )
    fig_spec.style_axes(
        ax2, title="2. Compare the velocity with the threshold", ylabel="dbar/s"
    )
    fig_spec.legend(ax2)

    _phase_points(ax3, minutes, df["PRES"].to_numpy(), result["SCI_PHASE"].to_numpy())
    fig_spec.style_axes(
        ax3, title="3. Label each sample", xlabel="Minutes", ylabel="PRES [dbar]"
    )
    fig_spec.legend(ax3)
    fig_spec.finish(fig)
    fig.savefig(os.path.join(out, "profiles", "find_profiles_method.png"))
    plt.close(fig)


def phases(out):
    pieces = [
        _flat(0.5, 600),
        _leg(0.5, 200, 1300),
        _leg(200, 60, 900),
        _leg(60, 200, 900),
        _leg(200, 0.5, 1300),
        _flat(0.5, 400),
        _leg(0.5, 120, 800),
        _flat(120, 1200, wobble=1.0),
        _leg(120, 0.5, 800),
        _flat(0.5, 300),
        _leg(0.5, 150, 1000),
    ]
    gap_after = sum(len(p) for p in pieces)
    pieces.append(_leg(110, 0.5, 750))
    df, minutes = _record(pieces, gap_after=gap_after)
    result = fp.find_profiles(df.copy())

    fig, axes = fig_spec.new_fig(nrows=4, sharex=True, height_ratios=(3, 1, 1, 1))
    ax_depth, ax_profile, ax_direction, ax_cycle = (row[0] for row in axes)
    _phase_points(
        ax_depth, minutes, df["PRES"].to_numpy(), result["SCI_PHASE"].to_numpy()
    )
    gap_start, gap_end = minutes[gap_after - 1], minutes[gap_after]
    for ax in (ax_depth, ax_profile, ax_direction, ax_cycle):
        ax.axvspan(gap_start, gap_end, color=fp.PHASE_COLOURS[0], alpha=0.2, lw=0)
    ax_depth.text(
        (gap_start + gap_end) / 2,
        100,
        "data gap",
        ha="center",
        fontsize=fig_spec.FS_TICK,
    )
    fig_spec.style_axes(ax_depth, title="SCI_PHASE", ylabel="PRES [dbar]")
    fig_spec.legend(ax_depth)

    for ax, var in (
        (ax_profile, "PROFILE_NUMBER"),
        (ax_direction, "PROFILE_DIRECTION"),
        (ax_cycle, "CYCLE"),
    ):
        _flat_steps(ax, minutes, result[var])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig_spec.style_axes(ax, title=var)
    ax_direction.set_yticks([-1, 0, 1])
    ax_cycle.set_xlabel("Minutes", fontsize=fig_spec.FS_LABEL)
    fig_spec.finish(fig)
    fig.savefig(os.path.join(out, "profiles", "find_profiles_phases.png"))
    plt.close(fig)


def _demo_days(start, days):
    with xr.open_dataset(get_demo_file()) as ds:
        df = ds[["TIME", "PRES"]].to_dataframe().reset_index(drop=True)
    end = pd.Timestamp(start) + pd.Timedelta(days=days)
    return (
        df[(df["TIME"] >= start) & (df["TIME"] < end)].dropna().reset_index(drop=True)
    )


def demo_phases(out):
    df = _demo_days("2024-06-14", 1)
    result = fp.find_profiles(df.copy())
    fig, axes = fig_spec.new_fig()
    fig.set_size_inches(fig_spec.FIG_W, TOP_PLOT_H)
    ax = axes[0][0]
    _phase_points(
        ax, df["TIME"].to_numpy(), df["PRES"].to_numpy(), result["SCI_PHASE"].to_numpy()
    )
    fig_spec.date_axis(ax, which="x")
    fig_spec.style_axes(ax, ylabel="PRES [dbar]")
    fig_spec.legend(ax)
    fig_spec.finish(fig)
    fig.savefig(os.path.join(out, "profiles", "find_profiles_demo.png"))
    plt.close(fig)


def pressure_track(out):
    df = _demo_days("2024-06-14", 3)
    # Break the line at data gaps rather than drawing across them.
    pres = df["PRES"].mask(df["TIME"].diff() > pd.Timedelta(minutes=5))
    fig, axes = fig_spec.new_fig()
    fig.set_size_inches(fig_spec.FIG_W, TOP_PLOT_H)
    ax = axes[0][0]
    ax.plot(df["TIME"], pres, color=fig_spec.CATEGORY[0], lw=1)
    ax.invert_yaxis()
    fig_spec.date_axis(ax, which="x")
    fig_spec.style_axes(ax, ylabel="PRES [dbar]")
    fig_spec.finish(fig)
    fig.savefig(os.path.join(out, "pressure", "pressure_track.png"))
    plt.close(fig)


def mld_method(out):
    reference_depth = 10
    depth = np.arange(1.0, 150.0, 1.0)
    rng = np.random.default_rng(2)
    # Well-mixed down to 45 m, then a pycnocline/thermocline below it.
    below = np.clip(depth - 45, 0, None)
    sigma0 = 26.80 + 0.01 * below - 1e-5 * below**2 + rng.normal(0, 0.004, depth.size)
    temp = 12.0 - 0.05 * below + 1e-4 * below**2 + rng.normal(0, 0.02, depth.size)
    step = SimpleNamespace(reference_depth=reference_depth)

    fig, axes = fig_spec.new_fig(ncols=2, sharey=True)
    panels = (
        (
            axes[0][0],
            sigma0,
            0.03,
            False,
            "SIGMA0 [kg/m3]",
            "Density: increase of 0.03",
        ),
        (axes[0][1], temp, 0.2, True, "TEMP [degC]", "Temperature: change of 0.2"),
    )
    for ax, values, threshold, two_sided, xlabel, title in panels:
        mld = MixedLayerDepthStep._profile_mld(
            step, depth, values, threshold, two_sided
        )
        near = np.abs(depth - reference_depth) <= REFERENCE_HALF_WINDOW
        reference = np.median(values[near])
        ax.axhspan(
            reference_depth - REFERENCE_HALF_WINDOW,
            reference_depth + REFERENCE_HALF_WINDOW,
            color=fig_spec.CATEGORY[0],
            alpha=0.2,
            lw=0,
            label="reference window",
        )
        fig_spec.points(ax, values, depth, color=fp.PHASE_COLOURS[0], label="profile")
        ax.axvline(reference, color="black", lw=1, label="reference value")
        edges = [reference + threshold] + ([reference - threshold] if two_sided else [])
        for i, edge in enumerate(edges):
            ax.axvline(
                edge,
                color="black",
                ls="--",
                lw=1,
                label="threshold" if i == 0 else None,
            )
        ax.axhline(mld, color=fig_spec.CATEGORY[1], lw=2, label=f"MLD = {mld:.0f} m")
        fig_spec.style_axes(ax, title=title, xlabel=xlabel)
        ax.legend(fontsize=fig_spec.FS_LEGEND, loc="lower left")
    axes[0][0].invert_yaxis()
    axes[0][0].set_ylabel("DEPTH [m]", fontsize=fig_spec.FS_LABEL)
    fig_spec.finish(fig)
    fig.savefig(os.path.join(out, "mld", "mld_method.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "_static"))
    args = ap.parse_args()
    for folder in ("profiles", "pressure", "mld"):
        os.makedirs(os.path.join(args.out, folder), exist_ok=True)
    for draw in (method, phases, demo_phases, pressure_track, mld_method):
        draw(args.out)
        print("drew", draw.__name__)


if __name__ == "__main__":
    main()
