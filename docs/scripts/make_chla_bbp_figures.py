"""Regenerate the CHLA / BBP user-guide figures (docs/_static/chla, docs/_static/bbp).

Runs a doc-matching pipeline on the first two months of Growler_677, then draws
each figure the markdown pages reference. Run from the repo root:

    python docs/scripts/make_chla_bbp_figures.py [--cache processed.pkl]

--cache stores the processed dataset so re-plotting skips the pipeline.
"""
import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pelagos_py.pipeline import Pipeline
from pelagos_py.steps.processing.chla_quenching import (
    CALC_SUFFIX,
    chla_quenching_correction,
)
from pelagos_py.utils import fig_spec

DATA = "/Users/orlpru/Desktop/Run_Pipeline/input/ReBELS/Growler_677.nc"
START, END = "2025-03-23T00:00:00", "2025-05-23T00:00:00"
HERE = os.path.dirname(os.path.abspath(__file__))
CHLA_DIR = os.path.join(HERE, "..", "_static", "chla")
BBP_DIR = os.path.join(HERE, "..", "_static", "bbp")
INF = float("inf")
QC = [3, 4, 9]

CONFIG = {
    "pipeline": {
        "name": "User-guide figures (Growler_677)",
        "description": "CHLA / BBP processing used for the docs figures",
        "visualisation": False,
    },
    "steps": [
        {"name": "Load OG1", "parameters": {
            "file_path": DATA, "filter_bad_time": True,
            "data_start": START, "data_end": END}},
        {"name": "Apply QC", "parameters": {"qc_settings": {
            "impossible date qc": {},
            "impossible location qc": {},
            "range qc": {
                "variable_ranges": {
                    "PRES": {3: [-5, -2.4, "inside"], 4: [-INF, -5, "inside"]},
                    "CHLA": {4: [-0.2, 100, "outside"]},
                    "BETA_BACKSCATTERING700": {3: [0, 0.01, "outside"],
                                               4: [-1.0e-4, 0.05, "outside"]},
                    "DOWNWELLING_PAR": {4: [-5, 2500, "outside"]},
                },
                "also_flag": {"PRES": ["CNDC", "TEMP"]},
            },
            "stuck value qc": {
                "variables": {"PRES": 2, "DOWNWELLING_PAR": 5},
                "also_flag": {"PRES": ["CNDC", "TEMP"]},
            },
        }}},
        {"name": "Interpolate Data", "parameters": {
            "variables": {"PRES": QC, "LATITUDE": QC, "LONGITUDE": QC}}},
        {"name": "Derive CTD", "parameters": {"to_derive": ["DEPTH"]}},
        {"name": "Find Profiles", "parameters": {}},
        {"name": "Derive CTD", "parameters": {
            "to_derive": ["PRAC_SALINITY", "ABS_SALINITY", "CONS_TEMP"]}},
        {"name": "Mixed Layer Depth", "parameters": {}},
        {"name": "Interpolate PAR", "parameters": {}},
        {"name": "BBP from Beta", "parameters": {
            "apply_to": "BETA_BACKSCATTERING700", "output_as": "BBP700"}},
        {"name": "Isolate BBP Spikes", "parameters": {
            "apply_to": "BBP700", "window_size": 50, "method": "median"}},
        {"name": "Deep Correction", "parameters": {
            "apply_to": "CHLA", "depth_var": "DEPTH", "depth_threshold": 950.0}},
        {"name": "Apply QC", "parameters": {"qc_settings": {"range qc": {
            "variable_ranges": {"DEPTH": {3: [0, 5, "inside"], 2: [0, 5, "outside"]}},
            "flag_instead": {"DEPTH": ["CHLA_ADJUSTED"]}}}}},
    ],
}

CHL = "CHLA_ADJUSTED"
BBP = "BBP700_BASELINE"
C_ORIG, C_CORR, C_RECON, C_RATIO, C_NIGHT = (fig_spec.CATEGORY[i] for i in (1, 2, 3, 4, 0))
CHL_LABEL = fig_spec.axis_label(CHL, "mg m-3")
BBP_LABEL = fig_spec.axis_label(BBP, "m-1")
DEPTH_LABEL = fig_spec.axis_label("DEPTH", "m")


# ----------------------------------------------------------------------------- data
def processed_data(cache):
    if cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    pipe = Pipeline(config=CONFIG)
    pipe.run()
    ds = pipe.get_data()
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ds


def run_method(ds, method, diagnostics=False, **extra):
    step = chla_quenching_correction(
        "CHLA Quenching",
        parameters={"method": method, "apply_to": CHL, **extra},
        diagnostics=diagnostics,
        context={"data": ds.copy()},
    )
    step.generate_diagnostics = lambda: None
    step._suppress_warn = True
    step.run()
    return step


def profile(step, pn):
    return step._profile_subsets([pn])[pn]


def arr(p, var):
    return np.asarray(p[var].values, dtype=float)


def scalar(p, name):
    return chla_quenching_correction._profile_scalar(p, name)


def elev(step, pn):
    return step._sun_elevation_for(int(pn))


def ratio_profile(p):
    f, b = arr(p, CHL + CALC_SUFFIX), arr(p, BBP + CALC_SUFFIX)
    return np.divide(f, b, out=np.full_like(f, np.nan), where=(b != 0))


def max_ratio(p, z_win):
    depth = arr(p, "DEPTH")
    r = np.where((depth <= z_win) & np.isfinite(depth), ratio_profile(p), np.nan)
    i = int(np.nanargmax(r))
    return r[i], float(depth[i])


def pick_day_profile(step, regime=None, need_par=False, quantile=0.85, require_change=False):
    change = np.abs(step.data[CHL].values - step.data_copy[CHL].values)
    change[~np.isfinite(change)] = 0.0
    cands = []
    for pn in step.sun_args.index:
        pn = int(pn)
        if elev(step, pn) < 25:
            continue
        p = profile(step, pn)
        mld, zeu, zipar = (scalar(p, k) for k in ("MLD", "ZEU", "Z_IPAR"))
        if not all(np.isfinite(v) and v > 0 for v in (mld, zeu, zipar)):
            continue
        if regime == "shallow" and not zipar > mld:
            continue
        if regime == "deep" and not zipar <= mld:
            continue
        depth, bbp = arr(p, "DEPTH"), arr(p, BBP)
        if np.count_nonzero(np.isfinite(bbp) & (depth <= 50)) < 20:
            continue
        if need_par:
            par = arr(p, step.par_var)
            if np.count_nonzero(np.isfinite(par) & (par > 0)) < 4:
                continue
        total = float(change[step._profile_index[pn]].sum())
        if require_change and total <= 0:
            continue
        cands.append((pn, total))
    if not cands:
        return None
    cands.sort(key=lambda c: c[1])
    return cands[int(quantile * (len(cands) - 1))][0]


# ----------------------------------------------------------------------------- plotting
def save(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def zmax_for(*depths):
    finite = [d for d in depths if np.isfinite(d)]
    return float(np.clip(1.6 * max(finite), 30, 200)) if finite else 100.0


def hline(ax, z, label, color="k", ls="--"):
    if np.isfinite(z):
        ax.axhline(z, ls=ls, color=color, lw=1.2, label=f"{label} = {z:.0f} m")


def line(ax, x, depth, lw=1.6, **kw):
    ok = np.isfinite(x) & np.isfinite(depth)
    order = np.argsort(depth[ok])
    ax.plot(x[ok][order], depth[ok][order], lw=lw, **kw)


def corrected_points(ax, orig, corr, depth):
    changed = np.isfinite(corr) & np.isfinite(orig) & (np.abs(corr - orig) > 1e-9)
    fig_spec.points(ax, corr[changed], depth[changed], color=C_CORR,
                    label=f"Corrected ({changed.sum()} points changed)")


def finish_profile(ax, zmax, xlabel, title=None, ylabel=DEPTH_LABEL):
    ax.set_ylim(zmax, 0)
    fig_spec.style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    fig_spec.legend(ax)


def stamp(step, pn):
    t = pd.Timestamp(step.sun_args.loc[pn, "TIME"])
    return f"profile {pn}, {t:%Y-%m-%d %H:%M} UTC, sun {elev(step, pn):.0f}°"


def fig_sackmann(step, pn):
    p = profile(step, pn)
    depth, orig, bbp = arr(p, "DEPTH"), arr(p, CHL), arr(p, BBP)
    corr = step.apply_sackmann2008_quenching_correction(p)
    mld = scalar(p, "MLD")
    r_max, z_r = max_ratio(p, mld)
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, orig, depth, color=C_ORIG, label="Original " + CHL)
    line(ax, bbp * r_max, depth, color=C_RECON, label=r"$b_{bp} \times R_{max}$ reconstruction")
    corrected_points(ax, orig, corr, depth)
    hline(ax, mld, "MLD")
    hline(ax, z_r, r"depth of $R_{max}$", color=C_RATIO, ls=":")
    finish_profile(ax, zmax_for(mld, z_r), CHL_LABEL)
    fig_spec.finish(fig, f"Sackmann et al. (2008) — {stamp(step, pn)}")
    save(fig, CHLA_DIR, "npq_sackmann2008.png")


def fig_xing2012(step, pn):
    p = profile(step, pn)
    depth, orig = arr(p, "DEPTH"), arr(p, CHL)
    corr = step.apply_xing2012_quenching_correction(p)
    mld = scalar(p, "MLD")
    f = np.where(depth <= mld, arr(p, CHL + CALC_SUFFIX), np.nan)
    z_max = float(depth[np.nanargmax(f)])
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, orig, depth, color=C_ORIG, label="Original " + CHL)
    corrected_points(ax, orig, corr, depth)
    hline(ax, mld, "MLD")
    hline(ax, z_max, "depth of max in-ML CHLA", color=C_RATIO, ls=":")
    finish_profile(ax, zmax_for(mld, z_max), CHL_LABEL)
    fig_spec.finish(fig, f"Xing et al. (2012) — {stamp(step, pn)}")
    save(fig, CHLA_DIR, "npq_xing2012.png")


def fig_biermann(step, pn):
    p = profile(step, pn)
    depth, orig = arr(p, "DEPTH"), arr(p, CHL)
    corr = step.apply_biermann2015_quenching_correction(p)
    zeu = scalar(p, "ZEU")
    f = np.where(depth <= zeu, arr(p, CHL + CALC_SUFFIX), np.nan)
    z_max = float(depth[np.nanargmax(f)])
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, orig, depth, color=C_ORIG, label="Original " + CHL)
    corrected_points(ax, orig, corr, depth)
    hline(ax, zeu, r"$Z_{eu}$")
    hline(ax, z_max, "depth of max CHLA (0–Zeu)", color=C_RATIO, ls=":")
    finish_profile(ax, zmax_for(zeu, z_max), CHL_LABEL)
    fig_spec.finish(fig, f"Biermann et al. (2015) — {stamp(step, pn)}")
    save(fig, CHLA_DIR, "npq_biermann2015.png")


def fig_hemsley(step, pn):
    reg = step._hemsley_regression
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, reg["bbp"], reg["fl"], color=C_NIGHT, size=3, alpha=0.35,
                    label=f"night samples ≤ 60 m (n={reg['n']})")
    x = np.array([np.nanmin(reg["bbp"]), np.nanmax(reg["bbp"])])
    ax.plot(x, reg["slope"] * x + reg["intercept"], color=C_CORR, lw=2,
            label=f"CHLA = {reg['slope']:.3g}·bbp + {reg['intercept']:.3g}  (R² = {reg['r2']:.2f})")
    fig_spec.style_axes(ax, xlabel=BBP_LABEL, ylabel=CHL_LABEL)
    fig_spec.legend(ax)
    fig_spec.finish(fig, "Hemsley et al. (2015) — deployment-wide nighttime regression")
    save(fig, CHLA_DIR, "npq_hemsley_regression.png")

    p = profile(step, pn)
    depth, orig, bbp = arr(p, "DEPTH"), arr(p, CHL), arr(p, BBP)
    corr = step.apply_hemsley2015_quenching_correction(p)
    zeu = scalar(p, "ZEU")
    recon = np.where(depth <= zeu, reg["slope"] * bbp + reg["intercept"], np.nan)
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, orig, depth, color=C_ORIG, label="Original " + CHL)
    line(ax, recon, depth, color=C_RECON, label="night regression × day bbp")
    corrected_points(ax, orig, corr, depth)
    hline(ax, zeu, r"$Z_{eu}$")
    finish_profile(ax, zmax_for(zeu), CHL_LABEL)
    fig_spec.finish(fig, f"Hemsley et al. (2015) — {stamp(step, pn)}")
    save(fig, CHLA_DIR, "npq_hemsley_profile.png")


def fig_swart(step, pn):
    p = profile(step, pn)
    depth, orig, bbp = arr(p, "DEPTH"), arr(p, CHL), arr(p, BBP)
    corr = step.apply_swart2015_quenching_correction(p)
    zeu = scalar(p, "ZEU")
    r_max, z_r = max_ratio(p, zeu)
    fig, axes = fig_spec.new_fig(1, 2, sharey=True)
    ax, axr = axes[0]
    fig_spec.points(ax, orig, depth, color=C_ORIG, label="Original " + CHL)
    line(ax, bbp * r_max, depth, color=C_RECON, label=r"$b_{bp} \times R_{max}$")
    corrected_points(ax, orig, corr, depth)
    hline(ax, zeu, r"$Z_{eu}$")
    hline(ax, z_r, r"depth of $R_{max}$", color=C_RATIO, ls=":")
    finish_profile(ax, zmax_for(zeu, z_r), CHL_LABEL)
    fig_spec.points(axr, ratio_profile(p), depth, color=C_RATIO, label=r"$F / b_{bp}$")
    axr.plot([r_max], [z_r], marker="*", ms=12, color=C_CORR, ls="",
             label=f"$R_{{max}}$ = {r_max:.3g}")
    hline(axr, zeu, r"$Z_{eu}$")
    finish_profile(axr, zmax_for(zeu, z_r), "CHLA / bbp", ylabel=None)
    fig_spec.finish(fig, f"Swart et al. (2015) — {stamp(step, pn)}")
    save(fig, CHLA_DIR, "npq_swart2015.png")


def quenching_depth_debug(z, fl_day, fl_night, max_photic_depth):
    # _quenching_depth with its intermediates exposed for plotting
    D = np.asarray(fl_night, float) - np.asarray(fl_day, float)
    mask_all = np.isfinite(z) & np.isfinite(D) & (z >= 0) & (z <= max_photic_depth)
    mask = mask_all & (D > 0)
    out = {"z_all": z[mask_all], "D_all": D[mask_all], "qd": np.nan,
           "anchor": None, "cands": []}
    if mask.sum() < 3:
        return out
    zz, DD = z[mask], D[mask]
    order = np.argsort(zz)
    zz, DD = zz[order], DD[order]
    top = zz <= 5
    if not top.any():
        return out
    anchor = int(np.argmax(np.where(top, DD, -np.inf)))
    z_a, D_a = zz[anchor], DD[anchor]
    cands = set()
    for i in np.argsort(np.abs(DD)):
        if zz[i] > z_a:
            cands.add(int(i))
        if len(cands) >= 5:
            break
    for i in range(len(DD) - 1):
        crossing = DD[i] == 0 or (DD[i] > 0) != (DD[i + 1] > 0)
        if crossing and zz[i + 1] > z_a:
            cands.add(i + 1)
    best, best_g = np.nan, -np.inf
    for i in cands:
        g = abs(D_a - DD[i]) / (zz[i] - z_a)
        if g > best_g:
            best_g, best = g, float(zz[i])
    out.update(qd=best, anchor=(D_a, z_a), cands=[(DD[i], zz[i]) for i in cands])
    return out


def fig_thomalla(step, pn):
    p = profile(step, pn)
    depth, orig, bbp = arr(p, "DEPTH"), arr(p, CHL), arr(p, BBP)
    ref = step._night_refs[step._thomalla_day_night[pn]]
    night = pd.Timestamp(int(ref["time"]))

    fig, axes = fig_spec.new_fig(1, 3, sharey=True)
    a1, a2, a3 = axes[0]
    zmax = zmax_for(step.max_photic_depth)
    fig_spec.points(a1, ref["fl"], ref["z"], color=C_NIGHT, label="night mean CHLA")
    finish_profile(a1, zmax, CHL_LABEL)
    fig_spec.points(a2, ref["fl"] / ref["ratio"], ref["z"], color=C_NIGHT, label="night mean bbp")
    finish_profile(a2, zmax, BBP_LABEL, ylabel=None)
    fig_spec.points(a3, ref["ratio"], ref["z"], color=C_RATIO, label="night CHLA : bbp")
    finish_profile(a3, zmax, "CHLA / bbp", ylabel=None)
    fig_spec.finish(fig, f"Thomalla et al. (2018) — 1 m binned night reference, {night:%Y-%m-%d}")
    save(fig, CHLA_DIR, "npq_thomalla_night_reference.png")

    ratio_at_z = np.interp(depth, ref["z"], ref["ratio"], right=np.nan)
    fl_night = np.interp(depth, ref["z"], ref["fl"], left=np.nan, right=np.nan)
    dbg = quenching_depth_debug(depth, arr(p, CHL + CALC_SUFFIX), fl_night, step.max_photic_depth)
    qd = dbg["qd"]

    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, dbg["D_all"], dbg["z_all"], color=C_ORIG, label="night − day CHLA")
    ax.axvline(0, color="0.4", lw=1, ls=":")
    if dbg["anchor"]:
        ax.plot([dbg["anchor"][0]], [dbg["anchor"][1]], marker="*", ms=13, ls="",
                color=C_CORR, label="anchor (max diff, top 5 m)")
        cd, cz = zip(*dbg["cands"])
        ax.plot(cd, cz, marker="s", ms=6, ls="", color=C_RECON, label="candidate depths")
    hline(ax, qd, "quenching depth", color=C_RATIO)
    finish_profile(ax, zmax_for(step.max_photic_depth), fig_spec.axis_label("night − day CHLA", "mg m-3"))
    fig_spec.finish(fig, f"Thomalla et al. (2018) — quenching depth, {stamp(step, pn)}\n(top-5 m flag 3 not gating the calculation)")
    save(fig, CHLA_DIR, "npq_thomalla_quenching_depth.png")

    corr = step.apply_thomalla2018_quenching_correction(p)
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, orig, depth, color=C_ORIG, label="Original " + CHL)
    line(ax, ratio_at_z * bbp, depth, color=C_RECON, label="night ratio × day bbp")
    corrected_points(ax, orig, corr, depth)
    hline(ax, qd, "quenching depth", color=C_RATIO)
    finish_profile(ax, zmax_for(qd, 40), CHL_LABEL)
    fig_spec.finish(fig, f"Thomalla et al. (2018) — {stamp(step, pn)}\n(top-5 m flag 3 not gating the calculation)")
    save(fig, CHLA_DIR, "npq_thomalla_profile.png")


def night_groups(step):
    # consecutive night profiles grouped into nights, as _build_night_references does
    pns = [int(p) for p in step.sun_args.index]
    times = {pn: pd.Timestamp(step.sun_args.loc[pn, "TIME"]).value for pn in pns}
    groups, current = [], []
    for pn in sorted(pns, key=times.get):
        if elev(step, pn) < step.night_max_elevation:
            current.append(pn)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups, times


def bin_profiles(step, pns):
    pnum = step.data_copy["PROFILE_NUMBER"].values
    m = np.isin(pnum, pns)
    return step._bin_night(step.data_copy["DEPTH"].values[m],
                           step.data_copy[CHL + CALC_SUFFIX].values[m],
                           step.data_copy[BBP + CALC_SUFFIX].values[m])


def fig_mitchell(step, pn):
    groups, times = night_groups(step)
    t = times[pn]
    nights = []
    for members in groups:
        ref = bin_profiles(step, members)
        if ref is not None:
            nights.append((float(np.median([times[m] for m in members])), members, ref))
    before = [n for n in nights if n[0] <= t]
    after = [n for n in nights if n[0] > t]
    if not before or not after:
        print("mitchell: no bracketing nights for profile", pn)
        return
    (t0, m0, r0), (t1, m1, r1) = before[-1], after[0]
    alpha = (t - t0) / (t1 - t0)
    z = np.arange(0.5, step.max_photic_depth * 1.6, 1.0)

    def on(ref, key):
        return np.interp(z, ref["z"], ref[key], left=np.nan, right=np.nan)

    mz = (1 - alpha) * on(r0, "ratio") + alpha * on(r1, "ratio")
    last, first = bin_profiles(step, [max(m0, key=times.get)]), bin_profiles(step, [min(m1, key=times.get)])
    flz = (1 - alpha) * on(last, "ratio") + alpha * on(first, "ratio")

    fig, axes = fig_spec.new_fig(1, 2, sharey=True)
    ax, bx = axes[0]
    d0, d1 = pd.Timestamp(int(t0)), pd.Timestamp(int(t1))
    line(ax, on(r0, "ratio"), z, color=C_NIGHT, ls="--", label=f"night mean, {d0:%d %b}")
    line(ax, on(r1, "ratio"), z, color=C_ORIG, ls="--", label=f"night mean, {d1:%d %b}")
    line(ax, mz, z, color=C_CORR, label=f"interpolated (α = {alpha:.2f})")
    finish_profile(ax, zmax_for(step.max_photic_depth), "night CHLA / bbp", title="Mean (MZ)")
    line(bx, on(last, "ratio"), z, color=C_NIGHT, ls="--", label=f"last profile, night of {d0:%d %b}")
    line(bx, on(first, "ratio"), z, color=C_ORIG, ls="--", label=f"first profile, night of {d1:%d %b}")
    line(bx, flz, z, color=C_CORR, label=f"interpolated (α = {alpha:.2f})")
    finish_profile(bx, zmax_for(step.max_photic_depth), "night CHLA / bbp", title="First–Last (FLZ)", ylabel=None)
    fig_spec.finish(fig, f"Mitchell et al. (2024) — time-interpolated night reference for {stamp(step, pn)}")
    save(fig, CHLA_DIR, "npq_mitchell_interpolation.png")

    p = profile(step, pn)
    depth, orig, bbp = arr(p, "DEPTH"), arr(p, CHL), arr(p, BBP)
    ratio_t = np.interp(depth, z, mz, right=np.nan)
    fl_t = (1 - alpha) * np.interp(depth, r0["z"], r0["fl"], left=np.nan, right=np.nan) \
        + alpha * np.interp(depth, r1["z"], r1["fl"], left=np.nan, right=np.nan)
    qd = quenching_depth_debug(depth, arr(p, CHL + CALC_SUFFIX), fl_t, step.max_photic_depth)["qd"]
    recon = ratio_t * bbp
    corr = np.where((depth <= qd) & (recon > orig), recon, orig)
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, orig, depth, color=C_ORIG, label="Original " + CHL)
    line(ax, recon, depth, color=C_RECON, label="interpolated night ratio × day bbp")
    corrected_points(ax, orig, corr, depth)
    hline(ax, qd, "quenching depth", color=C_RATIO)
    finish_profile(ax, zmax_for(qd, 40), CHL_LABEL)
    fig_spec.finish(fig, f"Mitchell et al. (2024, MZ) — {stamp(step, pn)}\n(top-5 m flag 3 not gating the calculation)")
    save(fig, CHLA_DIR, "npq_mitchell_profile.png")


def fig_xing2018(step, pn):
    p = profile(step, pn)
    depth, orig, bbp = arr(p, "DEPTH"), arr(p, CHL), arr(p, BBP)
    corr = step._apply_xing_terrats(p, hybrid=False)
    mld, zipar = scalar(p, "MLD"), scalar(p, "Z_IPAR")
    z_ref = min(mld, zipar)
    r_max, z_r = max_ratio(p, z_ref)
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, orig, depth, color=C_ORIG, label="Original " + CHL)
    line(ax, np.where(depth <= z_ref, bbp * r_max, np.nan), depth, color=C_RECON,
         label=r"$b_{bp} \times R_{max}$")
    corrected_points(ax, orig, corr, depth)
    hline(ax, mld, "MLD")
    hline(ax, zipar, r"$Z_{IPAR}$ (15 µmol m$^{-2}$ s$^{-1}$)", color="0.5")
    hline(ax, z_ref, r"NPQ layer $z_{ref}$ = min(MLD, $Z_{IPAR}$)", color=C_CORR, ls="-")
    hline(ax, z_r, r"depth of $R_{max}$", color=C_RATIO, ls=":")
    finish_profile(ax, zmax_for(mld, zipar), CHL_LABEL)
    fig_spec.finish(fig, f"Xing et al. (2018) S08+ — {stamp(step, pn)}")
    save(fig, CHLA_DIR, "npq_xing2018.png")


def fig_terrats(step, pn_deep, pn_shallow):
    fig, axes = fig_spec.new_fig(1, 2, sharey=True)
    zmax = 0
    for ax, pn, name in zip(axes[0], (pn_deep, pn_shallow), ("Deep mixing", "Shallow mixing")):
        if pn is None:
            ax.text(0.5, 0.5, f"no {name.lower()} profile\nin this record", ha="center",
                    va="center", transform=ax.transAxes)
            continue
        p = profile(step, pn)
        mld, zipar = scalar(p, "MLD"), scalar(p, "Z_IPAR")
        fig_spec.points(ax, arr(p, CHL), arr(p, "DEPTH"), color=C_ORIG, label=CHL)
        hline(ax, mld, "MLD")
        hline(ax, zipar, r"$Z_{IPAR}$", color="0.5")
        rel = "≤" if zipar <= mld else ">"
        ax.set_title(f"{name}: $Z_{{IPAR}}$ {rel} MLD  (profile {pn})", fontsize=fig_spec.FS_TITLE)
        zmax = max(zmax, zmax_for(mld, zipar))
    for i, ax in enumerate(axes[0]):
        finish_profile(ax, zmax, CHL_LABEL, ylabel=DEPTH_LABEL if i == 0 else None)
    fig_spec.finish(fig, "Terrats et al. (2020) — mixing regimes")
    save(fig, CHLA_DIR, "npq_terrats_mixing_regimes.png")

    r, ipar_mid, e = 0.092, 261.0, 2.2
    I = np.logspace(-1, 3.5, 300)
    s = r + (1 - r) / (1 + (I / ipar_mid) ** e)
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    ax.plot(I, s, color=C_CORR, lw=2, label=r"$s(I) = r + (1-r)\,/\,[1 + (I/I_{50})^{e}]$")
    ax.axvline(ipar_mid, color="0.5", ls="--", lw=1.2, label=f"$I_{{50}}$ = {ipar_mid:g}")
    ax.axhline(r, color=C_RATIO, ls=":", lw=1.2, label=f"r = {r:g}")
    ax.set_xscale("log")
    fig_spec.style_axes(ax, xlabel=fig_spec.axis_label("PAR", "µmol photons m-2 s-1"),
                        ylabel="retained fluorescence fraction s(I)")
    fig_spec.legend(ax)
    fig_spec.finish(fig, "Xing et al. (2018) sigmoid (e = 2.2)")
    save(fig, CHLA_DIR, "npq_terrats_sigmoid.png")

    if pn_shallow is None:
        print("terrats profile: no shallow-mixing profile with PAR")
        return
    p = profile(step, pn_shallow)
    depth, orig, bbp, par = arr(p, "DEPTH"), arr(p, CHL), arr(p, BBP), arr(p, step.par_var)
    chl_calc, bbp_calc = arr(p, CHL + CALC_SUFFIX), arr(p, BBP + CALC_SUFFIX)
    mld = scalar(p, "MLD")
    below = (depth > mld) & np.isfinite(depth)
    sig = np.full_like(orig, np.nan)
    s_prof = np.clip(r + (1 - r) / (1 + (np.clip(par[below], 1e-3, None) / ipar_mid) ** e), r, 1.0)
    sig[below] = orig[below] / s_prof
    r_mld = np.nan
    for k in np.where(below)[0][np.argsort(depth[below])]:
        if np.isfinite(sig[k]) and np.isfinite(chl_calc[k]) and np.isfinite(bbp_calc[k]) and bbp_calc[k] > 0:
            r_mld = sig[k] / bbp_calc[k]
            break
    above = np.where(depth <= mld, bbp * r_mld, np.nan)
    corr = step._apply_xing_terrats(p, hybrid=True)
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    fig_spec.points(ax, orig, depth, color=C_ORIG, label="Original " + CHL)
    fig_spec.points(ax, sig, depth, color=C_RATIO, label="sigmoid de-quenched (below MLD)")
    line(ax, above, depth, color=C_RECON, lw=3, label=r"$b_{bp} \times R_{MLD}$ (above MLD)")
    corrected_points(ax, orig, corr, depth)
    hline(ax, mld, "MLD")
    hline(ax, scalar(p, "Z_IPAR"), r"$Z_{IPAR}$", color="0.5")
    finish_profile(ax, zmax_for(mld, scalar(p, "Z_IPAR")), CHL_LABEL)
    fig_spec.finish(fig, f"Terrats et al. (2020) X18_S08 — {stamp(step, pn_shallow)}")
    save(fig, CHLA_DIR, "npq_terrats_profile.png")


def fig_diagnostics(step):
    # the step's own diagnostics panels, drawn one per figure
    fig = plt.figure(figsize=(fig_spec.FIG_W, 11), dpi=fig_spec.DPI)
    gs = fig.add_gridspec(1, 1, left=0.08, right=0.97, top=0.94, bottom=0.05)
    step._draw_method_comparison(fig, gs[0, 0])
    fig.suptitle("Day vs night CHLA by method (surface depth-bin medians)",
                 fontsize=fig_spec.FS_SUPTITLE, fontweight="bold")
    save(fig, CHLA_DIR, "npq_method_comparison.png")

    fig = plt.figure(figsize=(fig_spec.FIG_W, 9), dpi=fig_spec.DPI)
    gs = fig.add_gridspec(1, 1, left=0.08, right=0.95, top=0.94, bottom=0.08)
    step._draw_timeseries(fig, gs[0, 0])
    fig.suptitle(f"CHLA Quenching — method: {step.method}", fontsize=fig_spec.FS_SUPTITLE,
                 fontweight="bold")
    save(fig, CHLA_DIR, "npq_timeseries.png")

    day_pn, _ = step._example_profiles()
    fig, axes = fig_spec.new_fig()
    step._draw_profile_change(axes[0][0], day_pn, f"Example day profile (#{day_pn}), {step.method}")
    fig_spec.finish(fig)
    save(fig, CHLA_DIR, "npq_example_profile.png")


def fig_bbp(ds):
    beta = ds["BETA_BACKSCATTERING700"].values
    bbp = ds["BBP700"].values
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    ax.boxplot([beta[np.isfinite(beta)], bbp[np.isfinite(bbp)]], patch_artist=True,
               showfliers=False, boxprops=dict(facecolor=fig_spec.CATEGORY[1], alpha=0.6))
    ax.set_xticks([1, 2], ["BETA_BACKSCATTERING700 [m-1 sr-1]", "BBP700 [m-1]"])
    fig_spec.style_axes(ax, ylabel="value")
    fig_spec.finish(fig, "BBP from Beta — input vs output (whiskers 1.5 IQR, outliers hidden)")
    save(fig, BBP_DIR, "bbp_from_beta_boxplot.png")

    time = ds["TIME"].values
    t0 = pd.Timestamp(time[np.isfinite(bbp)][0]) + pd.Timedelta(days=20)
    win = (time >= np.datetime64(t0)) & (time < np.datetime64(t0 + pd.Timedelta(days=3)))
    raw, base, spk = (ds[v].values[win] for v in ("BBP700", "BBP700_BASELINE", "BBP700_SPIKES"))
    t = time[win]
    fig, axes = fig_spec.new_fig(nrows=2, sharex=True, height_ratios=(2, 1))
    ax1, ax2 = axes[0][0], axes[1][0]
    ok = np.isfinite(raw)
    ax1.plot(t[ok], raw[ok], ls="--", lw=0.8, color=fig_spec.FLAGGED, label="Raw BBP700")
    ok = np.isfinite(base)
    ax1.plot(t[ok], base[ok], color=fig_spec.CATEGORY[1], alpha=fig_spec.ALPHA, label="Baseline")
    ok = np.isfinite(spk)
    fig_spec.points(ax2, t[ok], spk[ok], color=fig_spec.CATEGORY[2], label="Spikes")
    for ax in (ax1, ax2):
        fig_spec.date_axis(ax, which="x", index=t)
        fig_spec.legend(ax)
    fig_spec.style_axes(ax1, ylabel=fig_spec.axis_label("BBP700", "m-1"))
    fig_spec.style_axes(ax2, xlabel="Time", ylabel=fig_spec.axis_label("BBP700", "m-1"))
    fig_spec.finish(fig, "Isolate BBP Spikes — baseline & spikes (3-day window, median, window_size 50)")
    save(fig, BBP_DIR, "bbp_baseline_spikes_timeseries.png")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    fig_spec.MAX_POINTS = 100_000

    ds = processed_data(args.cache)
    fig_bbp(ds)

    steps = {m: run_method(ds, m) for m in
             ("xing2012", "biermann2015", "sackmann2008", "swart2015", "hemsley2015")}
    # Thomalla anchors its quenching depth in the top 5 m, which the surface
    # flagging (flag 3) excludes from calculations; let flag 3 through for it.
    steps["thomalla2018"] = run_method(
        ds, "thomalla2018", qc_handling_settings={"calculation_flag_filter": [4, 9]})
    x18 = run_method(ds, "xing2018", diagnostics=True)

    pn = pick_day_profile(steps["xing2012"])
    pn_deep = pick_day_profile(x18, regime="deep")
    pn_shallow = pick_day_profile(x18, regime="shallow", need_par=True)
    print("example profile", pn, "| deep", pn_deep, "| shallow", pn_shallow,
          "| hybrid effective:", x18._effective_hybrid)

    fig_sackmann(steps["sackmann2008"], pn)
    fig_xing2012(steps["xing2012"], pn)
    fig_biermann(steps["biermann2015"], pn)
    fig_hemsley(steps["hemsley2015"], pn)
    fig_swart(steps["swart2015"], pn)
    pn_t = pick_day_profile(steps["thomalla2018"], require_change=True) or pn
    print("thomalla profile", pn_t)
    fig_thomalla(steps["thomalla2018"], pn_t)
    fig_mitchell(steps["thomalla2018"], pn_t)
    fig_xing2018(x18, pn)
    fig_terrats(x18, pn_deep, pn_shallow)
    fig_diagnostics(x18)


if __name__ == "__main__":
    main()
