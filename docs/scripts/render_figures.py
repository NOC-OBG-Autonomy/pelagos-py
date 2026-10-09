"""Render the user-guide figures listed in figures.yaml from a pipeline run.

    python docs/scripts/render_figures.py [docs/scripts/figures.yaml] [--data FILE] [--out docs/_static]

--data overrides the Load OG1 file_path; --profile replaces PROFILE in figures.yaml.
"""
import argparse
import os
import shutil
import sys
import tempfile

import yaml

from pelagos_py.pipeline import Pipeline
from pelagos_py.steps import create_step
from pelagos_py.utils import diagnostic_capture

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _safe(step_name):
    return step_name.replace(os.sep, "_").replace(" ", "_")


def _find(captured, step, figure, index=None):
    # index pins a variant's own capture; pipeline figures match on the step name
    prefix = "" if index is None else f"{index:02d}_"
    name = f"{prefix}{_safe(step)}_{figure}.png"
    hits = sorted(f for f in os.listdir(captured) if f.endswith(name))
    if not hits:
        sys.exit(f"no captured figure for step '{step}', figure '{figure}'")
    return os.path.join(captured, hits[0])


def _fill_profile(obj, profile):
    if isinstance(obj, dict):
        return {k: _fill_profile(v, profile) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill_profile(v, profile) for v in obj]
    return profile if obj == "PROFILE" else obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default=os.path.join(ROOT, "docs", "scripts", "figures.yaml"))
    ap.add_argument("--data", default=os.environ.get("PELAGOS_DOCS_DATA"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "_static"))
    ap.add_argument("--profile", type=int, default=None)
    args = ap.parse_args()

    with open(args.spec) as fh:
        spec = yaml.safe_load(fh)
    with open(os.path.join(ROOT, spec["pipeline"])) as fh:
        config = yaml.safe_load(fh)
    for step in config["steps"]:
        if step["name"] == "Load OG1":
            if args.data:
                step["parameters"]["file_path"] = args.data
            elif not os.path.isabs(step["parameters"]["file_path"]):
                step["parameters"]["file_path"] = os.path.join(ROOT, step["parameters"]["file_path"])

    captured = tempfile.mkdtemp(prefix="pelagos_docs_figs_")
    with diagnostic_capture.force_headless_backend():
        pipe = Pipeline(config=config)
        pipe.headless = True
        pipe._capture_diagnostics, pipe._captured_figures, pipe._capture_dir = True, [], captured
        pipe.run()
        ds = pipe.get_data()

        profile = args.profile
        for i, variant in enumerate(spec.get("variants", [])):
            if profile is None:
                profile = _example_profile(ds)
            variant = _fill_profile(variant, profile)
            step_config = {"name": variant["step"], "parameters": variant.get("parameters", {}),
                           "diagnostics": list(variant["figures"])}
            with diagnostic_capture.capture_figures(captured, variant["step"], 100 + i, []):
                create_step(step_config, {"data": ds.copy(), "global_parameters": {}}).run()
        diagnostic_capture.wait_for_saves()

    outputs = {path: (v["step"], v["figure"], None) for path, v in spec.get("figures", {}).items()}
    for i, variant in enumerate(spec.get("variants", [])):
        for figure, path in variant["figures"].items():
            outputs[path] = (variant["step"], figure, 100 + i)
    for path, (step, figure, index) in outputs.items():
        dest = os.path.join(args.out, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(_find(captured, step, figure, index), dest)
        print("wrote", os.path.relpath(dest, ROOT))
    shutil.rmtree(captured, ignore_errors=True)


def _example_profile(ds):
    # the day profile xing2012 changes most: the same profile every variant annotates
    step = create_step({"name": "CHLA Quenching", "diagnostics": False,
                        "parameters": {"method": "xing2012", "apply_to": "CHLA_ADJUSTED"}},
                       {"data": ds.copy(), "global_parameters": {}})
    step.run()
    pn = step._example_profiles()[0]
    print("example profile", pn)
    return pn


if __name__ == "__main__":
    main()
