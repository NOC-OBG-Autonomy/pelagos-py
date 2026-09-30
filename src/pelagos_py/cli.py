"""The ``pelagos-py`` terminal command: ``dashboard``, ``build``, ``run`` and ``demo``."""

import argparse
import subprocess
import sys
from pathlib import Path

from pelagos_py import Pipeline, get_demo_file

# The dashboard isn't packaged yet, so it only runs from a git clone.
REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_APP = REPO_ROOT / "dashboard" / "app.py"


def dashboard(args):
    if not DASHBOARD_APP.exists():
        sys.exit("The dashboard only runs from a git clone of pelagos-py for now.")
    try:
        subprocess.run([sys.executable, str(DASHBOARD_APP)], cwd=REPO_ROOT)
    except KeyboardInterrupt:
        pass


def build(args):
    Pipeline.make_config(args.file, ask=args.ask, config_path=args.output)


def run(args):
    if Path(args.path).suffix in (".yaml", ".yml"):
        pipeline = Pipeline.load_config(args.path)
    else:
        pipeline = Pipeline.make_config(args.path)
    pipeline.run()


def demo(args):
    path = get_demo_file(args.name)
    if path is not None:
        print(path)


def main():
    parser = argparse.ArgumentParser(prog="pelagos-py", description="Process glider data with pelagos-py.")
    commands = parser.add_subparsers(metavar="command", required=True)

    dashboard_cmd = commands.add_parser("dashboard", help="open the config dashboard in your browser")
    dashboard_cmd.set_defaults(func=dashboard)

    build_cmd = commands.add_parser("build", help="write a config for an OG1 file (next to it, as <name>.yaml)")
    build_cmd.add_argument("file", help="OG1 NetCDF file")
    build_cmd.add_argument("--ask", action="store_true", help="choose each option instead of using the defaults")
    build_cmd.add_argument("-o", "--output", help="where to save the config")
    build_cmd.set_defaults(func=build)

    run_cmd = commands.add_parser("run", help="run a config (.yaml), or build one for an OG1 file and run it")
    run_cmd.add_argument("path", help="config .yaml or OG1 NetCDF file")
    run_cmd.set_defaults(func=run)

    demo_cmd = commands.add_parser("demo", help="list the demo datasets, or download one and print its path")
    demo_cmd.add_argument("name", nargs="?", help="demo to download, e.g. nelson_646_r")
    demo_cmd.set_defaults(func=demo)

    args = parser.parse_args()
    args.func(args)
