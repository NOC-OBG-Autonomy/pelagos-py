# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The ``pelagos-py`` terminal command: ``dashboard``, ``build``, ``run`` and ``demo``."""

import argparse
from pathlib import Path

from pelagos_py import Pipeline, dashboard as dashboard_app, get_demo_file


def dashboard(args):
    try:
        dashboard_app.run()
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
    print(get_demo_file())


def main():
    parser = argparse.ArgumentParser(
        prog="pelagos-py", description="Process glider data with pelagos-py."
    )
    commands = parser.add_subparsers(metavar="command")

    dashboard_cmd = commands.add_parser(
        "dashboard", help="open the config dashboard in your browser"
    )
    dashboard_cmd.set_defaults(func=dashboard)

    build_cmd = commands.add_parser(
        "build", help="write a config for an OG1 file (next to it, as <name>.yaml)"
    )
    build_cmd.add_argument("file", help="OG1 NetCDF file")
    build_cmd.add_argument(
        "--ask",
        action="store_true",
        help="choose each option instead of using the defaults",
    )
    build_cmd.add_argument("-o", "--output", help="where to save the config")
    build_cmd.set_defaults(func=build)

    run_cmd = commands.add_parser(
        "run", help="run a config (.yaml), or build one for an OG1 file and run it"
    )
    run_cmd.add_argument("path", help="config .yaml or OG1 NetCDF file")
    run_cmd.set_defaults(func=run)

    demo_cmd = commands.add_parser(
        "demo", help="download the demo dataset and print its path"
    )
    demo_cmd.set_defaults(func=demo)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)
