"""Build a config for the demo glider file and run the pipeline on it.

If you have your own OG1 file, you can replace the get_demo_file call with its path.

If you have your own config, you can load it instead of making a new one. You can also
call make_config with ask=True to be prompted about each choice:

    Pipeline.make_config(file, ask=True)               # asking about each choice
    Pipeline.load_config("/path/to/your_config.yaml")  # a config you already have

make_config saves the config next to the data file (e.g. demo_OG1.yaml), so you can
edit it and rerun it with load_config.

The demo file (a month of NOC glider data from the North Atlantic, ~15 MB) is downloaded into
~/Documents/pelagos-py/demo_data the first time.
"""

from pelagos_py import Pipeline, get_demo_file

file = get_demo_file()
p = Pipeline.make_config(file)
p.run()
