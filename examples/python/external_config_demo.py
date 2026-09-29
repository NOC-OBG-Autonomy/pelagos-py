from pelagos_py.pipeline import Pipeline

# Build a config for the demo file from the default template and run it. The config is
# saved next to the file (Churchill_647.yaml), so it can be edited and rerun with
# Pipeline.load_config("examples/data/OG1/Churchill_647.yaml").run()
Pipeline.make_config("examples/data/OG1/Churchill_647.nc").run()
