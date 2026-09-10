import yaml
from pelagos_py.pipeline import Pipeline

# Same pipeline as external_config_demo.py, but the config is handed over as a dict
# (Pipeline(config=...)) rather than a path: build or edit it in Python before running.
with open("examples/configs/example_config_nelson.yaml") as f:
    demo_config = yaml.safe_load(f)

demo_config["pipeline"]["description"] = "Nelson demo, run from an in-memory config dict"

p = Pipeline(config=demo_config)
p.run()
