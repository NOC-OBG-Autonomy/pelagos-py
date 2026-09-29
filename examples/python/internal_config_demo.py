import yaml
from pelagos_py.pipeline import Pipeline
from pelagos_py.utils import config_builder

# Pipeline(config=...) takes the config as a dict rather than a file, e.g. to change
# settings in code. This starts from the config make_config would build for the file.
file_path = "examples/data/OG1/Nelson_646_R.nc"
text = config_builder.build(config_builder.DEFAULT_CONFIG.read_text(), file_path)
demo_config = yaml.safe_load(text)
demo_config["pipeline"]["on_step_fail"] = "stop"

p = Pipeline(config=demo_config)
p.run()
