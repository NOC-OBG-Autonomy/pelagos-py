<p align="center">
  <img src="https://raw.githubusercontent.com/NOC-OBG-Autonomy/pelagos-py/main/docs/_static/NOC_logo.svg" alt="NOC logo" width="120" />
</p>

# pelagos-py

pelagos-py processes glider and other autonomous-platform data in OG1 format into science-ready datasets. A pipeline is a sequence of steps (loading, quality control, corrections, export) built from a single YAML config file, so processing can be reproduced by sharing the config.

![pelagos-py building, running and exporting a pipeline](https://raw.githubusercontent.com/NOC-OBG-Autonomy/pelagos-py/main/docs/_static/pelagos_demo.gif)

## Installation

```bash
pip install pelagos-py
```

Or from source:

```bash
git clone https://github.com/NOC-OBG-Autonomy/pelagos-py.git
cd pelagos-py
pip install -e .
```

## Quick start

Build a config for the demo file (downloaded on first use) and run it:

```python
from pelagos_py import Pipeline, get_demo_file

pipeline = Pipeline.make_config(get_demo_file())
pipeline.run()
```

`make_config` saves the config as a YAML file next to the data, ready to edit and rerun with `Pipeline.load_config`.

## Documentation

- [Getting Started](https://noc-obg-autonomy.github.io/pelagos-py/getting_started.html): installation, running a pipeline and an example config
- [User Guide](https://noc-obg-autonomy.github.io/pelagos-py/user_guide.html): how each variable is processed
- [API Reference](https://noc-obg-autonomy.github.io/pelagos-py/api_reference.html): every step and its parameters
- [Contributing](https://github.com/NOC-OBG-Autonomy/pelagos-py/blob/main/CONTRIBUTING.md): reporting issues and developing pelagos-py

## License

[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

Copyright 2025-2026 The National Oceanography Centre and contributors.
