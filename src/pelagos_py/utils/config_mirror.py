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

# config_mirror.py
import os, yaml, json


class ConfigMirrorMixin:
    """
    Private, canonical config in self._parameters.
    Selected public attributes are mirrored to/from _parameters.

    - Call self._init_config_mirror() once in __init__ of subclasses.
    - Use load_config[_from_file]() to populate _parameters.
    - set mirror keys with _reset_parameter_bridge([...]).
    """

    def _init_config_mirror(self):
        object.__setattr__(self, "_parameters", {})  # canonical store
        object.__setattr__(
            self, "_param_attr_keys", set()
        )  # names mirrored as attributes

    # ---- write-through attribute mirroring ----
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name.startswith("_"):
            return
        try:
            param_keys = object.__getattribute__(self, "_param_attr_keys")
            parameters = object.__getattribute__(self, "_parameters")
        except Exception:
            return
        if isinstance(param_keys, set) and name in param_keys:
            parameters[name] = value

    # ---- bridge helpers ----
    def _sync_parameters_to_attributes(self):
        for key in list(self._param_attr_keys):
            if key in self._parameters:
                object.__setattr__(self, key, self._parameters[key])

    def _sync_attributes_to_parameters(self):
        for key in list(self._param_attr_keys):
            if hasattr(self, key):
                self._parameters[key] = getattr(self, key)

    def _reset_parameter_bridge(self, mirror_keys=None):
        """
        mirror_keys: list of top-level keys in _parameters that should also exist as attributes.
        If None, will mirror intersection of existing public attributes and _parameters keys.
        """
        if mirror_keys is None:
            pub_attr_names = {
                k
                for k in dir(self)
                if not k.startswith("_") and not callable(getattr(self, k, None))
            }
            mirror_keys = sorted(pub_attr_names & set(self._parameters.keys()))
        object.__setattr__(self, "_param_attr_keys", set(mirror_keys))
        self._sync_parameters_to_attributes()

    # ---- load / save ----
    def load_config(self, config_dict: dict, mirror_keys=None):
        self._parameters = dict(config_dict or {})
        self._reset_parameter_bridge(mirror_keys)

    def load_config_from_file(self, path: str, mirror_keys=None):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        self.load_config(cfg, mirror_keys=mirror_keys)
        return cfg

    def save_config(self, path: str):
        self._sync_attributes_to_parameters()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self._parameters, f, sort_keys=False)
        print(f"[Config] Saved → {path}")

    # ---- dot-path getters/setters for nested values ----
    def get_param(self, keypath: str, default=None):
        node = self._parameters
        for part in keypath.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_param(self, keypath: str, value, create=True):
        parts = keypath.split(".")
        node = self._parameters
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                if not create:
                    raise KeyError(f"Missing path '{p}' in '{keypath}'")
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
        # If top-level element is mirrored, refresh attribute
        top = parts[0]
        if top in self._param_attr_keys:
            object.__setattr__(self, top, self._parameters[top])

    # Optional pretty dump (debug)
    def dump_config_json(self) -> str:
        self._sync_attributes_to_parameters()
        return json.dumps(self._parameters, indent=2)
