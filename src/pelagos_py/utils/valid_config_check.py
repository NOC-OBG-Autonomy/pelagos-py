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

from pelagos_py.steps import STEP_CLASSES, QC_CLASSES
from pelagos_py.utils import file_probe, parameter_spec
from pelagos_py.utils.qc_handling import adjusted_name, prefer_adjusted, raw_name

# A pipeline needs exactly one of these to supply its base data.
LOADER_STEP_NAMES = ("Load OG1", "Generate Data")

# Variables only a loader step can produce.
LOADER_PROVIDED_VARIABLES = {"TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC"}

_NO_LOADER_HINT = (
    "No data-loading step ('Load OG1' or 'Generate Data') provides it -- add "
    "one, normally as the first step in the pipeline."
)


def _loader_hint(missing):
    return (
        _NO_LOADER_HINT
        if any(v in LOADER_PROVIDED_VARIABLES for v in missing)
        else None
    )


def _variable_parameter_names(step_class, parameters, include_optional=False):
    # input variables named by parameters (e.g. apply_to: "BBP700"); output_as and
    # variable_parameters_optional are skipped as they aren't always required
    schema = getattr(step_class, "parameter_schema", None) or {}
    optional = getattr(step_class, "variable_parameters_optional", ())
    names = []
    for attr in getattr(step_class, "variable_parameters", []):
        if attr == "output_as" or (attr in optional and not include_optional):
            continue
        default = (schema.get(attr) or {}).get("default")
        value = parameters.get(attr, default)
        if value is None:
            continue
        names.extend(list(value) if isinstance(value, (list, tuple, dict)) else [value])
    return names


def _read_file_variables(file_path, logger):
    # (None, None) if the file can't be read; callers then assume file variables exist
    probe = file_probe.probe_file(file_path, logger)
    if probe is None:
        return None, None
    return set(probe), {v for v, info in probe.items() if info["all_nan"]}


def _prepare_outputs(steps_list, file_vars):
    # variables a "Prepare OG1" step will rename into existence
    prep = next(
        (
            s
            for s in steps_list
            if isinstance(s, dict) and s.get("name") == "Prepare OG1"
        ),
        None,
    )
    if prep is None or file_vars is None:
        return set()
    cls = STEP_CLASSES.get("Prepare OG1")
    return set(cls.renames_for(file_vars, **(prep.get("parameters") or {})).values())


def _raise_missing_variables(
    logger,
    kind,
    label,
    missing,
    later_available,
    known_derived,
    file_vars,
    file_all_nan=None,
):
    # Raise the most specific error: produced later, produced by no step, or not
    # (or only as all-NaN) in the input file. Otherwise left for the run-time check.
    # later_available and known_derived hold raw names, so X_ADJUSTED counts for X.
    out_of_order = [v for v in missing if raw_name(v) in later_available]
    if out_of_order:
        missing_str = ", ".join(out_of_order)
        logger.error(
            "Validation Failed: %s '%s' requires %s, but it is produced by a "
            "later step. Reorder the pipeline so the producing step runs first.",
            kind,
            label,
            missing_str,
        )
        raise ValueError(
            f"Missing variables for {kind} '{label}': {missing_str}. These "
            f"are produced later in the pipeline — reorder the steps so they "
            f"run beforehand."
        )

    not_produced = [
        v
        for v in missing
        if raw_name(v) not in later_available and raw_name(v) in known_derived
    ]
    if not_produced:
        missing_str = ", ".join(not_produced)
        hint = _loader_hint(not_produced) or (
            "Add the step that derives it (e.g. 'Find Profiles' for "
            "PROFILE_NUMBER/PROFILE_DIRECTION)."
        )
        logger.error(
            "Validation Failed: %s '%s' requires %s, but no step in the "
            "pipeline produces it. %s",
            kind,
            label,
            missing_str,
            hint,
        )
        raise ValueError(
            f"Missing variables for {kind} '{label}': {missing_str}. No step "
            f"in the pipeline produces them. {hint}"
        )

    if file_vars is not None:
        unverified = [
            v
            for v in missing
            if raw_name(v) not in later_available and raw_name(v) not in known_derived
        ]
        missing_from_file = [
            v for v in unverified if prefer_adjusted(v, file_vars) not in file_vars
        ]
        if missing_from_file:
            missing_str = ", ".join(missing_from_file)
            logger.error(
                "Validation Failed: %s '%s' requires %s, but the input file "
                "does not contain it.",
                kind,
                label,
                missing_str,
            )
            raise ValueError(
                f"Missing variables for {kind} '{label}': {missing_str}. The "
                f"input file does not contain them."
            )

        if file_all_nan:
            all_nan_present = [v for v in unverified if v in file_all_nan]
            if all_nan_present:
                missing_str = ", ".join(all_nan_present)
                logger.error(
                    "Validation Failed: %s '%s' requires %s, but the input "
                    "file only contains placeholder (all-NaN) data for it.",
                    kind,
                    label,
                    missing_str,
                )
                raise ValueError(
                    f"Variable(s) required for {kind} '{label}': {missing_str} "
                    f"are present in the input file but contain only NaN "
                    f"(placeholder) data."
                )


def _missing_required_params(schema, parameters):
    """Names of required schema parameters absent from the supplied config.

    ``schema`` of ``None`` (a component not yet on the parameter schema, e.g. the
    oxygen steps) is treated as "no required parameters".
    """
    if not schema:
        return []
    return [
        name
        for name, spec in schema.items()
        if parameter_spec.is_required(spec) and name not in parameters
    ]


def _unknown_params(schema, parameters, allowed_extra=()):
    """Names of supplied parameters not declared in the schema.

    Mirrors the reject-unknown behaviour of :func:`parameter_spec.resolve`, but
    runs up front so config typos are caught before any step executes. ``schema``
    of ``None`` (a component not yet on the parameter schema) skips the check; an
    empty ``{}`` schema is strict, so any supplied parameter is unknown.
    ``allowed_extra`` permits framework keys (e.g. ``qc_handling_settings``).
    """
    if schema is None:
        return []
    return [
        name for name in parameters if name not in schema and name not in allowed_extra
    ]


def _qc_test_io(qc_class, qc_params):
    """Resolve a QC test's required and provided variables from its parameters.

    Mirrors how Apply QC resolves them at run time: dynamic tests derive their
    variables from the supplied parameters (so they are instantiated with no data
    to introspect), while static tests expose them as class attributes.
    """
    if getattr(qc_class, "dynamic", False):
        # `diagnostics` is a reserved per-test flag, not a QC parameter.
        params = {k: v for k, v in (qc_params or {}).items() if k != "diagnostics"}
        probe = qc_class(None, **params)
        return list(probe.required_variables), list(probe.qc_outputs)
    return (
        list(getattr(qc_class, "required_variables", [])),
        list(getattr(qc_class, "qc_outputs", [])),
    )


def _step_io(step_config):
    # (inputs, outputs) one step config names, before any _ADJUSTED resolution
    step_class = STEP_CLASSES.get(step_config["name"])
    if not step_class:
        return [], []
    parameters = step_config.get("parameters", {}) or {}
    inputs = list(getattr(step_class, "required_variables", []))
    inputs += _variable_parameter_names(step_class, parameters, include_optional=True)
    try:
        outputs = list(step_class.output_variables(parameters))
    except (TypeError, ValueError, AttributeError):
        # malformed parameters are reported by the per-step validation below
        outputs = []
    if step_config["name"] == "Apply QC":
        for qc_name, qc_params in (parameters.get("qc_settings") or {}).items():
            qc_class = QC_CLASSES.get(qc_name)
            if qc_class is None:
                continue
            try:
                qc_required, qc_outputs = _qc_test_io(qc_class, qc_params)
            except Exception:
                continue
            inputs += qc_required
            outputs += qc_outputs
    return inputs, outputs


def _made_available_after(step_io, index):
    # raw names that steps after `index` bring into existence; a step rewriting its
    # own input (BBP700 -> BBP700_ADJUSTED) doesn't make that input available
    names = set()
    for inputs, outputs in step_io[index + 1 :]:
        read = {raw_name(v) for v in inputs}
        names.update(raw_name(v) for v in outputs if raw_name(v) not in read)
    return names


def _later_producer(var, steps_list, step_io, index, skip_corrections=False):
    # skip_corrections: ignore steps that correct raw var, as those carry its QC flags over
    for later in range(index + 1, len(steps_list)):
        inputs, outputs = step_io[later]
        if var not in outputs:
            continue
        if skip_corrections and raw_name(var) in {raw_name(v) for v in inputs}:
            continue
        return steps_list[later]["name"]
    return None


def _warn_adjusted_resolution(
    logger, kind, label, names, available_vars, steps_list, step_io, index
):
    # warn where a step will read raw values that have (or will have) an _ADJUSTED version
    for name in names:
        used = prefer_adjusted(name, available_vars)
        if used not in available_vars or used.endswith("_ADJUSTED"):
            continue
        if name.endswith("_ADJUSTED"):
            logger.warning(
                "%s '%s' names %s, but only %s exists at that point -- falls back to %s.",
                kind,
                label,
                name,
                used,
                used,
                extra={"step_index": index},
            )
        if adjusted_name(used) in step_io[index][1]:
            continue  # this step starts the X -> X_ADJUSTED chain itself
        producer = _later_producer(
            adjusted_name(used),
            steps_list,
            step_io,
            index,
            skip_corrections=kind == "QC test",
        )
        if producer:
            logger.warning(
                "%s '%s' uses raw %s; %s is produced later by '%s' -- move it "
                "earlier to use adjusted values.",
                kind,
                label,
                used,
                adjusted_name(used),
                producer,
                extra={"step_index": index},
            )


def _non_loader_provided_variables(steps_list):
    # loaders claim TIME/LATITUDE/etc whatever the file holds, so leave them out
    provided = set()
    for step_config in steps_list:
        if (
            isinstance(step_config, dict)
            and step_config.get("name") in LOADER_STEP_NAMES
        ):
            continue
        provided.update(_step_io(step_config)[1])
    return provided


def _known_derived_variables():
    # raw names of every variable any registered step can produce, in this pipeline or not
    known = set()
    for step_class in STEP_CLASSES.values():
        known.update(getattr(step_class, "provided_variables", []))
        known.update(getattr(step_class, "qc_outputs", []))
    for qc_class in QC_CLASSES.values():
        known.update(getattr(qc_class, "qc_outputs", []))
    return {raw_name(v) for v in known}


def check_pipeline_variables(steps_list, logger, available_vars=None):
    file_vars = None
    file_all_nan = None
    if available_vars is None:
        logger.info("Checking pipeline variable requirements...")
        # nothing exists until a loader step runs, so a pipeline without one is flagged
        available_vars = set()

        loader_steps = [
            (i, s["name"])
            for i, s in enumerate(steps_list)
            if isinstance(s, dict) and s.get("name") in LOADER_STEP_NAMES
        ]
        if len(loader_steps) > 1:
            where = ", ".join(f"step {i + 1} ('{n}')" for i, n in loader_steps)
            logger.error(
                "Validation Failed: multiple data-loading steps found: %s. "
                "Only one step should load or generate the pipeline's base "
                "data -- remove the extra one.",
                where,
            )
            raise ValueError(
                f"Multiple data-loading steps found: {where}. Keep only one "
                f"'Load OG1' or 'Generate Data' step."
            )
        if len(loader_steps) == 1 and loader_steps[0][1] == "Load OG1":
            idx, _ = loader_steps[0]
            file_path = (steps_list[idx].get("parameters") or {}).get("file_path")
            if not file_path or not str(file_path).strip():
                logger.error(
                    "Validation Failed: 'Load OG1' has no 'file_path' set -- "
                    "this config does not include a data file. Set "
                    "'file_path' to your input NetCDF file before running."
                )
                exc = ValueError(
                    "'Load OG1' has no 'file_path' set -- this config does "
                    "not include a data file. Set 'file_path' to your input "
                    "NetCDF file before running."
                )
                exc.step_index = idx
                raise exc
            else:
                # with a real path, also check against what the file actually holds
                file_vars, file_all_nan = _read_file_variables(file_path, logger)
                if file_vars is not None:
                    # a later step can supply it under the right name (e.g. LATITUDE_GPS -> LATITUDE)
                    real_vars = file_vars - (file_all_nan or set())
                    other_provided = _non_loader_provided_variables(steps_list)
                    other_provided |= _prepare_outputs(steps_list, real_vars)
                    missing_base = sorted(
                        v
                        for v in LOADER_PROVIDED_VARIABLES
                        if v not in file_vars and v not in other_provided
                    )
                    if missing_base:
                        missing_str = ", ".join(missing_base)
                        logger.error(
                            "Validation Failed: 'Load OG1' file '%s' does not "
                            "contain %s, which every OG1-format file is "
                            "expected to provide.",
                            file_path,
                            missing_str,
                        )
                        exc = ValueError(
                            f"'Load OG1' file '{file_path}' does not contain "
                            f"{missing_str}, which every OG1-format file is "
                            f"expected to provide."
                        )
                        exc.step_index = idx
                        raise exc

    step_io = [_step_io(step_config) for step_config in steps_list]
    known_derived = _known_derived_variables()

    for index, step_config in enumerate(steps_list):
        try:
            step_name = step_config["name"]

            step_class = STEP_CLASSES.get(step_name)
            if not step_class:
                continue

            parameters = step_config.get("parameters", {}) or {}
            schema = getattr(step_class, "parameter_schema", None)
            allowed_extra = getattr(step_class, "framework_parameters", set())

            # Check for missing required parameters, driven by the declared schema.
            missing_params = _missing_required_params(schema, parameters)
            if missing_params:
                missing_str = ", ".join(missing_params)
                logger.error(
                    "Validation Failed: '%s' is missing required config parameters: %s.",
                    step_name,
                    missing_str,
                )
                raise ValueError(
                    f"Missing config parameters for '{step_name}': {missing_str}."
                )

            # Check for unknown parameters (config typos), driven by the same schema.
            unknown_params = _unknown_params(schema, parameters, allowed_extra)
            if unknown_params:
                unknown_str = ", ".join(unknown_params)
                valid_str = ", ".join(sorted(schema)) or "(none)"
                logger.error(
                    "Validation Failed: '%s' has unknown config parameters: %s. "
                    "Valid parameters: %s.",
                    step_name,
                    unknown_str,
                    valid_str,
                )
                raise ValueError(
                    f"Unknown config parameters for '{step_name}': {unknown_str}. "
                    f"Valid parameters: {valid_str}."
                )

            # Check for type mismatches (e.g. a bool where a float is expected).
            if schema is not None:
                bad_types = parameter_spec.type_errors(schema, parameters)
                if bad_types:
                    bad_str = "; ".join(bad_types)
                    logger.error(
                        "Validation Failed: '%s' has invalid parameter type(s): %s.",
                        step_name,
                        bad_str,
                    )
                    raise ValueError(
                        f"Invalid parameter type(s) for '{step_name}': {bad_str}."
                    )

            # Check for out-of-options values (e.g. an unknown 'method' choice).
            if schema is not None:
                bad_options = parameter_spec.option_errors(schema, parameters)
                if bad_options:
                    bad_str = "; ".join(bad_options)
                    logger.error(
                        "Validation Failed: '%s' has invalid parameter value(s): %s.",
                        step_name,
                        bad_str,
                    )
                    raise ValueError(
                        f"Invalid parameter value(s) for '{step_name}': {bad_str}."
                    )

            # Apply QC nests each test's settings under qc_settings — validate the
            # required parameters of every requested test up front. (Their variable
            # requirements are checked by Apply QC at run time, where _QC columns and
            # also_flag propagation are resolved.)
            if step_name == "Apply QC":
                for qc_name, qc_params in (parameters.get("qc_settings") or {}).items():
                    # `diagnostics` is a per-test flag, not a QC parameter
                    qc_params = {
                        k: v for k, v in (qc_params or {}).items() if k != "diagnostics"
                    }
                    qc_class = QC_CLASSES.get(qc_name)
                    if qc_class is None:
                        continue  # Apply QC raises a clear error for unknown tests at run time
                    qc_schema = getattr(qc_class, "parameter_schema", None)
                    qc_allowed_extra = getattr(qc_class, "framework_parameters", set())
                    qc_missing = _missing_required_params(qc_schema, qc_params or {})
                    if qc_missing:
                        missing_str = ", ".join(qc_missing)
                        logger.error(
                            "Validation Failed: QC test '%s' is missing required parameters: %s.",
                            qc_name,
                            missing_str,
                        )
                        raise ValueError(
                            f"Missing config parameters for QC test '{qc_name}': {missing_str}."
                        )

                    qc_unknown = _unknown_params(
                        qc_schema, qc_params or {}, qc_allowed_extra
                    )
                    if qc_unknown:
                        unknown_str = ", ".join(qc_unknown)
                        valid_str = ", ".join(sorted(qc_schema)) or "(none)"
                        logger.error(
                            "Validation Failed: QC test '%s' has unknown parameters: %s. "
                            "Valid parameters: %s.",
                            qc_name,
                            unknown_str,
                            valid_str,
                        )
                        raise ValueError(
                            f"Unknown config parameters for QC test '{qc_name}': {unknown_str}. "
                            f"Valid parameters: {valid_str}."
                        )

                    if qc_schema is not None:
                        qc_bad_types = parameter_spec.type_errors(
                            qc_schema, qc_params or {}
                        )
                        if qc_bad_types:
                            bad_str = "; ".join(qc_bad_types)
                            logger.error(
                                "Validation Failed: QC test '%s' has invalid parameter type(s): %s.",
                                qc_name,
                                bad_str,
                            )
                            raise ValueError(
                                f"Invalid parameter type(s) for QC test '{qc_name}': {bad_str}."
                            )

                    if qc_schema is not None:
                        qc_bad_options = parameter_spec.option_errors(
                            qc_schema, qc_params or {}
                        )
                        if qc_bad_options:
                            bad_str = "; ".join(qc_bad_options)
                            logger.error(
                                "Validation Failed: QC test '%s' has invalid parameter value(s): %s.",
                                qc_name,
                                bad_str,
                            )
                            raise ValueError(
                                f"Invalid parameter value(s) for QC test '{qc_name}': {bad_str}."
                            )

                    # resolve the variables the same way Apply QC does at run time
                    configured, _ = _qc_test_io(qc_class, qc_params)
                    qc_params = prefer_adjusted(qc_params, available_vars)
                    qc_required, qc_outputs = _qc_test_io(qc_class, qc_params)
                    qc_missing = [v for v in qc_required if v not in available_vars]
                    if qc_missing:
                        _raise_missing_variables(
                            logger,
                            "QC test",
                            qc_name,
                            qc_missing,
                            _made_available_after(step_io, index),
                            known_derived,
                            file_vars,
                            file_all_nan,
                        )
                    _warn_adjusted_resolution(
                        logger,
                        "QC test",
                        qc_name,
                        configured,
                        available_vars,
                        steps_list,
                        step_io,
                        index,
                    )

                    # Make this test's outputs available to later tests in the same
                    # Apply QC call, so a test that legitimately depends on an earlier
                    # test's output (e.g. a profile-level test needing TEMP_QC) is not
                    # falsely flagged as depending on a later step.
                    available_vars.update(qc_outputs)

            req_vars = list(getattr(step_class, "required_variables", []))
            req_vars.extend(_variable_parameter_names(step_class, parameters))

            own_provided = set(step_class.output_variables(parameters))
            if step_name == "Prepare OG1" and file_vars is not None:
                own_provided.update(
                    _prepare_outputs([step_config], file_vars - (file_all_nan or set()))
                )

            # an `optional: true` step skips when its target_variable is absent
            if parameters.get("optional") and file_vars is not None:
                target = parameters.get("target_variable")
                present = available_vars | file_vars
                if target and prefer_adjusted(target, present) not in present:
                    own_provided = set()
                    req_vars = [v for v in req_vars if v != target]

            missing_vars = [
                req
                for req in req_vars
                if prefer_adjusted(req, available_vars) not in available_vars
            ]

            if missing_vars:
                _raise_missing_variables(
                    logger,
                    "step",
                    step_name,
                    missing_vars,
                    _made_available_after(step_io, index),
                    known_derived,
                    file_vars,
                    file_all_nan,
                )
            _warn_adjusted_resolution(
                logger,
                "step",
                step_name,
                req_vars,
                available_vars,
                steps_list,
                step_io,
                index,
            )

            available_vars.update(own_provided)
            # all-NaN placeholders stay missing so the error can point at them
            if step_name == "Load OG1" and file_vars is not None:
                available_vars.update(file_vars - (file_all_nan or set()))

        except ValueError as exc:
            if not hasattr(exc, "step_index"):
                exc.step_index = index
            raise

    if steps_list:
        logger.info("Pipeline variable check successful.")

    return True
