# pelagos_py dashboard

A local web tool to build, validate and run pipeline configs. The pipeline does
not depend on it: a config made here is an ordinary YAML file you can run any
other way.

The step list and parameter forms come from the live `STEP_CLASSES` /
`QC_CLASSES` registries and each step's `describe_parameters()`, so a new step
shows up on the next server start. Validation calls the pipeline's own
`parameter_spec.resolve()`, so the dashboard accepts exactly what the pipeline
accepts.

## Running

```bash
pip install -e ".[dashboard]"
pelagos-py dashboard
```

Or from Python: `from pelagos_py import dashboard; dashboard.run()`. Then open
<http://localhost:8791>.

Saved configs go in `~/Documents/pelagos-py/configs/` and demo data in
`~/Documents/pelagos-py/demo_data/`. The pipeline runs from
`~/Documents/pelagos-py`, so relative paths in a config resolve from there.

**Sections** group a run of steps under a name. They can be renamed, collapsed
and dragged as a block. They are saved as `# ==== TITLE ====` banner comments
in the YAML and read back from them, so hand-written configs like
`src/pelagos_py/default_config.yaml` keep their sections. The pipeline ignores
them.

## Tuning a step while it runs

While a run is going, the builder, pipeline settings and YAML pane are locked,
so the screen always matches what is running.

A step with `diagnostics: true` pauses the run after drawing its plot. The Run
tab shows that step's figure, and only that step's card in the builder unlocks.
Then:

1. look at the plot,
2. change a parameter on the unlocked card,
3. **Re-run step**: the step runs again from a copy of the data before it, and
   the previous attempt stays as a thumbnail,
4. repeat, then **Continue** (which locks the config again).

Reordering and removing steps stay locked, because the runner works from the
step indices it started with. The parameters you settle on are already in the
config, so saving keeps them. Every figure is also kept in the **Plots** tab,
grouped by step and attempt.

**Apply QC steps pause once per test.** When such a step would pause, the
runner splits it into one run per test (as you could write by hand in the
config). Each test gets its own pause and plot, and only its part of the QC
editor unlocks. Re-run replays just that test; earlier tests in the step keep
their flags. Runs that never pause are not split.

## Manual QC

The `manual qc` test (in an Apply QC step with diagnostics on) turns the pause
into an editor. Each entry in its `variables` (one variable, or a group such as
`[CHLA, [TEMP, PSAL]]`) gets a plot. "+ QC variable" adds one, and "Apply flag
to" in the sidebar edits it ("All variables" ticks every variable with data).
There are five views, drawn in the browser from the paused run's data:

- **Time**: PRES vs time, coloured by the variable
- **Profile**: PRES vs the variable, all samples or one profile/cycle at a time
- **Variable**: the variable vs time, coloured by flag
- **T–S**: salinity vs temperature with σ0 contours
- **Map**: latitude vs longitude

Drag a box (or click a point) to flag the samples inside or outside it with the
flag chosen in the sidebar. × removes a box. ⌘/Ctrl- or Shift-drag zooms and
double-click resets. The right panel has the flag legend (the eye hides a
flag) and the box list with undo/redo (⌘Z / ⇧⌘Z). Only the listed variables get
flags; untouched points become 1 (good) unless that switch is off.

Each box is saved in the test's `boxes` parameter as a 2D range test (`x: [lo,
hi]`, `y: [lo, hi]`, `flag`, `mode`, `variables`), so the saved config gives
the same flags wherever the pipeline runs.

## Interactive plots

Diagnostic figures are matplotlib PNGs. Next to each PNG the runner also tries
to write a **plot spec** (`fig_spec.py`): a JSON file with the figure's labels,
limits, layout and styling, plus a binary file with every trace's full x/y.
`plot.js` draws this with WebGL, without thinning the data. Drag to zoom,
double-click (or **Reset**) to go back, click a point for its exact values,
click a legend entry to hide a series. Shared axes stay linked. **Image** in the
viewer toolbar switches back to the PNG.

Steps don't need to change; they still draw with matplotlib and call
`plt.show()`. A figure is either fully interactive or PNG-only. `fig_spec.py`
handles lines and markers (including `datetime64` time series), scatter points
and `axhline`/`axvline`. Anything else (histograms, `fill_between`,
`pcolormesh`, map projections) keeps the figure PNG-only, so a plot is never
shown half-drawn. The run log says which, and why:

```
  · plot: TEMP Spike Test (interactive)
  · plot: Profile summary (image only — patches)
  · plot: Track map (image only — projection:mercator)
```

Interactive thumbnails have a blue rule. To make a PNG-only plot interactive,
either draw it with lines/scatter in the step, or teach `fig_spec.py` the
artist named in the log (add a branch to `_scatter_trace`/`_line_trace` and
remove it from `_unsupported`).

Data is sent as float32 (dates as seconds since the figure's first timestamp),
about 12 MB per 1.5M-point trace. A float64 copy stays on the server for exact
click values. Per figure the runner writes `fig_NNN.png`, `fig_NNN.json`
(spec), `fig_NNN.f32` (float32 data) and `fig_NNN_full.npz` (float64).

## How the runner talks to the dashboard

`run_bootstrap.py` runs the pipeline in a subprocess on the Agg backend.
It sends status as single lines on stdout (fields tab-separated) and reads
commands on stdin.

| Marker | Fields | Meaning |
|--------|--------|---------|
| `__PELAGOS_STEP__` | index, name, QC test | a step is starting |
| `__PELAGOS_FIG__` | PNG file, caption, spec file, reason | a figure was saved |
| `__PELAGOS_LOG__` | index, name, QC test, base64 text | a step printed diagnostics but drew no figure |
| `__PELAGOS_FAIL__` | index, name, QC test, base64 text | a step failed (`on_step_fail: pause`) |
| `__PELAGOS_PAUSE__` | index, name, QC test | waiting for a command |
| `__PELAGOS_RERUN__` | index | a re-run is starting |
| `__PELAGOS_VARS__` | JSON | variables for the Manual QC pickers |
| `__PELAGOS_DATA__` | request id | requested columns written to `cols_<id>.bin` |
| `__PELAGOS_REPORT__` | path, file name | a PDF report was written |
| `__PELAGOS_MEM__` | RSS, peaks, dataset size, step | RAM meter reading after a step |
| `__PELAGOS_TIME__` | active seconds, paused, epoch | processing clock |

Commands on stdin: `continue`, `skip`, `rerun <json parameters>` and
`data <json request>`. Stop sends SIGINT. The server adds its own
`__PELAGOS_SAMPLE__` lines (time, RSS) for the RAM meter.

## Layout

| File | Role |
|------|------|
| `__init__.py` | `run()`: checks the extra is installed and starts the server |
| `app.py` | FastAPI backend: registry, validation, configs, demos, outputs, runs and the log stream |
| `run_bootstrap.py` | Subprocess that runs the pipeline and captures each `plt.show()` figure |
| `fig_spec.py` | matplotlib figure → plot spec + float32 data |
| `static/index.html` | Page shell |
| `static/js/app.js` | Start-up and wiring |
| `static/js/api.js` | Backend fetch wrappers |
| `static/js/forms.js` | Schema → form fields |
| `static/js/builder.js` | Step picker, pipeline flow list, sections, QC editor |
| `static/js/defaults.js` | Differences from the default pipeline, so the builder can flag and undo edits |
| `static/js/config.js` | Builder ⇄ YAML, save/load |
| `static/js/build.js` | Build panel: what the template must change for a picked file, then builds the config (`pelagos_py.utils.config_builder`) |
| `static/js/run.js` | Run, log console, captured figures |
| `static/js/review.js` | Paused-step panel: plots, parameters, re-run |
| `static/js/viewer.js` | Full-window figure viewer |
| `static/js/plot.js` | WebGL renderer: spec + binary → zoomable panels |
| `static/js/mem.js` | RAM meter from `__PELAGOS_MEM__` / `__PELAGOS_SAMPLE__` |
| `static/js/manual.js` | Manual QC tab |
| `static/js/inspect.js` | Inspect tab: variables, sensors and attributes of the input file |
| `static/js/demos.js` | Files tab: your input files and demo deployments, grouped by mission |
| `static/js/outputs.js` | Output files from runs (reports, exports, logs), with open/delete |
| `static/js/icons.js` | Inline SVG icons |
| `static/js/logo.js` | Sizes the brand text to match the logo |

## Offline use

CodeMirror and js-yaml load from a CDN. To run offline, copy them into
`static/vendor/` and point `index.html` at the local copies.
