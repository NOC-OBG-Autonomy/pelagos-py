# pelagos_py dashboard

A standalone web tool for **authoring, validating, and running** pelagos_py
pipeline configs. It is completely independent of the pipeline: it imports
`pelagos_py` only to *introspect* it. Nothing here is required to run the
pipeline operationally — a config authored in the dashboard is an ordinary YAML
file you can run any other way.

## What makes it "smart"

- **Auto-discovers steps.** The palette and every parameter form are generated
  from the live `STEP_CLASSES` / `QC_CLASSES` registries and each step's
  `describe_parameters()`. Add a new `@register_step` and it appears on the
  next server start — no dashboard changes.
- **Real validation.** `Validate` calls the pipeline's own
  `parameter_spec.resolve()` on the server, so what the dashboard accepts is
  exactly what the pipeline accepts (types, required params, `options`,
  unknown-key rejection). No duplicated validation logic to drift.
- **QC-aware.** Adding an *Apply QC* step gives you a picker of every registered
  QC test, each configured via its own schema.

## Running

```bash
pip install -r dashboard/requirements.txt   # fastapi, uvicorn, pyyaml
# (pelagos_py itself must be importable — installed, or run from the repo root)
python dashboard/app.py
```

Then open <http://localhost:8791>.

Steps can be grouped into **sections** (Pipeline → *Section*): a named,
contiguous run of steps that can be renamed, collapsed, dragged around as a
block, and dropped into. Sections are written to the YAML as `# ==== TITLE ====`
banner comments and read back from them, so hand-written configs like
`examples/configs/example_config_nelson.yaml` open with their sections intact.
The pipeline itself never sees them.

Configs you save live in `dashboard/configs/`. The pipeline runs as a
subprocess from the repo root, so relative paths in a config (e.g.
`examples/data/...`) resolve as they normally would.

## Tuning a step while it runs

**A running pipeline owns the config.** While it runs, the builder, the pipeline
settings and the YAML pane are all locked — the run is executing the YAML as it
was submitted, so an edit would leave the screen disagreeing with what is
actually running.

A step with `diagnostics: true` **pauses the run** once it has drawn its plot.
The Run tab then shows a review panel for that step alone: its figure (click for
a full-window viewer — arrow keys page through, click to zoom). That step's card
in the builder — and only that card — unlocks, expands and scrolls into view, so:

1. look at the plot,
2. adjust a parameter on the highlighted builder card,
3. **Re-run step** — the pipeline re-executes just that step from its pre-step
   snapshot and re-pauses, keeping the previous attempt as a thumbnail so you
   can compare,
4. repeat until it looks right, then **Continue** (which re-locks the config).

Reordering and removing steps stay locked even on the unlocked card: the runner
is working from the step indices it started with.

The parameters you settled on are already in the config, so saving keeps them.
Every figure of the run is also archived in the **Plots** tab, grouped by step
and attempt.

**Apply QC steps pause one test at a time.** A QC step runs every test it is
given in a single call, so pausing after the step would mean every plot at once
and a form covering every test. Instead the runner splits such a step into one
execution per test (exactly what the config could spell out by hand), and each
test gets its own pause, its own plot, and unlocks only that test's section of
the QC editor. Re-run replays that one test from the state just before it — the
tests already applied in the same step keep their flags. Splitting only happens
when the step would pause anyway, so an unattended run is unaffected.

## Manual QC: drawing flags on the plot

The `manual qc` test (in an Apply QC step, with diagnostics on) turns the pause
into an editor. Its plot — `y_variable` vs `x_variable`, PRES vs TIME by
default, with dropdowns to change either axis — is embedded live in the Run tab.
**⌘/Ctrl-drag** a region and a popover asks which flag (0–9, with meanings) to
give the samples **inside** or **outside** it, and on which variables (the y
variable by default). Each box goes straight into the test's `boxes` parameter
and the test is re-run so the plot shows the flags applied; the × on a box
removes it the same way. Plain drag still zooms.

Nothing lives only in the browser: a box is just a 2D range test in the YAML
(`x: [lo, hi]`, `y: [lo, hi]`, `flag`, `mode`, `variables`), so the saved
config replays the same flags anywhere the pipeline runs. The runner reports
the dataset's variables at each pause (`__PELAGOS_VARS__`) to fill the pickers.

## Interactive plots

Diagnostic figures are matplotlib, captured as PNGs — which cannot be zoomed
into. So alongside each PNG the runner also tries to write a **plot spec**
(`fig_spec.py`): a JSON file with the figure's labels, limits, layout and trace
styling, plus a binary of every trace's full x/y. The viewer (`plot.js`) draws
all of it in one WebGL context — no thinning, however many millions of points —
with a progress bar while the data streams in. Drag a box to zoom, double-click
(or **Reset**) to go back, click a point for its exact values, click a legend
entry to hide that series. Panels a step drew with `sharex`/`sharey` keep their
ranges linked. **Image** in the viewer toolbar switches back to the PNG.

Nothing in `pelagos_py` changes: steps still just draw with matplotlib and call
`plt.show()`, and a run outside the dashboard never touches any of this.

**No step needs changing.** Steps keep drawing exactly as they do; whether a
figure becomes interactive depends only on what it is made of. Serialising is
all-or-nothing per figure: `fig_spec.py` understands line and marker plots
(including time-series drawn straight from `datetime64`), scatter points, and
`axhline`/`axvline` reference lines (range bounds, min/max limits). A figure
containing anything else — a histogram, `fill_between`, a `pcolormesh`, a map
projection — gets **no** spec and stays PNG-only, so a plot is never shown
half-drawn or subtly wrong.

The run log says which is which, per plot, and names what stopped it:

```
  · plot: TEMP Spike Test (interactive)
  · plot: Profile summary (image only — patches)
  · plot: Track map (image only — projection:mercator)
```

Thumbnails of interactive figures also carry a blue rule. To make a PNG-only
plot interactive you either change the *step* to draw it with lines/scatter, or
teach `fig_spec.py` the artist named in the log — add a branch to
`_scatter_trace`/`_line_trace` and drop it from `_unsupported`.

Data goes over the wire as float32 (dates as seconds since the figure's first
timestamp) — about 12 MB per 1.5M-point trace — and a float64 copy stays on the
server so a clicked point reports exact values. Per figure the runner writes
`fig_NNN.json` (spec), `fig_NNN.f32` (float32 data) and `fig_NNN_full.npz`
(float64) beside the PNG.

## Layout

| File | Role |
|------|------|
| `app.py` | FastAPI backend: `/api/registry`, `/api/validate`, config CRUD, run + SSE log stream |
| `static/index.html` | Three-pane UI shell |
| `static/js/api.js` | Backend fetch wrappers |
| `static/js/forms.js` | Schema → form-field renderer (generic) |
| `static/js/builder.js` | Step palette, pipeline list, sections, QC editor |
| `static/js/config.js` | Builder ⇄ YAML, save/load |
| `static/js/run.js` | Run, streamed log console, captured-figure model |
| `static/js/review.js` | Paused-step panel: its plots + its parameters + re-run |
| `static/js/viewer.js` | Full-window figure viewer (lightbox) |
| `static/js/plot.js` | WebGL renderer: spec + binary → zoomable panels |
| `fig_spec.py` | matplotlib figure → plot spec + float32 data (dashboard-only, best-effort) |
| `static/js/app.js` | Bootstrap and wiring |

## Later / offline

CodeMirror and js-yaml load from a CDN for now. To run fully offline (e.g.
wrapped as a local Tauri/pywebview app), vendor those into `static/vendor/` and
point `index.html` at the local copies.
