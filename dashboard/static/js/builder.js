// The pipeline builder: step palette, the ordered list of steps, and the
// schema-driven forms inside each. Everything here is derived from the live
// registry, so new steps need no builder changes.

// `nodes` is the ordered pipeline: each entry is either a step item or a
// section (a named, contiguous run of steps). `items` is the flattened step
// list, kept in sync by syncItems() — everything outside the builder (YAML
// generation, validation indices, YAML⇄builder highlighting) reads that.
const STATE = {
  registry: null,
  stepsByName: {},
  qcByName: {},
  pipeline: { settings: {}, nodes: [], items: [] },
  _seq: 0,
  onChange: () => {}, // wired by app.js to regenerate YAML
};

function isSection(n) { return !!n && n.kind === 'section'; }

function makeSection(title = 'New container') {
  return { kind: 'section', id: ++STATE._seq, title, collapsed: false, steps: [] };
}

function syncItems() {
  const out = [];
  for (const n of STATE.pipeline.nodes) {
    if (isSection(n)) out.push(...n.steps);
    else out.push(n);
  }
  STATE.pipeline.items = out;
}

// Where does a step live? -> {list, index, section} (section null when loose).
function locateStep(id) {
  const nodes = STATE.pipeline.nodes;
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (isSection(n)) {
      const k = n.steps.findIndex((s) => s.id === id);
      if (k >= 0) return { list: n.steps, index: k, section: n };
    } else if (n.id === id) {
      return { list: nodes, index: i, section: null };
    }
  }
  return null;
}

function sectionOfStep(id) {
  const loc = locateStep(id);
  return loc ? loc.section : null;
}

// A step is treated as a QC container if it declares a `qc_settings` dict param
// (i.e. the Apply QC step). Detected by shape, not by name, so a future
// QC-applying step gets the same smart editor for free.
function isQcContainer(def) {
  return (def.parameters || []).some(
    (p) => p.name === 'qc_settings' && (p.type === 'dict' || (Array.isArray(p.type) && p.type.includes('dict')))
  );
}

function initValues(def) {
  const v = {};
  for (const spec of def.parameters || []) v[spec.name] = Forms.defaultValue(spec);
  return v;
}

// ------------------------------------------------------- all-diagnostics
// Every diagnostics switch that gates a plot: per QC test (its effective value)
// for an Apply QC step with tests, else the step's own switch.
function diagnosticsLeaves() {
  const leaves = [];
  for (const item of STATE.pipeline.items) {
    if (isQcContainer(item.def)) {
      const tests = Object.keys(item.values.qc_settings || {});
      if (!tests.length) { leaves.push(item.diagnostics); continue; }
      for (const name of tests) {
        const tv = item.values.qc_settings[name];
        leaves.push('diagnostics' in tv ? !!tv.diagnostics : !!item.diagnostics);
      }
    } else {
      leaves.push(!!item.diagnostics);
    }
  }
  return leaves;
}

function allDiagnosticsState() {
  const leaves = diagnosticsLeaves();
  if (!leaves.length) return 'off';
  if (leaves.every((v) => v)) return 'on';
  if (leaves.every((v) => !v)) return 'off';
  return 'custom';
}

// Flip every diagnostics switch in the pipeline to `v`, the same way a single
// step's own switch does (and, for Apply QC, clearing per-test overrides so
// every test unambiguously follows the master).
function setAllDiagnostics(v) {
  for (const item of STATE.pipeline.items) {
    item.diagnostics = v;
    if (isQcContainer(item.def)) {
      for (const name of Object.keys(item.values.qc_settings || {})) {
        delete item.values.qc_settings[name].diagnostics;
      }
    }
  }
  renderPipeline();
  STATE.onChange();
}

// ---------------------------------------------------------------- step picker
// A searchable popover of every registered step (and QC test), opened from an
// "Add step" button; a click inserts at that button's position.
let picker = null;

function closeStepPicker() {
  if (picker) { picker.remove(); picker.anchor.classList.remove('open'); }
  picker = null;
  document.removeEventListener('mousedown', onPickerOutside);
}

function onPickerOutside(e) {
  if (picker && !picker.contains(e.target)) closeStepPicker();
}

function openStepPicker(anchor, list, index) {
  const reopen = !picker || picker.anchor !== anchor;
  closeStepPicker();
  if (!reopen) return;
  picker = document.createElement('div');
  picker.className = 'step-picker';
  picker.anchor = anchor;
  const search = document.createElement('input');
  search.placeholder = 'Search steps…';
  const body = document.createElement('div');
  body.className = 'step-picker-list';
  picker.appendChild(search); picker.appendChild(body);
  const insert = (fn) => { closeStepPicker(); fn(); };
  const fill = () => renderPaletteInto(body, search.value, {
    step: (name) => insert(() => insertStepAt(name, list, index)),
    qc: (container, test) => insert(() => {
      // A test picked right under an Apply QC step joins it.
      const prev = list[index - 1];
      if (prev && prev.name === container && isQcContainer(prev.def)) addTestTo(prev, test);
      else insertQcAt(container, test, list, index);
    }),
  });
  search.oninput = fill;
  fill();
  anchor.after(picker);
  search.focus();
  setTimeout(() => document.addEventListener('mousedown', onPickerOutside), 0);
}

function renderPaletteInto(host, filter, on) {
  host.innerHTML = '';
  const f = filter.trim().toLowerCase();
  const cats = [
    ['input_output', 'Input / Output'],
    ['processing', 'Processing'],
    ['quality_control', 'Quality Control'],
    ['other', 'Other'],
  ];
  const all = STATE.registry.steps;
  for (const [cat, title] of cats) {
    const match = (x) => !f || x.name.toLowerCase().includes(f) || (x.description || '').toLowerCase().includes(f);
    // A QC container stays listed while any of its tests matches the search.
    const items = all.filter(
      (s) => s.category === cat && (match(s) || (isQcContainer(s) && STATE.registry.qc.some(match)))
    );
    if (!items.length) continue;
    const group = document.createElement('div');
    group.className = 'cat-group cat-' + cat;
    const h = document.createElement('div');
    h.className = 'cat-title'; h.textContent = title;
    group.appendChild(h);
    for (const s of items) {
      group.appendChild(paletteItem(s.name, s.description, { kind: 'new', name: s.name }, () => on.step(s.name)));
      // The QC tests sit under their container step so one can be dragged in
      // directly: onto an existing Apply QC card to join it, or anywhere else
      // to make a new Apply QC step holding just that test.
      if (isQcContainer(s)) {
        for (const qc of STATE.registry.qc) {
          if (!match(s) && !match(qc)) continue;
          const el = paletteItem(qc.name, qc.description, { kind: 'qc', container: s.name, test: qc.name },
            () => on.qc(s.name, qc.name));
          el.classList.add('palette-qc');
          group.appendChild(el);
        }
      }
    }
    host.appendChild(group);
  }
}

function paletteItem(name, description, drag, onClick) {
  const el = document.createElement('div');
  el.className = 'palette-item';
  el.draggable = true;
  el.innerHTML = `<div class="pi-name">${name}</div>` +
    (description ? `<div class="pi-desc">${description}</div>` : '');
  el.onclick = onClick;
  el.addEventListener('dragstart', (e) => {
    el.classList.add('dragging');
    dragState = drag;
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('text/plain', name); // Firefox needs some data set
  });
  el.addEventListener('dragend', () => {
    el.classList.remove('dragging'); clearDropIndicator(); dragState = null;
  });
  return el;
}

// The connector drawn between two slots of a list (or after its last one,
// `tail`), carrying the "+" that inserts a step at that position.
function flowLink(list, index, tail = false) {
  const el = document.createElement('div');
  el.className = 'flow-link' + (tail ? ' tail' : '');
  el.appendChild(Forms.button(tail ? 'Add step' : '', { icon: 'plus', iconSize: 12, cls: 'add-step sm', title: 'Insert a step here',
    onclick: (e) => { e.stopPropagation(); el.classList.add('open'); openStepPicker(el, list, index); } }));
  return el;
}

// Manual QC only makes sense paused on its plot, so it starts with diagnostics on.
function initTestValues(test) {
  const v = initValues(STATE.qcByName[test]);
  if (test === 'manual qc') v.diagnostics = true;
  return v;
}

function addTestTo(item, test) {
  if (!(test in item.values.qc_settings)) {
    item.values.qc_settings[test] = initTestValues(test);
  }
  item.qcOpen = Object.assign({}, item.qcOpen, { [test]: true });
  item.collapsed = false;
  renderPipeline();
  STATE.onChange();
}

function insertQcAt(container, test, list, index) {
  const item = makeItem(container);
  if (!item) return;
  item.values.qc_settings[test] = initTestValues(test);
  item.qcOpen = { [test]: true };
  list.splice(Math.max(0, Math.min(index, list.length)), 0, item);
  renderPipeline();
  STATE.onChange();
}

// ---------------------------------------------------------------- add/mutate
function makeItem(name) {
  const def = STATE.stepsByName[name];
  if (!def) return null;
  const item = {
    id: ++STATE._seq,
    name,
    def,
    values: initValues(def),
    diagnostics: false,
    collapsed: false,
  };
  if (isQcContainer(def)) item.values.qc_settings = {}; // ordered map of qc tests
  return item;
}

// Mirror what the demo loader does server-side: export next to the input as
// <stem>_Processed.nc. Leaves a hand-set output path alone.
function syncOutputPath(filePath) {
  const m = String(filePath || '').match(/^(.*?)([^/\\]+?)(\.[^./\\]*)?$/);
  if (!m || !m[2]) return;
  const auto = `${m[1]}${m[2]}_Processed.nc`;
  let changed = false;
  for (const item of STATE.pipeline.items) {
    if (!(item.def && item.def.parameters.some((p) => p.name === 'output_path'))) continue;
    const cur = item.values.output_path;
    if ((!cur || /Processed\.nc$/.test(cur)) && cur !== auto) { item.values.output_path = auto; changed = true; }
  }
  if (changed) renderPipeline();
}

// Insert a new step (dragged from the palette) into `list` at `index`.
function insertStepAt(name, list, index) {
  const item = makeItem(name);
  if (!item) return;
  list.splice(Math.max(0, Math.min(index, list.length)), 0, item);
  renderPipeline();
  STATE.onChange();
}

function removeStep(id) {
  const loc = locateStep(id);
  if (!loc) return;
  loc.list.splice(loc.index, 1);
  renderPipeline();
  STATE.onChange();
}

// Up/down buttons walk the step through the pipeline in execution order,
// crossing section boundaries (out of a section, or into the neighbouring one).
function moveStep(id, dir) {
  const loc = locateStep(id);
  if (!loc) return;
  const nodes = STATE.pipeline.nodes;
  const to = loc.index + dir;

  if (to >= 0 && to < loc.list.length) {
    const neighbour = loc.list[to];
    if (isSection(neighbour)) { // loose step at root meeting a section: step in
      const [it] = loc.list.splice(loc.index, 1);
      if (dir < 0) neighbour.steps.push(it); else neighbour.steps.unshift(it);
    } else {
      [loc.list[loc.index], loc.list[to]] = [loc.list[to], loc.list[loc.index]];
    }
  } else if (loc.section) { // off the edge of a section: pop out to root
    const at = nodes.indexOf(loc.section);
    const [it] = loc.list.splice(loc.index, 1);
    nodes.splice(dir < 0 ? at : at + 1, 0, it);
  } else {
    return; // already at the top/bottom of the pipeline
  }
  renderPipeline();
  STATE.onChange();
}

// Move an existing step to a drop position in `list` (index counts the dragged
// item itself when it is already in that list).
function moveStepTo(id, list, index) {
  const loc = locateStep(id);
  if (!loc) return;
  let to = index;
  if (loc.list === list && loc.index < index) to -= 1; // removal shifts later positions left
  const [it] = loc.list.splice(loc.index, 1);
  list.splice(Math.max(0, Math.min(to, list.length)), 0, it);
  renderPipeline();
  STATE.onChange();
}

// ---------------------------------------------------------------- sections
function addSection() {
  STATE.pipeline.nodes.push(makeSection());
  renderPipeline();
  STATE.onChange();
}

// Deleting a section removes it and every step inside it.
function removeSection(id) {
  const nodes = STATE.pipeline.nodes;
  const at = nodes.findIndex((n) => isSection(n) && n.id === id);
  if (at < 0) return;
  nodes.splice(at, 1);
  renderPipeline();
  STATE.onChange();
}

// Move a whole section (with its steps) to a root position.
function moveSectionTo(id, index) {
  const nodes = STATE.pipeline.nodes;
  const from = nodes.findIndex((n) => isSection(n) && n.id === id);
  if (from < 0) return;
  let to = index;
  if (from < index) to -= 1;
  to = Math.max(0, Math.min(to, nodes.length - 1));
  if (to === from) return;
  const [sec] = nodes.splice(from, 1);
  nodes.splice(to, 0, sec);
  renderPipeline();
  STATE.onChange();
}

// ------------------------------------------------------------- drag & drop
// Set on dragstart: {kind:'new', name} from the palette, {kind:'move', id} for
// reordering an existing step, or {kind:'section', id} for a whole section.
// Read by the pipeline drop target.
let dragState = null;
let dropIndicator = null;

// The Apply QC card a dragged QC test is over, if any.
function qcCardAt(target) {
  if (!dragState || dragState.kind !== 'qc') return null;
  const card = target && target.closest ? target.closest('.step-card, .fc-chip[data-step-id]') : null;
  if (!card) return null;
  const item = STATE.pipeline.items.find((i) => i.id === Number(card.dataset.stepId));
  return item && item.name === dragState.container ? { card, item } : null;
}

// The drop host under the pointer: a section body, or the root (loose steps).
// Sections never nest, so a section drag always resolves to the root.
function dropHostAt(target) {
  const root = document.getElementById('pipeline-steps');
  const flowRow = root.querySelector(':scope > .fc-row');
  if (dragState && dragState.kind === 'section') return flowRow || root;
  const body = target && target.closest ? target.closest('.section-body') : null;
  return body || flowRow || root;
}

// The list a host writes into: a section's steps, or the root node list.
function listForHost(host) {
  if (!host.classList.contains('section-body')) return STATE.pipeline.nodes;
  const sec = STATE.pipeline.nodes.find(
    (n) => isSection(n) && n.id === Number(host.dataset.secId)
  );
  return sec ? sec.steps : STATE.pipeline.nodes;
}

// Direct children that occupy a drop slot (the root also holds section cards).
function dropSlots(host) {
  const kids = host.children;
  return [...kids].filter((el) =>
    ['step-card', 'section-card', 'fc-chip', 'fc-stage', 'fc-cell'].some((c) => el.classList.contains(c)));
}

function computeDropIndex(host, x, y) {
  const slots = dropSlots(host);
  const row = host.classList.contains('fc-row'); // cells are laid out in columns
  for (let i = 0; i < slots.length; i++) {
    const r = slots[i].getBoundingClientRect();
    if (!row) { if (y < r.top + r.height / 2) return i; continue; }
    if (y < r.top || (y <= r.bottom && x < r.left + r.width / 2)) return i;
  }
  return slots.length;
}

function showDropIndicator(host, x, y) {
  if (!dropIndicator) {
    dropIndicator = document.createElement('div');
    dropIndicator.className = 'drop-indicator';
  }
  const index = computeDropIndex(host, x, y);
  const slots = dropSlots(host);
  const at = index < slots.length ? slots[index] : null;
  if (at) at.parentElement.insertBefore(dropIndicator, at); else host.appendChild(dropIndicator);
  if (host.classList.contains('fc-row')) Flow.placeIndicator(dropIndicator, at || slots[slots.length - 1], !at);
}

function clearDropIndicator() {
  if (dropIndicator && dropIndicator.parentNode) dropIndicator.remove();
  document.querySelectorAll('.drag-active').forEach((el) => el.classList.remove('drag-active'));
}

// Wire the pipeline area as a drop target. Delegated from the root, so section
// bodies added by later renders are picked up without re-wiring. Called once at
// boot.
function initBuilderDnD() {
  const root = document.getElementById('pipeline-steps');
  root.addEventListener('dragover', (e) => {
    if (!dragState) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = dragState.kind === 'new' || dragState.kind === 'qc' ? 'copy' : 'move';
    clearDropIndicator();
    const qc = qcCardAt(e.target);
    if (qc) { qc.card.classList.add('drag-active'); return; }
    const host = dropHostAt(e.target);
    host.classList.add('drag-active');
    showDropIndicator(host, e.clientX, e.clientY);
  });
  root.addEventListener('dragleave', (e) => {
    if (!root.contains(e.relatedTarget)) clearDropIndicator();
  });
  root.addEventListener('drop', (e) => {
    if (!dragState) return;
    e.preventDefault();
    const host = dropHostAt(e.target);
    const index = computeDropIndex(host, e.clientX, e.clientY);
    const list = listForHost(host);
    const st = dragState;
    const qc = qcCardAt(e.target);
    clearDropIndicator();
    dragState = null;
    if (st.kind === 'qc') {
      if (qc) addTestTo(qc.item, st.test);
      else insertQcAt(st.container, st.test, list, index);
    }
    else if (st.kind === 'new') insertStepAt(st.name, list, index);
    else if (st.kind === 'section') moveSectionTo(st.id, index);
    else moveStepTo(st.id, list, index);
  });
}

// ---------------------------------------------------------------- settings
// Collapse state for the settings card, kept across re-renders (module-level
// rather than on STATE.pipeline, which is replaced whenever a config loads).
let settingsCollapsed = true;

function renderSettings() {
  const host = document.getElementById('pipeline-settings');
  host.innerHTML = '';
  const card = document.createElement('div');
  card.className = 'step-card settings';
  const head = document.createElement('div');
  head.className = 'step-head';
  head.innerHTML = '<span class="step-name">Pipeline settings</span>';
  const toggle = document.createElement('button');
  toggle.className = 'icon-btn';
  toggle.innerHTML = Icon.svg(settingsCollapsed ? 'right' : 'down');
  toggle.title = 'collapse';
  toggle.onclick = (e) => { e.stopPropagation(); settingsCollapsed = !settingsCollapsed; renderSettings(); };
  head.appendChild(toggle);
  head.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    settingsCollapsed = !settingsCollapsed;
    renderSettings();
  });
  card.appendChild(head);
  const body = document.createElement('div');
  body.className = 'step-body' + (settingsCollapsed ? ' collapsed' : '');
  for (const spec of STATE.registry.pipeline_fields) {
    if (!(spec.name in STATE.pipeline.settings)) {
      STATE.pipeline.settings[spec.name] = Forms.defaultValue(spec);
    }
    body.appendChild(Forms.render(spec, STATE.pipeline.settings, STATE.onChange));
  }
  body.appendChild(makeAllDiagnosticsRow());
  const diagHint = document.createElement('div');
  diagHint.className = 'hint';
  diagHint.textContent = 'On / Off flips the diagnostics switch of every step and QC test at once '
    + '(a step\'s "more diagnostics" is switched off too). Custom means the switches differ.';
  body.appendChild(diagHint);
  card.appendChild(body);
  host.appendChild(card);
  applyRunLock();
  renderAllDiagnosticsRow();
}

// UI-only on/off/custom summary of every diagnostics switch, with buttons to flip
// them all; nothing of its own is written to the YAML.
function makeAllDiagnosticsRow() {
  const row = document.createElement('div');
  row.id = 'all-diag-row';
  row.className = 'diag-row all-diag-row';
  const label = document.createElement('span');
  label.className = 'switch-label all-diag-label';
  label.textContent = 'All diagnostics';
  row.appendChild(label);
  row.appendChild(Forms.el('span', { id: 'all-diag-state', class: 'tag',
    title: 'On: every step and QC test draws its default plots. Off: none. Custom: the switches differ between steps.' }));
  row.appendChild(Forms.seg([[true, 'On'], [false, 'Off']], null, setAllDiagnostics));
  return row;
}

// Refresh just the all-diagnostics row's text/button state in place, without
// rebuilding the rest of the settings card (which would drop focus from
// whatever pipeline-settings field the user is mid-edit on).
function renderAllDiagnosticsRow() {
  const row = document.getElementById('all-diag-row');
  if (!row) return;
  const state = allDiagnosticsState();
  const label = { on: 'On', off: 'Off', custom: 'Custom' }[state];
  const stateEl = document.getElementById('all-diag-state');
  stateEl.textContent = label;
  stateEl.className = 'tag ' + ({ on: 'ok', custom: 'accent' }[state] || '');
  row.querySelector('.seg').set({ on: true, off: false }[state]);
}

// ---------------------------------------------------------------- run lock
//
// While a pipeline is running the config is frozen: the run is executing the
// YAML as it was submitted, so an edit anywhere else would leave the screen
// disagreeing with what is actually running. When the run pauses on a step, that
// step alone is unlocked — for a QC step split test by test, only the paused
// test — because that is exactly what Re-run will act on.
const RunLock = {
  running: false,
  index: null,   // step left editable while paused, or null for "all locked"
  test: null,    // QC test within it, when the step was split
  runningIndex: null, // step currently executing (not paused), for the highlight

  begin() { RunLock.set(true, null, null); },
  end() { RunLock.runningIndex = null; RunLock.set(false, null, null); },

  // Unlock the paused step and bring it into view, expanded. Passing a null
  // index re-locks everything (still running, no longer paused).
  pauseAt(index, test) {
    RunLock.runningIndex = null; // it's paused now, not mid-execution
    if (index === null || index === undefined) {
      RunLock.set(true, null, null);
      return;
    }
    // Collapse everything else, so the one open card is the editable one.
    STATE.pipeline.items.forEach((s, i) => { s.collapsed = i !== index; });
    settingsCollapsed = true;
    // Open the paused test *before* rendering, or an already-expanded card
    // would keep its old sections open.
    const item = STATE.pipeline.items[index];
    if (item && test) item.qcOpen = { [test]: true };
    RunLock.set(true, index, test || null);
    focusStepInBuilder(index);
    if (viewMode === 'flow' && item) Flow.openDialog(item);
  },

  // Called as each step (or split QC test) starts executing. Collapses every
  // card — so the previous step's editor doesn't just sit there open while
  // it's greyed out and no longer relevant — and calls out the running one
  // with a highlight, scrolled into view.
  stepStarted(index) {
    if (index === RunLock.runningIndex) return; // duplicate marker, e.g. a re-run
    RunLock.runningIndex = index;
    STATE.pipeline.items.forEach((s) => { s.collapsed = true; s.qcOpen = null; });
    settingsCollapsed = true;
    // The card stays collapsed, but its section must be open or it wouldn't
    // be in view to scroll to at all.
    const item = STATE.pipeline.items[index];
    const sec = item && sectionOfStep(item.id);
    if (sec) sec.collapsed = false;
    Flow.closeDialog();
    renderPipeline();
    const card = stepElement(index);
    if (card) card.scrollIntoView({ block: 'center', behavior: 'smooth' });
  },

  set(running, index, test) {
    RunLock.running = running;
    RunLock.index = index;
    RunLock.test = test;
    document.body.classList.toggle('run-locked', running);
    // The YAML pane is the other way into the config, so it locks with it.
    if (editor) editor.setOption('readOnly', running ? 'nocursor' : false);
    renderSettings();
    renderPipeline();
  },

  // True for the one step card the user may edit right now.
  editable(index) {
    return !RunLock.running || RunLock.index === index;
  },
};

// Disable every control under `root`, optionally sparing the subtree `keep`.
function freezeControls(root, keep) {
  for (const el of root.querySelectorAll('input, select, textarea, button')) {
    if (keep && keep.contains(el)) continue;
    el.disabled = true;
  }
}

// Apply the run lock to what has just been rendered. Called at the end of
// renderPipeline/renderSettings so every path through the builder is covered.
function applyRunLock() {
  const running = RunLock.running;
  // Anything that would add, remove or replace steps wholesale, including
  // loading another config out from under the run.
  for (const id of ['btn-clear', 'btn-add-section']) {
    const b = document.getElementById(id);
    if (b) b.disabled = running;
  }
  Config.updateControls(); // Delete follows the picker lock
  const picker = document.querySelector('#config-select .cfg-trigger');
  if (picker) {
    picker.disabled = running;
    picker.title = running ? 'Locked while the pipeline is running' : '';
  }
  document.querySelectorAll('#pipeline-steps :is(.add-step, .fc-plus)').forEach((b) => { b.disabled = running; });
  if (!running) return;
  const settings = document.getElementById('pipeline-settings');
  if (settings) {
    settings.classList.add('locked');
    freezeControls(settings);
  }
  const cards = document.querySelectorAll('#pipeline-steps .step-card');
  cards.forEach(lockCard);
}

function lockCard(card, index) {
  if (!RunLock.editable(index)) {
    card.classList.add('locked');
    card.classList.toggle('running', index === RunLock.runningIndex);
    freezeControls(card);
    return;
  }
  card.classList.add('unlocked');
  // Structural controls stay frozen even on the unlocked card: reordering or
  // removing it mid-run would break the step indices the runner is using.
  for (const b of card.querySelectorAll('.step-head button')) {
    if (b.title !== 'collapse' && b.title !== 'close') b.disabled = true;
  }
  // A split QC step exposes only the paused test's fields.
  const test = RunLock.test && card.querySelector(
    `.qc-test[data-test="${CSS.escape(RunLock.test)}"] .qc-test-body`);
  if (RunLock.test) freezeControls(card.querySelector('.step-body'), test || undefined);
}

// The element drawn for step `index` in whichever view is showing.
function stepElement(index) {
  return document.querySelectorAll('#pipeline-steps .step-card, #pipeline-steps .fc-chip[data-step-id]')[index];
}

// ---------------------------------------------------------------- steps
// How the pipeline differs from default.yaml, recomputed on every render
// (see Defaults.compute).
let defaultsView = { byId: new Map(), first: [], after: new Map(), sectionFirst: new Map(), sectionsAfter: new Map(), sectionsFirst: [] };

// A field edit re-marks what differs from the default in place (a full render
// would drop focus from the field being typed in), after a short debounce.
let marksTimer = null;
function onFieldEdit() {
  STATE.onChange();
  clearTimeout(marksTimer);
  marksTimer = setTimeout(refreshDefaultMarks, 150);
}

function refreshDefaultMarks() {
  defaultsView = Defaults.compute();
  if (viewMode === 'flow') Flow.render(document.getElementById('pipeline-steps'));
  for (const card of document.querySelectorAll('#pipeline-steps .step-card[data-step-id], .fc-dialog .step-card[data-step-id]')) {
    const item = STATE.pipeline.items.find((i) => i.id === Number(card.dataset.stepId));
    if (item) applyDefaultMarks(card, item);
  }
  applyRunLock();
}

// Badges, restore notes and changed-field bars for one card, replacing any
// already there. Ghost rows for removed steps/tests only change on a render.
function applyDefaultMarks(card, item) {
  const info = defaultsView.byId.get(item.id);
  const diff = info && !info.extra ? info.diff : null;
  const setBadge = (name, kind) => {
    name.parentNode.querySelectorAll('.default-badge').forEach((b) => b.remove());
    if (kind === 'extra') name.after(Defaults.badge('not in default', 'extra'));
    else if (kind === 'edited') name.after(Defaults.badge('edited'));
  };
  const setNote = (host, note) => {
    host.querySelectorAll(':scope > .default-note.head-note').forEach((n) => n.remove());
    if (note) { note.classList.add('head-note'); host.prepend(note); }
  };
  const changesNote = (n, what, onRestore) =>
    Defaults.note(`${n} ${n === 1 ? 'change' : 'changes'} from the ${what} —`, 'Restore defaults', onRestore);
  const markFields = (host, diffs, values) => {
    for (const f of host.querySelectorAll(':scope > .field[data-param]')) {
      const d = diffs && diffs.find((p) => p.name === f.dataset.param);
      f.classList.toggle('changed', !!d);
      f.querySelectorAll(':scope > .default-note').forEach((n) => n.remove());
      if (d) f.appendChild(Defaults.changedNote(d.base, () => Defaults.restoreParam(values, d.name, d.base)));
    }
  };

  const body = card.querySelector(':scope > .step-body');
  card.classList.toggle('extra', !!(info && info.extra));
  setBadge(card.querySelector('.step-head .step-name'), info && info.extra ? 'extra' : diff && diff.count ? 'edited' : null);
  setNote(body, diff && diff.count ? changesNote(diff.count, 'default pipeline', () => Defaults.restoreStep(item, info.base)) : null);
  markFields(body, diff && diff.params, item.values);

  const qcd = diff && diff.qc;
  const baseQc = qcd ? ((info.base.parameters || {}).qc_settings || {}) : {};
  for (const el of card.querySelectorAll('.qc-test[data-test]')) {
    const t = el.dataset.test;
    const added = qcd && qcd.added.includes(t);
    const changed = qcd && qcd.changed[t];
    setBadge(el.querySelector('.qc-test-name'), added ? 'extra' : changed ? 'edited' : null);
    const tb = el.querySelector('.qc-test-body');
    setNote(tb, added
      ? Defaults.note('Not in the default pipeline —', 'Remove', () => { delete item.values.qc_settings[t]; Defaults.commit(); }, 'extra')
      : changed ? changesNote(changed.length, 'default', () => Defaults.restoreTest(item, t, baseQc)) : null);
    markFields(tb, changed, item.values.qc_settings[t]);
  }
}

// A step card followed by placeholders for the default steps missing after it.
function appendStepCard(host, item) {
  host.appendChild(renderStepCard(item));
  for (const g of defaultsView.after.get(item.id) || []) host.appendChild(Defaults.ghostStep(g, item.id));
}

// ---------------------------------------------------------------- view
// 'flow' draws collapsed steps as compact chips joined by arrows (a flowchart);
// 'list' is the plain card list. Same DOM either way, the mode is a body class.
let viewMode = 'flow';
try { viewMode = localStorage.getItem('pelagos.view') || 'flow'; } catch (e) { /* no storage */ }

function setViewMode(mode) {
  viewMode = mode;
  try { localStorage.setItem('pelagos.view', mode); } catch (e) { /* no storage */ }
  document.body.classList.toggle('view-flow', mode === 'flow');
  document.getElementById('flow-legend').classList.toggle('hidden', mode !== 'flow');
  document.querySelectorAll('#view-toggle button').forEach((b) => b.classList.toggle('on', b.dataset.view === mode));
  Flow.closeDialog();
  renderPipeline();
}

// One line of "what it acts on" for a chip, read off the step's parameters.
const SUMMARY_KEYS = ['to_derive', 'target_variable', 'apply_to', 'shift_vars', 'par_var', 'variables', 'variable', 'method', 'export_format', 'standards'];
function stepSummary(item) {
  const v = item.values || {};
  if (isQcContainer(item.def)) {
    return Object.entries(v.qc_settings || {}).map(([t, tv]) => {
      const vars = Object.keys((tv && (tv.variable_ranges || tv.variables)) || {});
      return t.replace(/\s+qc$/, '') + (vars.length ? ': ' + vars.join(', ') : '');
    }).join(' · ');
  }
  const parts = [];
  for (const k of SUMMARY_KEYS) {
    if (!(k in v) || v[k] === '' || v[k] == null) continue;
    const val = v[k];
    parts.push(Array.isArray(val) ? val.join(', ') : typeof val === 'object' ? Object.keys(val).join(', ') : String(val));
    if (parts.length === 2) break;
  }
  if (v.output_as) parts.push('→ ' + v.output_as);
  return parts.join(' · ');
}

function renderPipeline() {
  const host = document.getElementById('pipeline-steps');
  // Emptying the host would snap the pane's scroll to the top; put it back.
  const scroller = host.closest('.builder-scroll'), top = scroller.scrollTop;
  host.innerHTML = '';
  closeStepPicker();
  syncItems();
  defaultsView = Defaults.compute();
  document.getElementById('empty-hint').style.display =
    STATE.pipeline.nodes.length ? 'none' : 'block';
  if (viewMode === 'flow') {
    Flow.render(host);
    applyRunLock();
    renderAllDiagnosticsRow();
    scroller.scrollTop = top;
    return;
  }
  host.className = '';

  for (const g of defaultsView.sectionsFirst) host.appendChild(Defaults.ghostSection(g));
  for (const g of defaultsView.first) host.appendChild(Defaults.ghostStep(g, null));
  let secIndex = 0;
  const nodes = STATE.pipeline.nodes;
  nodes.forEach((node, i) => {
    if (i) host.appendChild(flowLink(nodes, i));
    if (isSection(node)) host.appendChild(renderSection(node, secIndex++));
    else appendStepCard(host, node);
    for (const g of defaultsView.sectionsAfter.get(node.id) || []) host.appendChild(Defaults.ghostSection(g));
  });
  host.appendChild(flowLink(nodes, nodes.length, true));
  applyRunLock();
  renderAllDiagnosticsRow(); // step list just changed shape: on/off/custom may have too
  scroller.scrollTop = top;
}

// A section's colour token (--sec-* in style.css), keyed on its title then its
// steps; unmatched sections alternate blue-grey / grey by position.
const SECTION_COLOURS = [
  [/chla|chlorophyll|quench/, 'chla'], [/profile/, 'profiles'], [/import|load|export|write/, 'io'],
  [/\blat|\blon|coord|position/, 'coords'], [/cross.?cal/, 'crosscal'], [/interpol/, 'interp'],
  [/salin|cndc|conduct/, 'salinity'], [/ctd|temp|pres/, 'ctd'], [/backscatter|bbp|beta/, 'bbp'],
  [/oxy|doxy/, 'oxygen'], [/par\b|irradiance|light/, 'par'], [/mixed layer|mld|density/, 'mld'],
];
function sectionColour(sec, index) {
  const title = (sec.title || '').toLowerCase();
  const names = sec.steps.map((s) => (s.name || '').toLowerCase()).join(' | ');
  for (const text of [title, names]) {
    for (const [re, name] of SECTION_COLOURS) if (re.test(text)) return name;
  }
  return 'alt-' + (index % 2);
}

// One section: a titled, collapsible container holding a contiguous run of
// steps. The body is its own drop target (see dropHostAt).
function renderSection(sec, index = 0) {
  const card = document.createElement('div');
  card.className = 'section-card' + (sec.collapsed ? ' collapsed' : '');
  card.style.setProperty('--sec', `var(--sec-${sectionColour(sec, index)})`);

  const head = document.createElement('div');
  head.className = 'section-head';
  head.innerHTML =
    `<span class="drag" title="Drag to move the whole container">${Icon.svg('grip')}</span>` +
    `<span class="sec-chevron">${Icon.svg(sec.collapsed ? 'right' : 'down', 14)}</span>`;

  const title = document.createElement('input');
  title.className = 'section-title';
  title.value = sec.title;
  title.placeholder = 'Container name';
  title.oninput = () => { sec.title = title.value; STATE.onChange(); };
  title.onclick = (e) => e.stopPropagation(); // clicking the name shouldn't collapse
  head.appendChild(title);

  const count = document.createElement('span');
  count.className = 'sec-count';
  count.textContent = sec.steps.length + (sec.steps.length === 1 ? ' step' : ' steps');
  head.appendChild(count);

  const del = document.createElement('button');
  del.className = 'icon-btn';
  del.innerHTML = Icon.svg('close');
  del.title = 'remove container and its steps';
  del.onclick = (e) => { e.stopPropagation(); removeSection(sec.id); };
  head.appendChild(del);

  head.addEventListener('click', (e) => {
    if (e.target.closest('button') || e.target.closest('.drag')) return;
    sec.collapsed = !sec.collapsed;
    renderPipeline();
  });

  // Same handle-gated drag as step cards, so the title stays editable.
  const handle = head.querySelector('.drag');
  handle.addEventListener('mousedown', () => { card.draggable = true; });
  handle.addEventListener('mouseup', () => { card.draggable = false; });
  card.addEventListener('dragstart', (e) => {
    dragState = { kind: 'section', id: sec.id };
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', 'section-' + sec.id);
    card.classList.add('dragging');
    e.stopPropagation();
  });
  card.addEventListener('dragend', () => {
    card.draggable = false;
    card.classList.remove('dragging');
    clearDropIndicator();
    dragState = null;
  });

  const body = document.createElement('div');
  body.className = 'section-body';
  body.dataset.secId = String(sec.id);
  if (sec.collapsed) body.style.display = 'none';
  for (const g of defaultsView.sectionFirst.get(sec.id) || []) body.appendChild(Defaults.ghostStep(g, null, sec.id));
  sec.steps.forEach((item, i) => {
    if (i) body.appendChild(flowLink(sec.steps, i));
    appendStepCard(body, item);
  });
  body.appendChild(flowLink(sec.steps, sec.steps.length, true));

  card.appendChild(head);
  card.appendChild(body);
  return card;
}

function renderStepCard(item) {
  const card = document.createElement('div');
  card.className = 'step-card cat-' + item.def.category;
  if (isQcContainer(item.def) && 'manual qc' in (item.values.qc_settings || {})) card.classList.add('manual-qc');
  card.dataset.stepId = item.id;

  // header
  const head = document.createElement('div');
  head.className = 'step-head';
  head.innerHTML =
    `<span class="drag" title="Drag to reorder">${Icon.svg('grip')}</span>`;

  // Up/down stacked into one compact control, sat right next to the grip.
  const move = document.createElement('span');
  move.className = 'step-move';
  const mkMove = (ico, delta, title) => {
    const b = document.createElement('button');
    b.className = 'icon-btn move-btn'; b.innerHTML = Icon.svg(ico, 12); b.title = title;
    b.onclick = (e) => { e.stopPropagation(); moveStep(item.id, delta); };
    return b;
  };
  move.appendChild(mkMove('up', -1, 'move up'));
  move.appendChild(mkMove('down', 1, 'move down'));
  head.appendChild(move);

  // Name, plus — for a collapsed Apply QC step — the QC tests it holds as chips
  // inline on the same bar, so its contents are visible without opening it.
  // (Expanded, the QC editor below shows them, so they'd only be duplicated.)
  const title = document.createElement('span');
  title.className = 'step-title';
  const name = document.createElement('span');
  name.className = 'step-name'; name.textContent = item.name;
  title.appendChild(name);
  const info = defaultsView.byId.get(item.id);
  const sub = stepSummary(item);
  if (sub && item.collapsed) {
    const el = document.createElement('span');
    el.className = 'step-sub'; el.textContent = sub;
    title.appendChild(el);
  }
  if (isQcContainer(item.def) && item.collapsed) {
    for (const t of Object.keys(item.values.qc_settings || {})) {
      title.appendChild(Forms.el('span', { class: 'tag qc', textContent: t }));
    }
  }
  head.appendChild(title);

  // Expand/collapse: a quiet chevron that rotates to point at its state.
  const expand = document.createElement('button');
  expand.className = 'icon-btn step-expand' + (item.collapsed ? ' collapsed' : '');
  expand.innerHTML = Icon.svg('down', 15);
  expand.title = item.collapsed ? 'expand' : 'collapse';
  expand.setAttribute('aria-expanded', String(!item.collapsed));
  expand.onclick = (e) => { e.stopPropagation(); item.collapsed = !item.collapsed; renderPipeline(); };
  head.appendChild(expand);

  const del = document.createElement('button');
  del.className = 'icon-btn step-del'; del.innerHTML = Icon.svg('close'); del.title = 'remove';
  del.onclick = (e) => { e.stopPropagation(); removeStep(item.id); };
  head.appendChild(del);
  card.appendChild(head);

  // Click anywhere on the header (except a button or the drag handle) toggles.
  head.addEventListener('click', (e) => {
    if (e.target.closest('button') || e.target.closest('.drag') || Flow.dialogFor === item.id) return;
    item.collapsed = !item.collapsed;
    renderPipeline();
  });

  // Reorder by dragging the ⋮⋮ handle: the card is only draggable while the
  // handle is held, so text selection in the body still works normally.
  const handle = head.querySelector('.drag');
  handle.addEventListener('mousedown', () => { card.draggable = true; });
  handle.addEventListener('mouseup', () => { card.draggable = false; });
  card.addEventListener('dragstart', (e) => {
    dragState = { kind: 'move', id: item.id };
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', String(item.id));
    card.classList.add('dragging');
  });
  card.addEventListener('dragend', () => {
    card.draggable = false;
    card.classList.remove('dragging');
    clearDropIndicator();
    dragState = null;
  });

  // Interacting with a step highlights its lines in the YAML pane.
  card.addEventListener('mousedown', () => highlightYamlForStep(item.id));
  card.addEventListener('focusin', () => highlightYamlForStep(item.id));

  // body
  const body = document.createElement('div');
  body.className = 'step-body' + (item.collapsed ? ' collapsed' : '');

  if (!item.def.schema_declared) {
    const note = document.createElement('div');
    note.className = 'hint';
    note.textContent = 'This step has not declared a parameter schema yet; edit its parameters in the YAML pane.';
    body.appendChild(note);
  }

  for (const spec of item.def.parameters || []) {
    if (spec.name === 'qc_settings' && isQcContainer(item.def)) {
      body.appendChild(renderQcEditor(item, spec, info));
    } else {
      body.appendChild(Forms.render(spec, item.values, onFieldEdit));
    }
  }

  // diagnostics toggle (every step supports it). For the QC container the
  // diagnostics controls live inside the QC editor (a master + per-test), so
  // don't render a second one here.
  if (!isQcContainer(item.def)) {
    const diag = document.createElement('div');
    diag.className = 'diag-row';
    // `diagnostics` is true/false, or 'all' for steps that offer more plots
    // (item.def.more_diagnostics); a YAML list of names shows as plain on.
    const more = item.def.more_diagnostics ? Forms.switchEl(item.diagnostics === 'all', (v) => {
      item.diagnostics = v ? 'all' : true;
      sw.input.checked = true;
      renderAllDiagnosticsRow(); STATE.onChange();
    }) : null;
    const sw = Forms.switchEl(!!item.diagnostics, (v) => {
      item.diagnostics = v;
      if (more) more.input.checked = false;
      renderAllDiagnosticsRow(); STATE.onChange();
    });
    sw.input.id = 'diag-' + item.id;
    const lbl = document.createElement('label');
    lbl.className = 'switch-label';
    lbl.htmlFor = sw.input.id;
    lbl.textContent = 'diagnostics'; lbl.style.margin = '0';
    diag.appendChild(sw.el); diag.appendChild(lbl);
    body.appendChild(diag);
    if (more) {
      const row = document.createElement('div');
      row.className = 'diag-row diag-more';
      more.input.id = 'diag-more-' + item.id;
      const mlbl = document.createElement('label');
      mlbl.className = 'switch-label';
      mlbl.htmlFor = more.input.id;
      mlbl.textContent = 'more diagnostics'; mlbl.style.margin = '0';
      mlbl.title = 'Also draw every extra plot this step offers (turns diagnostics on)';
      row.appendChild(more.el); row.appendChild(mlbl);
      body.appendChild(row);
    }
  }

  card.appendChild(body);
  applyDefaultMarks(card, item);
  return card;
}

// Expand and scroll to a step card by index (driven by the YAML cursor). Only
// re-renders when it has to un-collapse the target, to stay cheap on every
// cursor move.
function focusStepInBuilder(index) {
  const items = STATE.pipeline.items;
  if (index < 0 || index >= items.length) return;
  const sec = sectionOfStep(items[index].id);
  if (viewMode !== 'flow' && (items[index].collapsed || (sec && sec.collapsed))) {
    items[index].collapsed = false;
    if (sec) sec.collapsed = false;
    renderPipeline();
  }
  const card = stepElement(index);
  if (!card) return;
  card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  card.classList.add('flash');
  setTimeout(() => card.classList.remove('flash'), 600);
}

// ---------------------------------------------------------------- QC editor
// `info` is the step's entry in defaultsView, for the removed-test rows.
function renderQcEditor(item, spec, info) {
  const qcd = info && !info.extra ? info.diff.qc : null;
  const baseQc = qcd ? ((info.base.parameters || {}).qc_settings || {}) : {};
  const wrap = document.createElement('div');
  wrap.className = 'field';
  const label = document.createElement('label');
  label.textContent = 'QC tests';
  wrap.appendChild(label);
  if (spec.description) {
    const hint = document.createElement('div');
    hint.className = 'hint'; hint.textContent = spec.description;
    wrap.appendChild(hint);
  }

  const editor = document.createElement('div');
  editor.className = 'qc-editor';

  // master diagnostics toggle: turns diagnostic plots on/off for ALL tests.
  // Bound to the step-level `item.diagnostics` (the inherited default); flipping
  // it clears every per-test override so "all on / all off" is unambiguous.
  const master = document.createElement('div');
  master.className = 'qc-master';
  const msw = Forms.switchEl(item.diagnostics, (v) => {
    item.diagnostics = v;
    for (const name of Object.keys(item.values.qc_settings)) {
      delete item.values.qc_settings[name].diagnostics;
    }
    renderPipeline();
    STATE.onChange();
  });
  msw.input.id = 'qc-master-' + item.id;
  const mlbl = document.createElement('label');
  mlbl.className = 'switch-label';
  mlbl.htmlFor = msw.input.id;
  mlbl.textContent = 'diagnostics — all tests';
  master.appendChild(msw.el); master.appendChild(mlbl);
  editor.appendChild(master);

  // add-test row
  const addRow = document.createElement('div');
  addRow.className = 'qc-add';
  const sel = Forms.select(STATE.registry.qc.map((q) => q.name), null, null, { placeholder: '— add a QC test —' });
  const addBtn = Forms.button('Add', { icon: 'plus' });
  addBtn.onclick = () => {
    const name = sel.value;
    if (!name || name in item.values.qc_settings) return;
    item.values.qc_settings[name] = initTestValues(name);
    sel.value = '';
    renderPipeline();
    STATE.onChange();
  };
  addRow.appendChild(sel); addRow.appendChild(addBtn);
  editor.appendChild(addRow);

  // configured tests (insertion order = application order)
  for (const qcName of Object.keys(item.values.qc_settings)) {
    const qcDef = STATE.qcByName[qcName];
    const testValues = item.values.qc_settings[qcName];
    const test = document.createElement('div');
    test.className = 'qc-test';
    test.dataset.test = qcName;

    const th = document.createElement('div');
    th.className = 'qc-test-head';
    // Which tests are expanded is remembered on the item (UI state, never
    // serialised), so a re-render — or a pause unlocking one test — can open
    // the right one.
    const openNow = !!(item.qcOpen && item.qcOpen[qcName]);
    const chev = document.createElement('span');
    chev.className = 'qc-chevron';
    chev.innerHTML = Icon.svg(openNow ? 'down' : 'right', 14);
    th.appendChild(chev);
    th.insertAdjacentHTML('beforeend', `<span class="qc-test-name">${qcName}</span>`);
    const rm = document.createElement('button');
    rm.className = 'icon-btn'; rm.innerHTML = Icon.svg('close'); rm.title = 'remove test';
    rm.onclick = (e) => {
      e.stopPropagation();
      delete item.values.qc_settings[qcName];
      renderPipeline(); STATE.onChange();
    };
    th.appendChild(rm);

    const tb = document.createElement('div');
    tb.className = 'qc-test-body';
    tb.style.display = openNow ? 'block' : 'none';
    th.onclick = () => {
      const open = tb.style.display === 'none';
      tb.style.display = open ? 'block' : 'none';
      chev.innerHTML = Icon.svg(open ? 'down' : 'right', 14);
      item.qcOpen = Object.assign({}, item.qcOpen, { [qcName]: open });
    };

    // metadata: what the test does, needs, and produces (self-documenting even
    // when the test has no tunable parameters).
    if (qcDef && qcDef.description) {
      const d = document.createElement('div');
      d.className = 'hint qc-meta'; d.textContent = qcDef.description;
      tb.appendChild(d);
    }
    const reqs = (qcDef && qcDef.required_variables) || [];
    const outs = (qcDef && qcDef.qc_outputs) || [];
    if (reqs.length) {
      const r = document.createElement('div');
      r.className = 'hint qc-meta'; r.textContent = 'Requires: ' + reqs.join(', ');
      tb.appendChild(r);
    }
    if (outs.length) {
      const o = document.createElement('div');
      o.className = 'hint qc-meta'; o.textContent = 'Outputs: ' + outs.join(', ');
      tb.appendChild(o);
    }

    // per-test diagnostics override. Absent => inherit the master; toggling
    // writes an explicit true/false for this test only.
    const diagDefault = 'diagnostics' in testValues ? testValues.diagnostics : item.diagnostics;
    const drow = document.createElement('div');
    drow.className = 'diag-row';
    const dsw = Forms.switchEl(!!diagDefault, (v) => { testValues.diagnostics = v; renderAllDiagnosticsRow(); STATE.onChange(); });
    dsw.input.id = 'qc-diag-' + item.id + '-' + qcName.replace(/\s+/g, '_');
    const dlbl = document.createElement('label');
    dlbl.className = 'switch-label'; dlbl.htmlFor = dsw.input.id;
    dlbl.textContent = 'diagnostics'; dlbl.style.margin = '0';
    drow.appendChild(dsw.el); drow.appendChild(dlbl);
    tb.appendChild(drow);

    const params = (qcDef && qcDef.parameters) || [];
    if (!params.length) {
      const none = document.createElement('div');
      none.className = 'hint'; none.textContent = 'No parameters.';
      tb.appendChild(none);
    }
    for (const p of params) tb.appendChild(Forms.render(p, testValues, onFieldEdit));

    test.appendChild(th); test.appendChild(tb);
    editor.appendChild(test);
  }
  for (const t of (qcd ? qcd.removed : [])) editor.appendChild(Defaults.ghostTest(item, t, baseQc));

  wrap.appendChild(editor);
  return wrap;
}
