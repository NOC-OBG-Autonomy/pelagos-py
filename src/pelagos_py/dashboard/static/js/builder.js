// The pipeline builder. Built from the live registry, so new steps need no changes here.

// `nodes` holds steps and sections; `items` is the flat step list (syncItems) the rest of the app reads.
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

// Detected by shape (a `qc_settings` param), not by name, so a new QC-applying step also works.
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
// Per QC test for an Apply QC step with tests, else the step's own switch.
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

// Also clears Apply QC per-test overrides so every test follows the master.
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
  const fill = () => {
    renderPaletteInto(body, search.value, {
      step: (name) => insert(() => insertStepAt(name, list, index)),
      qc: (container, test) => insert(() => {
        // A test picked right under an Apply QC step joins it.
        const prev = list[index - 1];
        if (prev && prev.name === container && isQcContainer(prev.def)) addTestTo(prev, test);
        else insertQcAt(container, test, list, index);
      }),
    });
    // Containers only go at the top level: they never nest.
    const atRoot = list === STATE.pipeline.nodes;
    if (atRoot && 'new container'.includes(search.value.trim().toLowerCase())) {
      const item = paletteItem('New container', 'An empty group to put steps in', null,
        () => insert(() => addSection(index)));
      item.draggable = false;
      body.prepend(item);
    }
  };
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
      // QC tests are listed under their container so one can be dragged straight in.
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
  el.innerHTML = `<div class="pi-name">${escapeHtml(name)}</div>` +
    (description ? `<div class="pi-desc">${escapeHtml(description)}</div>` : '');
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

// While running, arrows into steps already reached are lit.
function flowLink(list, index, tail = false) {
  const el = document.createElement('div');
  el.className = 'flow-link' + (tail ? ' tail' : '');
  const next = list[index];
  const into = STATE.pipeline.items.indexOf(isSection(next) ? next.steps[0] : next);
  const cur = RunLock.running ? (RunLock.runningIndex ?? RunLock.index) : null;
  if (cur != null && into >= 0 && into <= cur) el.classList.add('done');
  const empty = tail && list === STATE.pipeline.nodes && !list.length;
  if (empty) el.classList.add('empty');
  el.appendChild(Forms.button(empty ? 'Add step' : '', { icon: 'plus', iconSize: 12, cls: 'add-step sm',
    title: tail && list !== STATE.pipeline.nodes ? 'Add a step at the end of this container' : 'Insert a step here',
    onclick: (e) => { e.stopPropagation(); el.classList.add('open'); openStepPicker(el, list, index); } }));
  return el;
}

// The marker doubles as the drag handle: on hover the dot turns into a grip.
function stepKind(item) {
  if (isQcContainer(item.def)) return 'manual qc' in (item.values.qc_settings || {}) ? 'manual' : 'qc';
  return { input_output: 'io', quality_control: 'qc' }[item.def.category] || 'proc';
}
function stepIcon(kind) {
  const el = document.createElement('span');
  el.className = 'step-mark drag k-' + kind;
  el.title = 'Drag to reorder';
  el.innerHTML = `<span class="step-ico"></span>${Icon.svg('grip', 16)}`;
  return el;
}
function testLabel(test) {
  if (test === 'manual qc') return 'Manual QC';
  const t = test.replace(/\s+qc$/, '');
  return t.charAt(0).toUpperCase() + t.slice(1);
}
function testVars(tv) {
  const v = tv && (tv.variable_ranges || tv.variables);
  return !v ? [] : Array.isArray(v) ? v : Object.keys(v);
}
function stepText(name, sub, nameCls = 'step-name') {
  const text = document.createElement('span');
  text.className = 'step-text';
  text.appendChild(Forms.el('span', { class: nameCls, textContent: name }));
  if (sub) text.appendChild(Forms.el('span', { class: 'step-sub', textContent: sub }));
  return text;
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
  if (isQcContainer(def)) item.values.qc_settings = {};
  return item;
}

// Mirrors the demo loader: export next to the input as <stem>_Processed.nc.
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
function addSection(index) {
  STATE.pipeline.nodes.splice(index, 0, makeSection());
  renderPipeline();
  STATE.onChange();
}

function removeSection(id) {
  const nodes = STATE.pipeline.nodes;
  const at = nodes.findIndex((n) => isSection(n) && n.id === id);
  if (at < 0) return;
  nodes.splice(at, 1);
  renderPipeline();
  STATE.onChange();
}

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
// {kind: 'new', name}, {kind: 'move', id} or {kind: 'section', id}, set on dragstart.
let dragState = null;
let dropIndicator = null;

function qcCardAt(target) {
  if (!dragState || dragState.kind !== 'qc') return null;
  const card = target && target.closest ? target.closest('.step-card') : null;
  if (!card) return null;
  const item = STATE.pipeline.items.find((i) => i.id === Number(card.dataset.stepId));
  return item && item.name === dragState.container ? { card, item } : null;
}

// Sections never nest, so a section drag always resolves to the root.
function dropHostAt(target) {
  const root = document.getElementById('pipeline-steps');
  if (dragState && dragState.kind === 'section') return root;
  const body = target && target.closest ? target.closest('.section-body') : null;
  return body || root;
}

function listForHost(host) {
  if (!host.classList.contains('section-body')) return STATE.pipeline.nodes;
  const sec = STATE.pipeline.nodes.find(
    (n) => isSection(n) && n.id === Number(host.dataset.secId)
  );
  return sec ? sec.steps : STATE.pipeline.nodes;
}

function dropSlots(host) {
  const kids = host.children;
  return [...kids].filter((el) =>
    el.classList.contains('step-card') || el.classList.contains('section-card'));
}

function computeDropIndex(host, y) {
  const slots = dropSlots(host);
  for (let i = 0; i < slots.length; i++) {
    const r = slots[i].getBoundingClientRect();
    if (y < r.top + r.height / 2) return i;
  }
  return slots.length;
}

// The root's first slot and an empty pipeline have no arrow, so those get a line.
function dropLinkFor(host, index) {
  const slots = dropSlots(host);
  if (index === slots.length) return host.querySelector(':scope > .flow-link.tail:not(.empty)');
  let el = slots[index].previousElementSibling;
  while (el && !el.classList.contains('flow-link') && !slots.includes(el)) el = el.previousElementSibling;
  return el && el.classList.contains('flow-link') ? el : null;
}

function showDropIndicator(host, y) {
  const link = dropLinkFor(host, computeDropIndex(host, y));
  if (link) {
    link.classList.add('drop-target');
    return;
  }
  if (!dropIndicator) {
    dropIndicator = document.createElement('div');
    dropIndicator.className = 'drop-indicator';
  }
  const index = computeDropIndex(host, y);
  const slots = dropSlots(host);
  const at = index < slots.length ? slots[index] : null;
  if (at) at.parentElement.insertBefore(dropIndicator, at); else host.appendChild(dropIndicator);
}

function clearDropIndicator() {
  if (dropIndicator && dropIndicator.parentNode) dropIndicator.remove();
  document.querySelectorAll('.drag-active').forEach((el) => el.classList.remove('drag-active'));
  document.querySelectorAll('.drop-target').forEach((el) => el.classList.remove('drop-target'));
}

// Delegated from the root, so section bodies added by later renders need no re-wiring.
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
    showDropIndicator(host, e.clientY);
  });
  root.addEventListener('dragleave', (e) => {
    if (!root.contains(e.relatedTarget)) clearDropIndicator();
  });
  root.addEventListener('drop', (e) => {
    if (!dragState) return;
    e.preventDefault();
    const host = dropHostAt(e.target);
    const index = computeDropIndex(host, e.clientY);
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
function renderSettings() {
  const host = document.getElementById('pipeline-settings');
  host.innerHTML = '';
  const card = document.createElement('section');
  card.className = 'files-section settings-box';
  card.appendChild(Forms.el('div', { class: 'demos-head' }, Forms.el('strong', { textContent: 'Pipeline settings' })));
  const settings = STATE.pipeline.settings;
  const body = document.createElement('div');
  body.className = 'settings-body';
  const highlightSetting = (e) => {
    const field = e.target.closest('.field[data-param]');
    if (field) highlightYamlForStep(null, ['pipeline', field.dataset.param]);
  };
  body.addEventListener('mousedown', highlightSetting);
  body.addEventListener('focusin', highlightSetting);
  for (const spec of STATE.registry.pipeline_fields) {
    if (!(spec.name in settings)) settings[spec.name] = Forms.defaultValue(spec);
    const field = Forms.render(spec, settings, STATE.onChange);
    const hint = field.querySelector(':scope > .hint');
    if (hint) { field.title = hint.textContent; hint.remove(); }
    body.appendChild(field);
  }
  const diagRow = makeAllDiagnosticsRow();
  diagRow.title = 'On / Off flips the diagnostics switch of every step and QC test at once '
    + '(a step\'s "more diagnostics" is switched off too). Mixed means the switches differ.';
  body.appendChild(Forms.el('div', { class: 'settings-pair' }, body.lastElementChild, diagRow));
  card.appendChild(body);
  host.appendChild(card);
  applyRunLock();
  renderAllDiagnosticsRow();
}

// UI-only: writes nothing of its own to the YAML.
function makeAllDiagnosticsRow() {
  const row = document.createElement('div');
  row.id = 'all-diag-row';
  row.className = 'diag-row all-diag-row';
  const label = document.createElement('span');
  label.className = 'switch-label all-diag-label';
  label.textContent = 'All diagnostics';
  row.appendChild(label);
  row.appendChild(Forms.seg([[true, 'On'], [false, 'Off']], null, setAllDiagnostics));
  row.appendChild(Forms.el('span', { id: 'all-diag-state', class: 'all-diag-mixed',
    title: 'The diagnostics switches differ between steps' }));
  return row;
}

// In place, since rebuilding the settings card would drop focus from the field being edited.
function renderAllDiagnosticsRow() {
  const row = document.getElementById('all-diag-row');
  if (!row) return;
  const state = allDiagnosticsState();
  document.getElementById('all-diag-state').textContent = state === 'custom' ? 'mixed' : '';
  row.querySelector('.seg').set({ on: true, off: false }[state]);
}

// ---------------------------------------------------------------- run lock
// Frozen while running; only the paused step (or QC test) unlocks, since that's what Re-run acts on.
const RunLock = {
  running: false,
  index: null,   // null: all locked
  test: null,
  runningIndex: null,
  runningTest: null,

  begin() { RunLock.set(true, null, null); },
  end() { RunLock.runningIndex = null; RunLock.runningTest = null; RunLock.set(false, null, null); },

  // A null index re-locks everything (still running, no longer paused).
  pauseAt(index, test) {
    RunLock.runningIndex = null;
    RunLock.runningTest = null;
    if (index === null || index === undefined) {
      RunLock.set(true, null, null);
      return;
    }
    STATE.pipeline.items.forEach((s, i) => { s.collapsed = i !== index; });
    // Before rendering, or an already-expanded card keeps its old sections open.
    const item = STATE.pipeline.items[index];
    if (item && test) item.qcOpen = { [test]: true };
    RunLock.set(true, index, test || null);
    focusStepInBuilder(index);
  },

  stepStarted(index, test = null) {
    if (index === RunLock.runningIndex && test === RunLock.runningTest) return; // duplicate marker, e.g. a re-run
    const sameStep = index === RunLock.runningIndex;
    RunLock.runningIndex = index;
    RunLock.runningTest = test;
    // Next test of the same QC step: no re-render.
    if (sameStep) {
      const card = stepElement(index);
      if (card) markRunningTest(card);
      return;
    }
    STATE.pipeline.items.forEach((s) => { s.collapsed = true; s.qcOpen = null; });
    // Its section must be open for the card to scroll into view.
    const item = STATE.pipeline.items[index];
    const sec = item && sectionOfStep(item.id);
    if (sec) sec.collapsed = false;
    renderPipeline();
    const card = stepElement(index);
    if (card) card.scrollIntoView({ block: 'center', behavior: 'smooth' });
  },

  set(running, index, test) {
    RunLock.running = running;
    RunLock.index = index;
    RunLock.test = test;
    document.body.classList.toggle('run-locked', running);
    if (editor) editor.setOption('readOnly', running ? 'nocursor' : false);
    renderSettings();
    renderPipeline();
  },

  editable(index) {
    return !RunLock.running || RunLock.index === index;
  },
};

function markRunningTest(card) {
  const rows = [...card.querySelectorAll('.qc-row')];
  const at = rows.findIndex((row) => row.dataset.test === RunLock.runningTest);
  rows.forEach((row, i) => {
    row.classList.toggle('running', i === at);
    row.classList.toggle('pending', at >= 0 && i > at);
  });
}

function freezeControls(root, keep) {
  for (const el of root.querySelectorAll('input, select, textarea, button')) {
    if (keep && keep.contains(el)) continue;
    el.disabled = true;
  }
}

// Called at the end of renderPipeline/renderSettings so every path is covered.
function applyRunLock() {
  const running = RunLock.running;
  // Loading another config would replace the steps out from under the run.
  Config.updateControls();
  const picker = document.querySelector('#config-select .cfg-trigger');
  if (picker) {
    picker.disabled = running;
    picker.title = running ? 'Locked while the pipeline is running' : '';
  }
  document.querySelectorAll('#pipeline-steps .add-step').forEach((b) => { b.disabled = running; });
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
    if (index === RunLock.runningIndex) {
      card.classList.add('running');
      // Carry the pulse on from where it was, so a re-render doesn't restart it.
      card.style.animationDelay = -(performance.now() % 1800) + 'ms';
      markRunningTest(card);
    }
    freezeControls(card);
    return;
  }
  card.classList.add('unlocked');
  // Reordering or removing a step mid-run would break the runner's step indices.
  for (const b of card.querySelectorAll('.step-head button')) {
    if (b.title !== 'collapse' && b.title !== 'close') b.disabled = true;
  }
  const test = RunLock.test && card.querySelector(
    `.qc-test[data-test="${CSS.escape(RunLock.test)}"] .qc-test-body`);
  if (RunLock.test) freezeControls(card.querySelector('.step-body'), test || undefined);
}

function stepElement(index) {
  return document.querySelectorAll('#pipeline-steps .step-card')[index];
}

// ---------------------------------------------------------------- steps
// How the pipeline differs from default.yaml (see Defaults.compute).
let defaultsView = {
  byId: new Map(), first: [], after: new Map(),
  sectionFirst: new Map(), sectionsAfter: new Map(), sectionsFirst: [],
};

// Field edits re-mark in place; a full render would drop focus from the field.
let marksTimer = null;
function onFieldEdit() {
  STATE.onChange();
  clearTimeout(marksTimer);
  marksTimer = setTimeout(refreshDefaultMarks, 150);
}

function refreshDefaultMarks() {
  defaultsView = Defaults.compute();
  for (const card of document.querySelectorAll('#pipeline-steps .step-card[data-step-id]')) {
    const item = STATE.pipeline.items.find((i) => i.id === Number(card.dataset.stepId));
    if (item) applyDefaultMarks(card, item);
  }
  applyRunLock();
}

// Ghost rows for removed steps/tests only change on a render.
function applyDefaultMarks(card, item) {
  const info = defaultsView.byId.get(item.id);
  const diff = info && !info.extra ? info.diff : null;
  const setBadge = (name, kind) => {
    const slot = badgeSlot(name);
    slot.parentNode.querySelectorAll('.default-badge').forEach((b) => b.remove());
    if (kind === 'extra') slot.after(Defaults.badge('not in default', 'extra'));
    else if (kind === 'edited') slot.after(Defaults.badge('edited'));
  };
  const setNote = (host, note) => {
    host.querySelectorAll(':scope > .default-note.head-note').forEach((n) => n.remove());
    if (note) { note.classList.add('head-note'); host.prepend(note); }
  };
  const changesNote = (n, what, onRestore) =>
    Defaults.note(`${n} ${n === 1 ? 'change' : 'changes'} from the ${what} —`, 'Restore defaults', onRestore);
  const markFields = (host, diffs, values) => {
    let shown = 0;
    for (const f of host.querySelectorAll(':scope > .field[data-param]')) {
      const d = diffs && diffs.find((p) => p.name === f.dataset.param);
      f.classList.toggle('changed', !!d);
      f.querySelectorAll(':scope > .default-note').forEach((n) => n.remove());
      if (d) {
        f.appendChild(Defaults.changedNote(d.base, () => Defaults.restoreParam(values, d.name, d.base)));
        shown++;
      }
    }
    return shown;
  };

  const body = card.querySelector(':scope > .step-body');
  card.classList.toggle('extra', !!(info && info.extra));
  const name = card.querySelector('.step-head .step-name'); // absent on a collapsed Apply QC
  if (name) setBadge(name, info && info.extra ? 'extra' : diff && diff.count ? 'edited' : null);

  // A change is noted once, at the most detailed place that can show it.
  let shown = markFields(body, diff && diff.params, item.values);
  const qcd = diff && diff.qc;
  if (qcd) {
    shown += (diff.params || []).filter((p) => p.name === 'qc_settings').length;
    shown += qcd.removed.length; // removed tests have ghost rows
  }
  const baseQc = qcd ? ((info.base.parameters || {}).qc_settings || {}) : {};
  for (const el of card.querySelectorAll('.qc-test[data-test]')) {
    const t = el.dataset.test;
    const added = qcd && qcd.added.includes(t);
    const changed = qcd && qcd.changed[t];
    setBadge(el.querySelector('.qc-test-name'), added ? 'extra' : changed ? 'edited' : null);
    const tb = el.querySelector('.qc-test-body');
    const inFields = markFields(tb, changed, item.values.qc_settings[t]);
    const unshown = changed ? changed.length - inFields : 0;
    setNote(tb, added
      ? Defaults.note('Not in the default pipeline —', 'Remove', () => { delete item.values.qc_settings[t]; Defaults.commit(); }, 'extra')
      : unshown ? changesNote(unshown, 'default', () => Defaults.restoreTest(item, t, baseQc)) : null);
    if (added || changed) shown++;
  }
  const unshown = diff ? diff.count - shown : 0;
  setNote(body, unshown > 0 ? changesNote(unshown, 'default pipeline', () => Defaults.restoreStep(item, info.base)) : null);
}

function badgeSlot(name) {
  return name.closest('.step-text') || name;
}

// step id -> [{param, text}]
let issueMarks = new Map();
function setIssueMarks(marks) {
  issueMarks = marks;
  applyIssueMarks();
}

function applyIssueMarks() {
  for (const card of document.querySelectorAll('#pipeline-steps .step-card[data-step-id]')) {
    const fields = issueMarks.get(Number(card.dataset.stepId));
    const name = card.querySelector('.step-head .step-name');
    if (name) {
      const slot = badgeSlot(name);
      slot.parentNode.querySelectorAll('.issue-badge').forEach((b) => b.remove());
      if (fields) slot.after(Forms.el('span', { class: 'tag danger issue-badge', textContent: 'issue' }));
    }
    for (const f of card.querySelectorAll(':scope > .step-body > .field[data-param]')) {
      const problem = fields && fields.find((p) => p.param === f.dataset.param);
      f.classList.toggle('invalid', !!problem);
      f.querySelectorAll(':scope > .issue-note').forEach((n) => n.remove());
      if (problem) f.appendChild(Forms.el('div', { class: 'issue-note', textContent: problem.text }));
    }
  }
}

function appendStepCard(host, item) {
  host.appendChild(renderStepCard(item));
  for (const g of defaultsView.after.get(item.id) || []) host.appendChild(Defaults.ghostStep(g, item.id));
}

// ---------------------------------------------------------------- view
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
  applyIssueMarks();
  applyYamlFocus();
  renderAllDiagnosticsRow();
  scroller.scrollTop = top;
}

// --sec-* tokens in style.css; unmatched sections alternate by position.
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

function renderSection(sec, index = 0) {
  const card = document.createElement('div');
  card.className = 'section-card' + (sec.collapsed ? ' collapsed' : '');
  card.style.setProperty('--sec', `var(--sec-${sectionColour(sec, index)})`);

  const head = document.createElement('div');
  head.className = 'section-head';
  head.innerHTML = `<span class="sec-chevron">${Icon.svg('right', 13)}</span><span class="step-mark drag" title="Drag to move the whole container"><span class="sec-dot"></span>${Icon.svg('grip', 16)}</span>`;

  const title = document.createElement('input');
  title.className = 'section-title';
  title.value = sec.title;
  title.placeholder = 'Container name';
  title.oninput = () => { sec.title = title.value; STATE.onChange(); };
  title.onclick = (e) => e.stopPropagation(); // clicking the name shouldn't collapse
  head.appendChild(title);

  const names = sec.steps.flatMap((s) => isQcContainer(s.def) && Object.keys(s.values.qc_settings || {}).length
    ? Object.keys(s.values.qc_settings).map(testLabel) : [s.name]);
  head.appendChild(Forms.el('span', { class: sec.collapsed ? 'sec-seq' : 'sec-count',
    textContent: sec.collapsed ? names.join('  ›  ') : sec.steps.length + (sec.steps.length === 1 ? ' step' : ' steps') }));

  const tools = document.createElement('span');
  tools.className = 'step-tools';
  const del = document.createElement('button');
  del.className = 'icon-btn';
  del.innerHTML = Icon.svg('close', 14);
  del.title = 'remove container and its steps';
  del.onclick = (e) => { e.stopPropagation(); removeSection(sec.id); };
  tools.appendChild(del);
  head.appendChild(tools);

  head.addEventListener('click', (e) => {
    if (e.target.closest('button') || e.target.closest('.drag')) return;
    sec.collapsed = !sec.collapsed;
    renderPipeline();
  });

  // Handle-gated drag, so the title stays editable.
  const handle = head.querySelector('.drag');
  handle.addEventListener('mousedown', () => { card.draggable = true; });
  handle.addEventListener('mouseup', () => { card.draggable = false; });
  card.addEventListener('dragstart', (e) => {
    if (e.target !== card) return; // a step inside being dragged, bubbling up
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
    body.appendChild(flowLink(sec.steps, i));
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

  const head = document.createElement('div');
  head.className = 'step-head';
  const expand = document.createElement('button');
  expand.className = 'icon-btn step-expand' + (item.collapsed ? ' collapsed' : '');
  expand.innerHTML = Icon.svg('down', 15);
  expand.title = item.collapsed ? 'expand' : 'collapse';
  expand.setAttribute('aria-expanded', String(!item.collapsed));
  expand.onclick = (e) => { e.stopPropagation(); item.collapsed = !item.collapsed; renderPipeline(); };
  head.appendChild(expand);
  const title = document.createElement('span');
  title.className = 'step-title';
  const info = defaultsView.byId.get(item.id);
  const tests = isQcContainer(item.def) && item.collapsed ? Object.keys(item.values.qc_settings || {}) : [];
  if (tests.length) {
    head.classList.add('qc-stack');
    tests.forEach((t, i) => {
      if (i) title.appendChild(Forms.el('span', { class: 'qc-link' }));
      const row = document.createElement('span');
      row.className = 'qc-row';
      row.dataset.test = t;
      row.appendChild(stepIcon(t === 'manual qc' ? 'manual' : 'qc'));
      row.appendChild(stepText(testLabel(t), testVars(item.values.qc_settings[t]).join(', '), 'qc-row-name'));
      row.onclick = (e) => { e.stopPropagation(); item.collapsed = false; item.qcOpen = { [t]: true }; renderPipeline(); };
      title.appendChild(row);
    });
  } else {
    title.appendChild(stepIcon(stepKind(item)));
    title.appendChild(stepText(item.name, item.collapsed ? stepSummary(item) : ''));
    if (item.def.beta) {
      title.appendChild(Forms.el('span', { class: 'tag beta', textContent: 'beta',
        title: 'Beta: works, but the method is still being refined.' }));
    }
  }
  head.appendChild(title);

  const tools = document.createElement('span');
  tools.className = 'step-tools';
  const del = document.createElement('button');
  del.className = 'icon-btn step-del'; del.innerHTML = Icon.svg('close', 14); del.title = 'remove';
  del.onclick = (e) => { e.stopPropagation(); removeStep(item.id); };
  tools.appendChild(del);
  head.appendChild(tools);

  card.appendChild(head);

  head.addEventListener('click', (e) => {
    if (e.target.closest('button') || e.target.closest('.drag')) return;
    item.collapsed = !item.collapsed;
    renderPipeline();
  });

  // Draggable only while the handle is held, so text selection in the body still works.
  for (const handle of head.querySelectorAll('.drag')) {
    handle.addEventListener('mousedown', () => { card.draggable = true; });
    handle.addEventListener('mouseup', () => { card.draggable = false; });
  }
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

  card.addEventListener('mousedown', (e) => highlightYamlForStep(item.id, boxPath(e.target)));
  card.addEventListener('focusin', (e) => highlightYamlForStep(item.id, boxPath(e.target)));

  const body = document.createElement('div');
  body.className = 'step-body' + (item.collapsed ? ' collapsed' : '');

  // Apply QC has its own master and per-test switches in the QC editor.
  if (!isQcContainer(item.def)) {
    const diagTop = document.createElement('div');
    diagTop.className = 'diag-top';
    const diag = document.createElement('div');
    diag.className = 'diag-row';
    // true/false, or 'all' for more plots; a YAML list of names shows as plain on.
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
    diagTop.appendChild(diag);
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
      diagTop.appendChild(row);
    }
    body.appendChild(diagTop);
  }


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

  card.appendChild(body);
  applyDefaultMarks(card, item);
  return card;
}

function boxPath(target) {
  const test = target.closest('.qc-test[data-test]');
  const field = target.closest('.field[data-param]');
  if (test) {
    const inTest = field && test.contains(field);
    return ['parameters', 'qc_settings', test.dataset.test, ...(inTest ? [field.dataset.param] : [])];
  }
  return field ? ['parameters', field.dataset.param] : null;
}

// kept so a re-render can re-mark it
let yamlFocus = null;

function focusFieldInBuilder(index, path) {
  yamlFocus = { index, path };
  const box = applyYamlFocus(true);
  if (box) box.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function applyYamlFocus(open = false) {
  document.querySelectorAll('.yaml-focus').forEach((el) => el.classList.remove('yaml-focus'));
  if (!yamlFocus) return null;
  const box = yamlFocusBox(yamlFocus.index, yamlFocus.path, open);
  if (box) box.classList.add('yaml-focus');
  return box;
}

function yamlFocusBox(index, path, open) {
  const param = (host, name) => host.querySelector(`:scope > .field[data-param="${CSS.escape(name)}"]`);
  if (index == null) {
    if (path[0] !== 'pipeline' || !path[1]) return null;
    return param(document.querySelector('#pipeline-settings .settings-body'), path[1]);
  }
  const card = stepElement(index);
  if (!card || path[0] !== 'parameters' || !path[1]) return null;
  const body = card.querySelector(':scope > .step-body');
  if (path[1] !== 'qc_settings' || !path[2]) return param(body, path[1]);
  const test = card.querySelector(`.qc-test[data-test="${CSS.escape(path[2])}"]`);
  if (!test) return null;
  const testBody = test.querySelector('.qc-test-body');
  if (open && testBody.style.display === 'none') test.querySelector('.qc-test-head').click();
  return (path[3] && param(testBody, path[3])) || test;
}

// Only re-renders to un-collapse the target, since this runs on every cursor move.
function focusStepInBuilder(index) {
  const items = STATE.pipeline.items;
  if (index < 0 || index >= items.length) return;
  const sec = sectionOfStep(items[index].id);
  if (items[index].collapsed || (sec && sec.collapsed)) {
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

  // Flipping the master clears per-test overrides so all on / all off is unambiguous.
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

  // insertion order = application order
  for (const qcName of Object.keys(item.values.qc_settings)) {
    const qcDef = STATE.qcByName[qcName];
    const testValues = item.values.qc_settings[qcName];
    const test = document.createElement('div');
    test.className = 'qc-test';
    test.dataset.test = qcName;

    const th = document.createElement('div');
    th.className = 'qc-test-head';
    // Kept on the item (never serialised) so a re-render or pause opens the right test.
    const openNow = !!(item.qcOpen && item.qcOpen[qcName]);
    const chev = document.createElement('span');
    chev.className = 'qc-chevron';
    chev.innerHTML = Icon.svg(openNow ? 'down' : 'right', 14);
    th.appendChild(chev);
    th.insertAdjacentHTML('beforeend', `<span class="qc-test-name">${escapeHtml(qcName)}</span>`);
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

    // Absent inherits the master.
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
