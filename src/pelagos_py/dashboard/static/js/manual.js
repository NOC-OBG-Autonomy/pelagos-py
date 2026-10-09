// Manual QC tab, shown while a run is paused on the `manual qc` test.
// Boxes drawn on the plot go straight into the test's `boxes` parameter, so the config is the only state.

const ManualQC = {
  TEST: 'manual qc',
  FLAGS: [
    [0, 'no QC performed'], [1, 'good'], [2, 'probably good'], [3, 'probably bad'],
    [4, 'bad'], [5, 'value changed'], [6, 'not used'], [7, 'not used'],
    [8, 'estimated / interpolated'], [9, 'missing value'],
  ],
  // Same palette as pelagos_py.utils.fig_spec.FLAG_COLOURS.
  COLOURS: { 0: '#9aa5ad', 1: '#1f6fd6', 2: '#7fb2e5', 3: '#e8912b', 4: '#d6392f',
    5: '#9aa5ad', 6: '#9aa5ad', 7: '#9aa5ad', 8: '#17b6c4', 9: '#111111' },
  // Argo merge table COMBINATRIX[existing][new], from qc_handling via /api/registry.
  get COMBINATRIX() { return STATE.registry.combinatrix; },
  // Darker outline so a box stays visible over points of the same flag.
  border(flag) {
    const hex = ManualQC.COLOURS[flag] || '#333333';
    const c = parseInt(hex.slice(1), 16);
    const d = (v) => Math.round(v * 0.55).toString(16).padStart(2, '0');
    return '#' + d(c >> 16) + d((c >> 8) & 255) + d(c & 255);
  },
  VIEWS: [['time', 'Time'], ['profile', 'Profile'], ['variable', 'Variable'], ['ts', 'T–S'], ['map', 'Map']],

  // Kept across re-runs and tab switches within one pause.
  tool: { flag: 4, mode: 'inside', override: true },
  chart: null,
  fig: null,
  hi: null,      // box index hovered in the list
  sel: null,     // box index selected in the list
  dirty: false,  // boxes edited since the last re-run
  view: 'time',
  coloured: false, // Time/T–S/Map colour by the shown variable rather than by flag
  gi: 0,         // index into the test's `variables`
  param: null,   // the variable shown (null: the bubble's first)
  // `by` ALL shows every sample, else one profile/cycle id at `pos`.
  prof: { by: 'ALL', pos: 0 },
  hidden: new Set(), // flags toggled off in the legend
  past: [],      // undo snapshots of the box list, current one last
  future: [],    // redo snapshots
  cols: {},      // column name -> promise of {values, date, stops}
  index: {},     // PROFILE_NUMBER / CYCLE -> {ids, at: Map id -> sample indices}
  nearest: {},   // point box (JSON) -> its nearest sample
  flagCache: null, // replayed flags of one variable, see replayFlags
  sigma: {},     // T–S window -> promise of sigma0 contours

  isActive() { return Review.active && Review.test === ManualQC.TEST; },
  tab() { return document.querySelector('.tab[data-tab="manual"]'); },
  host() { return document.getElementById('manual-workspace'); },
  rail() { return document.getElementById('manual-rail'); },

  values() {
    const v = Review.testValues();
    if (v && !Array.isArray(v.boxes)) v.boxes = [];
    if (v && !Array.isArray(v.variables)) v.variables = [];
    return v;
  },

  mod() { return navigator.platform.toLowerCase().includes('mac') ? '⌘' : 'Ctrl'; },
  xVar() { const v = ManualQC.values(); return (v && v.x_variable) || 'TIME'; },
  yVar() { const v = ManualQC.values(); return (v && v.y_variable) || 'PRES'; },
  boxY(b) { return b.y_variable || ManualQC.yVar(); },
  boxX(b) { return b.x_variable || ManualQC.xVar(); },
  boxVars(b) { return b.variables || [ManualQC.boxY(b)]; },

  groups() {
    const values = ManualQC.values();
    return (values ? values.variables : []).map((v) => (Array.isArray(v) ? v : [v]));
  },
  group() { return ManualQC.groups()[ManualQC.gi] || null; },
  inGroup(b, group) {
    const vars = ManualQC.boxVars(b);
    return !!group && vars.length === group.length && vars.every((v) => group.includes(v));
  },
  paramVar() {
    const group = ManualQC.group();
    return ManualQC.param || (group ? group[0] : '');
  },
  flagVar() {
    const group = ManualQC.group() || [];
    const param = ManualQC.paramVar();
    return group.includes(param) ? param : group[0];
  },

  // Colour shows whatever isn't on the axes.
  axes() {
    const p = ManualQC.paramVar();
    return {
      time: { x: ManualQC.xVar(), y: ManualQC.yVar() },
      profile: { x: p, y: ManualQC.yVar() },
      variable: { x: ManualQC.xVar(), y: p },
      ts: ManualQC.tsAxes(),
      map: { x: 'LONGITUDE', y: 'LATITUDE' },
    }[ManualQC.view];
  },
  colourView() { return ['time', 'ts', 'map'].includes(ManualQC.view); },
  colourBy() { return ManualQC.colourView() && ManualQC.coloured ? ManualQC.paramVar() : null; },
  tsAxes() {
    const have = ManualQC.variables();
    return { x: ['PSAL', 'PRAC_SALINITY', 'ABS_SALINITY'].find((v) => have.includes(v)),
      y: ['TEMP', 'CONS_TEMP'].find((v) => have.includes(v)) };
  },
  viewMissing(view) {
    const have = ManualQC.variables();
    if (view === 'ts') {
      const { x, y } = ManualQC.tsAxes();
      if (!x || !y) return 'Needs a salinity (e.g. PSAL) and a temperature';
    }
    if (view === 'map' && !(have.includes('LATITUDE') && have.includes('LONGITUDE'))) {
      return 'Needs LATITUDE and LONGITUDE';
    }
    return '';
  },

  // A single profile also shows the boxes limited to it.
  shownBoxes() {
    const values = ManualQC.values();
    const { x, y } = ManualQC.axes();
    const group = ManualQC.group();
    const single = ManualQC.view === 'profile' && ManualQC.prof.by !== 'ALL';
    const key = ManualQC.prof.by === 'CYCLE' ? 'cycles' : 'profiles';
    const inView = (b) => (!b.profiles && !b.cycles) || (single && (b[key] || []).map(Number).includes(ManualQC.profileId()));
    return (values ? values.boxes : []).map((b, i) => ({ b, i }))
      .filter(({ b }) => b && ManualQC.inGroup(b, group) && ManualQC.boxX(b) === x && ManualQC.boxY(b) === y && inView(b));
  },

  boxView(b) {
    const x = ManualQC.boxX(b), y = ManualQC.boxY(b);
    const ts = ManualQC.tsAxes();
    if (x === 'LONGITUDE' && y === 'LATITUDE') return { view: 'map', label: 'Map' };
    if (x === ts.x && y === ts.y) return { view: 'ts', label: 'T–S' };
    if (x === ManualQC.xVar() && y === ManualQC.yVar()) return { view: 'time', label: 'Time' };
    if (x === ManualQC.xVar()) return { view: 'variable', param: y, label: `Variable · ${y}` };
    return { view: 'profile', param: x, label: `Profile · ${x}` };
  },

  goToBox(i) {
    const b = ManualQC.values().boxes[i];
    const { view, param } = ManualQC.boxView(b);
    const p = ManualQC.prof;
    ManualQC.view = view;
    if (param) ManualQC.param = param;
    ManualQC.sel = i; ManualQC.hi = null;
    const by = b.profiles ? 'PROFILE_NUMBER' : b.cycles ? 'CYCLE' : null;
    if (view === 'profile') p.by = by || 'ALL';
    ManualQC.host().querySelector('.manual-plots').replaceWith(ManualQC.toolbar());
    if (!by) { ManualQC.draw(); return; }
    ManualQC.columns([by]).then((cols) => {
      p.pos = Math.max(0, ManualQC.profileIndex(by, cols[by]).ids.indexOf(Number((b.profiles || b.cycles)[0])));
      ManualQC.draw();
    });
  },

  // What the runner reported at this pause, plus whatever the config already names.
  variables() {
    const out = [...(Run.variables || [])];
    for (const v of [ManualQC.xVar(), ManualQC.yVar(), ...ManualQC.groups().flat()]) if (!out.includes(v)) out.push(v);
    return out;
  },

  open() {
    ManualQC.cols = {}; ManualQC.index = {}; ManualQC.nearest = {}; ManualQC.flagCache = null;
    ManualQC.hi = null; ManualQC.sel = null;
    ManualQC.dirty = false;
    ManualQC.hidden = new Set(); ManualQC.past = []; ManualQC.future = [];
    ManualQC.clicks = null;
    ManualQC.tab().classList.remove('hidden');
    ManualQC.dropChart();
    ManualQC.host().innerHTML = ''; ManualQC.rail().innerHTML = '';
    Run.showTab('manual');
  },

  // Clearing the host alone leaves the chart's WebGL context and key listener alive.
  dropChart() {
    if (ManualQC.chart) ManualQC.chart.destroy();
    ManualQC.chart = null;
  },

  close() {
    ManualQC.dropChart(); ManualQC.fig = null;
    ManualQC.dirty = false;
    ManualQC.buttons();
    const tab = ManualQC.tab();
    tab.classList.add('hidden');
    if (tab.classList.contains('on')) Run.showTab('run');
    ManualQC.host().innerHTML = ''; ManualQC.rail().innerHTML = '';
  },

  // A new figure means the test ran, so rebuild the editor; the same figure just redraws.
  render(fig) {
    const host = ManualQC.host();
    if (fig && fig === ManualQC.fig && host.children.length) { ManualQC.draw(); return; }
    ManualQC.fig = fig;
    ManualQC.dirty = false;
    ManualQC.buttons();
    ManualQC.dropChart();
    host.innerHTML = ''; ManualQC.rail().innerHTML = '';
    if (!fig) {
      const hint = document.createElement('div');
      hint.className = 'hint review-empty';
      hint.textContent = Review.busy ? 'Re-running — the plot will appear here.'
        : 'The test produced no figure. Re-run it with diagnostics on, or Continue.';
      host.appendChild(hint);
      return;
    }
    if (!ManualQC.past.length) ManualQC.record();
    ManualQC.gi = Math.min(ManualQC.gi, Math.max(0, ManualQC.groups().length - 1));
    ManualQC.rail().appendChild(ManualQC.sidebar());
    host.appendChild(ManualQC.toolbar());
    const main = Forms.el('div', { class: 'manual-main' });
    main.append(Forms.el('div', { class: 'manual-stage' }), ManualQC.panel());
    host.appendChild(main);
    ManualQC.history();
    ManualQC.draw();
  },

  rebuild() {
    ManualQC.host().querySelector('.manual-plots').replaceWith(ManualQC.toolbar());
    ManualQC.rail().replaceChildren(ManualQC.sidebar());
  },

  section(title) {
    const sec = document.createElement('div');
    sec.className = 'manual-sec';
    const h = document.createElement('div');
    h.className = 'manual-sec-title'; h.textContent = title;
    sec.appendChild(h);
    return sec;
  },

  sidebar() {
    const side = document.createElement('div');
    side.className = 'manual-side';
    const tools = document.createElement('div');
    tools.className = 'manual-tools';
    side.appendChild(tools);
    const t = ManualQC.tool;
    const flags = ManualQC.section('Flag new boxes as');
    const grid = document.createElement('div');
    grid.className = 'manual-flags';
    const btns = [];
    for (const [f, meaning] of ManualQC.FLAGS) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'manual-flag' + (f === t.flag ? ' on' : '');
      b.style.setProperty('--fc', ManualQC.COLOURS[f]);
      b.innerHTML = `<b>${f}</b><span>${meaning}</span>`;
      b.onclick = () => { t.flag = f; btns.forEach((x) => x.classList.toggle('on', x === b)); };
      btns.push(b); grid.appendChild(b);
    }
    flags.appendChild(grid);
    const modes = document.createElement('div');
    modes.className = 'manual-modes';
    for (const [m, text] of [['inside', 'Inside the box'], ['outside', 'Outside the box']]) {
      const l = document.createElement('label');
      const r = document.createElement('input');
      r.type = 'radio'; r.name = 'manual-mode'; r.checked = m === t.mode;
      r.onchange = () => { t.mode = m; };
      l.appendChild(r); l.appendChild(document.createTextNode(' ' + text));
      modes.appendChild(l);
    }
    flags.appendChild(modes);
    const ov = document.createElement('label');
    ov.className = 'manual-toggle';
    const ovsw = Forms.switchEl(t.override, (checked) => { t.override = checked; });
    ov.appendChild(ovsw.el);
    ov.appendChild(document.createTextNode(' Override existing flags'));
    ov.title = 'On: the box sets this flag outright, even bad → good. Off: merge by the Argo combinatrix, which never lowers a flag.';
    flags.appendChild(ov);
    tools.appendChild(flags);

    if (ManualQC.group()) {
      const vars = ManualQC.section('Apply flag to');
      vars.appendChild(ManualQC.varPicker());
      tools.appendChild(vars);
    }

    const rest = ManualQC.section('Untouched points');
    const row = document.createElement('label');
    row.className = 'manual-toggle';
    const values = ManualQC.values();
    const on = !values || values.flag_remaining_good !== false;
    const sw = Forms.switchEl(on, (checked) => {
      const v = ManualQC.values();
      if (!v) return;
      v.flag_remaining_good = checked;
      ManualQC.commit();
    });
    row.appendChild(sw.el);
    row.appendChild(document.createTextNode(' Set to 1 (good)'));
    rest.appendChild(row);
    const rh = document.createElement('div');
    rh.className = 'hint';
    rh.textContent = 'Points no box covers, for the variables being QC\'d. Off leaves them at 0 (no QC) for a later step.';
    rest.appendChild(rh);
    tools.appendChild(rest);
    return side;
  },

  panel() {
    const side = Forms.el('div', { class: 'manual-right' });
    const legend = ManualQC.section('Flags');
    legend.append(Forms.el('div', { class: 'manual-legend', id: 'manual-legend' }),
      Forms.el('div', { class: 'hint', id: 'manual-legend-hint' }));
    side.appendChild(legend);

    const boxes = ManualQC.section('Boxes');
    boxes.classList.add('manual-boxes');
    boxes.firstChild.append(
      Forms.button('', { icon: 'undo', iconSize: 14, cls: 'icon-btn', id: 'manual-undo', title: `Undo (${ManualQC.mod()}Z)`, onclick: () => ManualQC.undo() }),
      Forms.button('', { icon: 'redo', iconSize: 14, cls: 'icon-btn', id: 'manual-redo', title: `Redo (⇧${ManualQC.mod()}Z)`, onclick: () => ManualQC.redo() }));
    boxes.appendChild(Forms.el('div', { class: 'manual-list', id: 'manual-list' }));
    side.appendChild(boxes);
    return side;
  },

  renderLegend(counts) {
    const el = document.getElementById('manual-legend');
    if (!el) return;
    el.innerHTML = '';
    for (const [f, meaning] of ManualQC.FLAGS) {
      const hidden = ManualQC.hidden.has(f);
      if (!counts[f] && !hidden) continue;
      const row = Forms.button('', { cls: 'manual-leg' + (hidden ? ' off' : ''), title: hidden ? 'Show' : 'Hide',
        onclick: () => {
          if (hidden) ManualQC.hidden.delete(f); else ManualQC.hidden.add(f);
          ManualQC.draw();
        } });
      row.style.setProperty('--fc', ManualQC.COLOURS[f]);
      row.append(Forms.el('i'), Forms.el('span', { textContent: `${f} ${meaning}` }),
        Forms.el('b', { textContent: (counts[f] || 0).toLocaleString() }),
        Forms.el('span', { class: 'manual-leg-eye', html: Icon.svg(hidden ? 'eyeOff' : 'eye', 14) }));
      el.appendChild(row);
    }
    const hint = document.getElementById('manual-legend-hint');
    if (hint) hint.textContent = ManualQC.group() ? `Flags of ${ManualQC.flagVar()}, with your boxes. Hiding a flag hides its points and boxes.` : '';
  },

  record() {
    const values = ManualQC.values();
    const snap = JSON.stringify(values ? values.boxes : []);
    if (snap === ManualQC.past[ManualQC.past.length - 1]) return;
    ManualQC.past.push(snap);
    ManualQC.future = [];
  },

  undo() {
    if (ManualQC.past.length < 2) return;
    ManualQC.future.push(ManualQC.past.pop());
    ManualQC.restore(ManualQC.past[ManualQC.past.length - 1]);
  },

  redo() {
    if (!ManualQC.future.length) return;
    const snap = ManualQC.future.pop();
    ManualQC.past.push(snap);
    ManualQC.restore(snap);
  },

  restore(snap) {
    const values = ManualQC.values();
    if (!values) return;
    values.boxes = JSON.parse(snap);
    ManualQC.sel = null; ManualQC.hi = null;
    ManualQC.commit({ record: false });
  },

  // A double-click resets the zoom, so drop the points its two clicks just added.
  undoClicks() {
    const clicks = ManualQC.clicks;
    ManualQC.clicks = null;
    if (!clicks || Date.now() - clicks.at > 600) return;
    while (ManualQC.past.length > 1 && ManualQC.past[ManualQC.past.length - 1] !== clicks.before) ManualQC.past.pop();
    ManualQC.restore(clicks.before);
  },

  history() {
    const undo = document.getElementById('manual-undo'), redo = document.getElementById('manual-redo');
    if (undo) undo.disabled = ManualQC.past.length < 2;
    if (redo) redo.disabled = !ManualQC.future.length;
  },

  setGroup(gi) {
    if (gi === ManualQC.gi) return;
    ManualQC.gi = gi;
    ManualQC.param = null;
    ManualQC.sel = null; ManualQC.hi = null;
    ManualQC.rebuild();
    ManualQC.draw();
  },

  addGroup(v) {
    const values = ManualQC.values();
    if (!values) return;
    values.variables.push(v);
    ManualQC.gi = values.variables.length - 1;
    ManualQC.param = null;
    ManualQC.rebuild();
    ManualQC.commit();
  },

  removeGroup(gi) {
    const values = ManualQC.values();
    const group = ManualQC.groups()[gi];
    const mine = values.boxes.filter((b) => b && ManualQC.inGroup(b, group));
    if (mine.length && !confirm(`Stop QC'ing ${group.join(' + ')} and remove its ${mine.length} box${mine.length === 1 ? '' : 'es'}?`)) return;
    values.boxes = values.boxes.filter((b) => !mine.includes(b));
    values.variables.splice(gi, 1);
    ManualQC.gi = Math.max(0, Math.min(ManualQC.gi, values.variables.length - 1));
    ManualQC.param = null;
    ManualQC.rebuild();
    ManualQC.commit();
  },

  // Boxes move with the bubble so they keep flagging its variables.
  setGroupVars(vars) {
    const values = ManualQC.values();
    const old = ManualQC.group();
    if (!values || !old || !vars.length) return;
    for (const b of values.boxes) if (b && ManualQC.inGroup(b, old)) b.variables = [...vars];
    values.variables[ManualQC.gi] = vars.length === 1 ? vars[0] : [...vars];
    if (!vars.includes(ManualQC.param)) ManualQC.param = null;
    ManualQC.host().querySelector('.manual-plots').replaceWith(ManualQC.toolbar());
    ManualQC.commit();
  },

  varPicker() {
    const chosen = new Set(ManualQC.group());
    const candidates = () => ManualQC.variables().filter((v) => v !== 'TIME' && !(Run.emptyVariables || []).includes(v));
    const wrap = document.createElement('div');
    wrap.className = 'manual-pick';
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'selectish manual-pick-btn';
    const summary = () => {
      const v = [...chosen];
      const all = candidates().every((c) => chosen.has(c));
      btn.replaceChildren(Forms.el('span', { textContent: all ? `All variables (${v.length})` : v.join(', ') }));
      btn.title = v.join(', ');
    };
    summary();
    const menu = document.createElement('div');
    menu.className = 'manual-pick-menu hidden';
    const search = document.createElement('input');
    search.type = 'search'; search.placeholder = 'Search variables…';
    menu.appendChild(search);
    const opts = document.createElement('div');
    opts.className = 'manual-pick-opts';
    menu.appendChild(opts);
    const changed = () => { summary(); fill(); ManualQC.setGroupVars([...chosen]); };
    const option = (text, checked, onchange, cls = '') => {
      const l = Forms.el('label', { class: cls });
      const c = Forms.el('input', { type: 'checkbox', checked });
      c.onchange = () => onchange(c.checked);
      l.append(c, text);
      opts.appendChild(l);
    };
    const fill = () => {
      opts.innerHTML = '';
      const q = search.value.trim().toLowerCase();
      if (!q) {
        const all = candidates();
        option('All variables', all.every((v) => chosen.has(v)), (on) => {
          if (on) all.forEach((v) => chosen.add(v));
          else { const first = ManualQC.group()[0]; chosen.clear(); chosen.add(first); }
          changed();
        }, 'manual-pick-all');
      }
      for (const v of candidates()) {
        if (q && !v.toLowerCase().includes(q)) continue;
        option(v, chosen.has(v), (on) => {
          if (on) chosen.add(v); else if (chosen.size > 1) chosen.delete(v);
          changed();
        });
      }
    };
    search.oninput = fill;
    const close = () => { menu.classList.add('hidden'); document.removeEventListener('pointerdown', away); };
    const away = (e) => { if (!wrap.contains(e.target)) close(); };
    btn.onclick = () => {
      if (!menu.classList.contains('hidden')) { close(); return; }
      search.value = ''; fill();
      menu.classList.remove('hidden');
      document.addEventListener('pointerdown', away);
      search.focus();
    };
    wrap.appendChild(btn); wrap.appendChild(menu);
    return wrap;
  },

  toolbar() {
    const bar = Forms.el('div', { class: 'manual-plots' });
    const values = ManualQC.values();
    const boxes = values ? values.boxes : [];
    const bubbles = Forms.el('div', { class: 'manual-bar' });
    const seg = Forms.el('div', { class: 'seg' });
    ManualQC.groups().forEach((group, gi) => {
      const n = boxes.filter((b) => b && ManualQC.inGroup(b, group)).length;
      seg.appendChild(Forms.button('', { cls: 'manual-plot' + (gi === ManualQC.gi ? ' on' : ''), onclick: () => ManualQC.setGroup(gi),
        html: `<span>${group.join(' + ')}</span>` + (n ? `<i>${n}</i>` : ''),
        title: `Flags ${group.join(', ')}` + (n ? ` · ${n} box${n === 1 ? '' : 'es'}` : '') }));
    });
    if (seg.children.length) bubbles.appendChild(seg);
    if (ManualQC.group()) {
      bubbles.appendChild(Forms.button('', { icon: 'close', iconSize: 14, cls: 'icon-btn', title: 'Stop QC\'ing ' + ManualQC.group().join(' + '),
        onclick: () => ManualQC.removeGroup(ManualQC.gi) }));
    }
    const taken = new Set(ManualQC.groups().filter((g) => g.length === 1).map((g) => g[0]));
    bubbles.appendChild(ManualQC.variableSelect(null, (v) => { if (v) ManualQC.addGroup(v); },
      { placeholder: '+ QC variable', skip: taken }));
    bar.appendChild(bubbles);
    if (!ManualQC.group()) {
      bar.appendChild(Forms.el('span', { class: 'hint', textContent: 'Pick a variable to QC to draw boxes: each gets its own plot.' }));
      return bar;
    }

    const views = Forms.el('div', { class: 'manual-bar' });
    if (ManualQC.viewMissing(ManualQC.view)) ManualQC.view = 'time';
    const vseg = Forms.seg(ManualQC.VIEWS, ManualQC.view, (v) => ManualQC.setView(v), 'sm');
    ManualQC.VIEWS.forEach(([view], k) => {
      const missing = ManualQC.viewMissing(view);
      if (!missing) return;
      vseg.children[k].disabled = true;
      vseg.children[k].title = missing;
    });
    views.appendChild(vseg);
    views.appendChild(Forms.el('span', { class: 'hint', textContent: 'showing' }));
    const colourView = ManualQC.colourView();
    const shown = colourView && !ManualQC.coloured ? '' : ManualQC.paramVar();
    views.appendChild(ManualQC.variableSelect(shown, (v) => {
      if (colourView) ManualQC.coloured = !!v;
      if (v) ManualQC.param = v;
      ManualQC.sel = null; ManualQC.hi = null;
      ManualQC.draw();
    }, { cls: 'sm manual-param', placeholder: colourView ? 'flags only' : undefined }));
    if (ManualQC.view === 'profile') views.append(...ManualQC.profileControls());
    bar.appendChild(views);
    return bar;
  },

  setView(view) {
    if (view === ManualQC.view) return;
    ManualQC.view = view;
    ManualQC.sel = null; ManualQC.hi = null;
    ManualQC.host().querySelector('.manual-plots').replaceWith(ManualQC.toolbar());
    ManualQC.draw();
  },

  // A variable joins the first group with a name it starts with.
  VARIABLE_GROUPS: [
    ['Physics', ['TEMP', 'CONS_TEMP', 'PSAL', 'PRAC_SALINITY', 'ABS_SALINITY', 'DENSITY', 'SIGMA']],
    ['Bio-optics', ['CHLA', 'DOXY', 'MOLAR_DOXY', 'OXYSAT', 'BBP', 'DOWNWELLING_PAR', 'RAW_DOWNWELLING_PAR']],
  ],

  // SCI_PHASE first, then the groups, other variables, and finally those all NaN or 0.
  variableSelect(value, onChange, { placeholder, cls = '', skip = new Set() } = {}) {
    const vars = ManualQC.variables().filter((v) => v !== 'TIME' && !skip.has(v));
    const empty = new Set(Run.emptyVariables);
    const top = vars.filter((v) => v === 'SCI_PHASE' && !empty.has(v));
    let rest = vars.filter((v) => !empty.has(v) && !top.includes(v));
    const groups = [];
    for (const [label, prefixes] of ManualQC.VARIABLE_GROUPS) {
      const rank = (v) => prefixes.findIndex((pre) => v.startsWith(pre));
      const members = rest.filter((v) => rank(v) >= 0).sort((a, b) => rank(a) - rank(b));
      rest = rest.filter((v) => !members.includes(v));
      groups.push([label, members]);
    }
    groups.push(['Other', rest], ['No data (all NaN or 0)', vars.filter((v) => empty.has(v))]);
    const sel = Forms.select(top, value, onChange, { placeholder, cls });
    for (const [label, members] of groups) {
      if (!members.length) continue;
      const group = Forms.el('optgroup', { label });
      for (const v of members) group.appendChild(Forms.el('option', { value: v, textContent: v, selected: v === value }));
      sel.appendChild(group);
    }
    return sel;
  },

  profileControls() {
    const p = ManualQC.prof;
    const by = Forms.seg([['ALL', 'All'], ['PROFILE_NUMBER', 'Profile'], ['CYCLE', 'Cycle']], p.by, (v) => ManualQC.setBy(v), 'sm');
    ['PROFILE_NUMBER', 'CYCLE'].forEach((name, k) => {
      if (ManualQC.variables().includes(name)) return;
      by.children[k + 1].disabled = true;
      by.children[k + 1].title = `Needs ${name}: run Find Profiles before this QC step`;
    });
    const nav = Forms.el('div', { class: 'manual-nav' + (p.by === 'ALL' ? ' hidden' : '') });
    nav.appendChild(Forms.button('◀', { cls: 'sm', title: 'Previous (←)', onclick: () => ManualQC.step(-1) }));
    const id = Forms.el('input', { type: 'number', class: 'sm manual-nav-id', title: 'Jump to an id' });
    id.onchange = () => {
      const index = ManualQC.index[p.by];
      if (!index) return;
      // ids can have gaps
      const at = index.ids.findIndex((v) => v >= Number(id.value));
      p.pos = at < 0 ? index.ids.length - 1 : at;
      ManualQC.draw();
    };
    nav.appendChild(id);
    nav.appendChild(Forms.el('span', { class: 'hint manual-nav-of' }));
    nav.appendChild(Forms.button('▶', { cls: 'sm', title: 'Next (→)', onclick: () => ManualQC.step(1) }));
    return [by, nav];
  },

  profileIndex(by, col) {
    if (!ManualQC.index[by]) {
      const at = new Map();
      col.values.forEach((v, i) => {
        if (!isFinite(v)) return;
        if (!at.has(v)) at.set(v, []);
        at.get(v).push(i);
      });
      ManualQC.index[by] = { ids: [...at.keys()].sort((a, b) => a - b), at };
    }
    return ManualQC.index[by];
  },

  profileId() {
    const index = ManualQC.index[ManualQC.prof.by];
    return index ? index.ids[ManualQC.prof.pos] : null;
  },

  step(delta) {
    const index = ManualQC.index[ManualQC.prof.by];
    if (!index) return;
    const pos = Math.max(0, Math.min(index.ids.length - 1, ManualQC.prof.pos + delta));
    if (pos === ManualQC.prof.pos) return;
    ManualQC.prof.pos = pos;
    ManualQC.draw();
  },

  // The new id is the one holding the current profile's first sample.
  setBy(by) {
    const p = ManualQC.prof;
    const old = ManualQC.index[p.by];
    const first = old && old.at.get(old.ids[p.pos]);
    p.by = by;
    p.pos = 0;
    ManualQC.host().querySelector('.manual-nav').classList.toggle('hidden', by === 'ALL');
    ManualQC.sel = null; ManualQC.hi = null;
    if (by === 'ALL') { ManualQC.draw(); return; }
    ManualQC.columns([by]).then((cols) => {
      const index = ManualQC.profileIndex(by, cols[by]);
      if (first) p.pos = Math.max(0, index.ids.indexOf(cols[by].values[first[0]]));
      ManualQC.draw();
    });
  },

  columns(names) {
    const missing = [...new Set(names)].filter((n) => !(n in ManualQC.cols));
    if (missing.length) {
      const req = API.runColumns(missing).then((buf) => {
        const hl = new DataView(buf).getUint32(0, true);
        const { columns } = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
        const out = {};
        let off = 4 + hl;
        for (const c of columns) {
          const Arr = c.dtype === 'f8' ? Float64Array : Float32Array;
          const bytes = c.n * Arr.BYTES_PER_ELEMENT;
          out[c.name] = { values: new Arr(buf.slice(off, off + bytes)), date: c.dtype === 'f8', stops: c.stops, categories: c.categories };
          off += bytes;
        }
        return out;
      });
      for (const n of missing) {
        ManualQC.cols[n] = req.then((out) => out[n] || null);
        ManualQC.cols[n].catch(() => delete ManualQC.cols[n]);
      }
    }
    return Promise.all(names.map((n) => ManualQC.cols[n])).then((list) => Object.fromEntries(names.map((n, i) => [n, list[i]])));
  },

  sigma0(xlim, ylim) {
    const key = [...xlim, ...ylim].map((v) => v.toPrecision(4)).join(',');
    if (!ManualQC.sigma[key]) ManualQC.sigma[key] = API.sigma0(xlim[0], xlim[1], ylim[0], ylim[1]).catch(() => []);
    return ManualQC.sigma[key];
  },

  replayColumns(v) {
    const values = ManualQC.values();
    const names = [v + '_QC'];
    for (const b of (values ? values.boxes : [])) {
      if (!b || !ManualQC.boxVars(b).includes(v)) continue;
      names.push(ManualQC.boxX(b), ManualQC.boxY(b));
      if (b.profiles) names.push('PROFILE_NUMBER');
      if (b.cycles) names.push('CYCLE');
    }
    return names;
  },

  // Mirrors manual_qc._box_mask; null when a column it needs is missing.
  boxHit(b, cols) {
    const X = cols[ManualQC.boxX(b)], Y = cols[ManualQC.boxY(b)];
    if (!X || !Y) return null;
    const x = X.values, y = Y.values;
    const scope = b.profiles ? cols.PROFILE_NUMBER : b.cycles ? cols.CYCLE : null;
    const ids = (b.profiles || b.cycles || []).map(Number);
    const valid = (i) => isFinite(x[i]) && isFinite(y[i]) && (!scope || ids.includes(scope.values[i]));
    const bound = (v, col) => {
      if (!col.date) return Number(v);
      // Naive ISO times are UTC, as pandas reads them.
      const hasZone = /(Z|[+-]\d\d:?\d\d)$/i.test(v);
      return Date.parse(/T/.test(v) && !hasZone ? v + 'Z' : v);
    };
    let inside;
    if (b.x.length === 1) {
      const key = JSON.stringify(b);
      if (!(key in ManualQC.nearest)) {
        const px = bound(b.x[0], X), py = bound(b.y[0], Y);
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (let i = 0; i < x.length; i++) {
          if (!valid(i)) continue;
          x0 = Math.min(x0, x[i]); x1 = Math.max(x1, x[i]); y0 = Math.min(y0, y[i]); y1 = Math.max(y1, y[i]);
        }
        const sx = (x1 - x0) || 1, sy = (y1 - y0) || 1;
        let best = -1, bestD = Infinity;
        for (let i = 0; i < x.length; i++) {
          if (!valid(i)) continue;
          const d = ((x[i] - px) / sx) ** 2 + ((y[i] - py) / sy) ** 2;
          if (d < bestD) { bestD = d; best = i; }
        }
        ManualQC.nearest[key] = best;
      }
      const hit = ManualQC.nearest[key];
      inside = (i) => i === hit;
    } else {
      const [x0, x1] = b.x.map((v) => bound(v, X)).sort((m, n) => m - n);
      const [y0, y1] = b.y.map((v) => bound(v, Y)).sort((m, n) => m - n);
      inside = (i) => valid(i) && x[i] >= x0 && x[i] <= x1 && y[i] >= y0 && y[i] <= y1;
    }
    return (b.mode || 'inside') === 'outside' ? (i) => valid(i) && !inside(i) : inside;
  },

  // Mirrors manual_qc.return_qc (before flag_remaining_good), so unapplied boxes show too.
  // Cached per variable: a box appended since the last call is applied on top, not replayed with the rest.
  replayFlags(v, cols) {
    const values = ManualQC.values();
    const boxes = (values ? values.boxes : []).filter((b) => b && ManualQC.boxVars(b).includes(v));
    const keys = boxes.map((b) => JSON.stringify(b));
    const src = cols[v + '_QC'];
    let cache = ManualQC.flagCache;
    const reuse = cache && cache.v === v && cache.src === src
      && cache.keys.length <= keys.length && cache.keys.every((k, j) => k === keys[j]);
    if (!reuse) cache = ManualQC.flagCache = { v, src, keys: [], flags: Uint8Array.from(src.values) };
    const flags = cache.flags;
    for (let j = cache.keys.length; j < boxes.length; j++) {
      const b = boxes[j];
      const hit = ManualQC.boxHit(b, cols);
      if (hit) {
        for (let i = 0; i < flags.length; i++) {
          if (flags[i] === 9 || !hit(i)) continue;
          flags[i] = b.override === false ? ManualQC.COMBINATRIX[flags[i]][b.flag] : b.flag;
        }
      }
      cache.keys.push(keys[j]);
    }
    return flags;
  },

  // 1st-99th percentile so one spike doesn't wash out the colour scale; a sample is plenty.
  colourRange(col) {
    if (!col.range) {
      const step = Math.max(1, Math.floor(col.values.length / 200000));
      const sample = [];
      for (let i = 0; i < col.values.length; i += step) if (isFinite(col.values[i])) sample.push(col.values[i]);
      sample.sort((a, b) => a - b);
      col.range = sample.length ? [sample[Math.floor(sample.length * 0.01)], sample[Math.floor((sample.length - 1) * 0.99)]] : null;
    }
    return col.range;
  },

  // Padded [lo, hi] of arr, over the samples in idx (all when null).
  limits(arr, idx) {
    let lo = Infinity, hi = -Infinity;
    const n = idx ? idx.length : arr.length;
    for (let k = 0; k < n; k++) {
      const v = arr[idx ? idx[k] : k];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo)) return [0, 1];
    const pad = (hi - lo || 1) * 0.05;
    return [lo - pad, hi + pad];
  },

  // The chart wants Float32, with dates as seconds since the column's first sample; worked out once per column.
  chartValues(col) {
    if (!col.chart) {
      let t0 = 0, values = col.values;
      if (col.date) {
        t0 = values.find((v) => isFinite(v)) || 0;
        values = new Float32Array(col.values.length);
        for (let i = 0; i < values.length; i++) values[i] = (col.values[i] - t0) / 1000;
      }
      col.chart = { t0, values, lim: ManualQC.limits(values, null) };
    }
    return col.chart;
  },

  // Shown over the plot while it loads or redraws; text null hides it.
  loading(stage, text) {
    stage.classList.toggle('loading', !!text);
    if (text) stage.dataset.loading = text;
  },

  async draw() {
    const stage = ManualQC.host().querySelector('.manual-stage');
    if (!stage) return;
    const token = ManualQC.drawToken = (ManualQC.drawToken || 0) + 1;
    const message = (text) => {
      ManualQC.loading(stage, null);
      Plot.purge(stage); ManualQC.chart = null;
      stage.innerHTML = `<div class="hint review-empty">${escapeHtml(text)}</div>`;
      ManualQC.renderLegend({}); ManualQC.renderList();
    };
    // Nothing picked to QC yet: still show a plain PRES vs time plot.
    const plain = !ManualQC.group();
    const p = ManualQC.prof, view = plain ? 'time' : ManualQC.view;
    const { x: xv, y: yv } = plain ? { x: ManualQC.xVar(), y: ManualQC.yVar() } : ManualQC.axes();
    const cv = plain ? null : ManualQC.colourBy(), fv = plain ? null : ManualQC.flagVar();
    const single = view === 'profile' && p.by !== 'ALL';
    const names = [xv, yv, ...(plain ? [] : ManualQC.replayColumns(fv))];
    if (single) names.push(p.by);
    if (cv) names.push(cv);
    ManualQC.loading(stage, names.every((n) => n in ManualQC.cols) ? 'Drawing…' : 'Loading data…');
    let cols;
    try {
      cols = await ManualQC.columns(names);
    } catch (err) {
      if (token === ManualQC.drawToken) message('Could not load the data: ' + err.message);
      return;
    }
    if (token !== ManualQC.drawToken) return; // superseded by a later change
    if (!cols[xv] || !cols[yv]) { message(`No ${yv} or ${xv} in the data.`); return; }
    // Let the overlay paint before the loops below hold up the page.
    await new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve)));
    if (token !== ManualQC.drawToken) return;

    let idx = null, id = null, index = null;
    const label = p.by === 'CYCLE' ? 'Cycle' : 'Profile';
    if (single) {
      index = ManualQC.profileIndex(p.by, cols[p.by]);
      if (!index.ids.length) { message(`No ${label.toLowerCase()}s to show.`); return; }
      p.pos = Math.min(p.pos, index.ids.length - 1);
      id = index.ids[p.pos];
      idx = index.at.get(id);
      const bar = ManualQC.host().querySelector('.manual-plots');
      const idInput = bar && bar.querySelector('.manual-nav-id');
      if (idInput) { idInput.value = id; bar.querySelector('.manual-nav-of').textContent = `${p.pos + 1} of ${index.ids.length}`; }
    }

    const X = ManualQC.chartValues(cols[xv]), Y = ManualQC.chartValues(cols[yv]);
    const xs = X.values, ys = Y.values;
    const xdate = cols[xv].date;
    const xlim = idx ? ManualQC.limits(xs, idx) : X.lim;
    const ylim = idx ? ManualQC.limits(ys, idx) : Y.lim;
    const traces = [], data = {};
    const add = (x, y, t, rgba) => {
      data['0_' + traces.length] = { x, y, rgba: rgba || null };
      traces.push({ mode: 'markers', size: 4, opacity: 1, ...t });
    };
    const subset = (list, arr) => Float32Array.from(list, (i) => arr[i]);
    if (view === 'ts') {
      const levels = await ManualQC.sigma0(xlim, ylim);
      if (token !== ManualQC.drawToken) return;
      const lx = [], ly = [];
      for (const { lines } of levels) for (const line of lines) { for (const [s, t] of line) { lx.push(s); ly.push(t); } lx.push(NaN); ly.push(NaN); }
      data['0_' + traces.length] = { x: Float32Array.from(lx), y: Float32Array.from(ly), rgba: null };
      traces.push({ mode: 'lines', color: '#c3c9d0', opacity: 1, width: 1, dash: 'solid', size: 0, label: '_sigma0' });
    }
    const around = single ? [index.ids[p.pos - 1], index.ids[p.pos + 1]].filter((v) => v !== undefined).flatMap((v) => index.at.get(v)) : [];
    if (around.length) add(subset(around, xs), subset(around, ys), { label: 'neighbours', color: '#d0d4d8', size: 3 });

    // One pass to count, one to fill typed arrays: 10M+ points are too many for array filter/map.
    const values = ManualQC.values();
    const remainingGood = !values || values.flag_remaining_good !== false;
    const flags = plain ? null : ManualQC.replayFlags(fv, cols);
    const flagAt = (i) => (!flags ? 0 : remainingGood && flags[i] === 0 ? 1 : flags[i]);
    const col = cv && cols[cv];
    const cvals = col ? col.values : null;
    const n = idx ? idx.length : xs.length;
    const counts = {}, perFlag = {};
    let nNone = 0, nWith = 0;
    for (let k = 0; k < n; k++) {
      const i = idx ? idx[k] : k;
      if (!isFinite(xs[i]) || !isFinite(ys[i])) continue;
      const f = flagAt(i);
      const hasColour = !cvals || isFinite(cvals[i]);
      if (hasColour) counts[f] = (counts[f] || 0) + 1;
      if (ManualQC.hidden.has(f)) continue;
      if (!col) perFlag[f] = (perFlag[f] || 0) + 1;
      else if (hasColour) nWith++;
      else nNone++;
    }
    ManualQC.renderLegend(plain ? {} : counts);

    let cbar = null;
    if (!col) {
      const out = {};
      for (const f in perFlag) out[f] = { x: new Float32Array(perFlag[f]), y: new Float32Array(perFlag[f]), k: 0 };
      for (let k = 0; k < n; k++) {
        const i = idx ? idx[k] : k;
        if (!isFinite(xs[i]) || !isFinite(ys[i])) continue;
        const o = out[flagAt(i)];
        if (!o) continue;
        o.x[o.k] = xs[i]; o.y[o.k] = ys[i]; o.k++;
      }
      for (const f of Object.keys(out).map(Number).sort((a, b) => a - b)) {
        const meaning = (ManualQC.FLAGS.find(([m]) => m === f) || [f, ''])[1];
        add(out[f].x, out[f].y, { label: plain ? yv : `${f} (${meaning})`, color: ManualQC.COLOURS[f] });
      }
    } else {
      const hexRGB = (h) => [1, 3, 5].map((j) => parseInt(h.slice(j, j + 2), 16));
      const grey = hexRGB('#d0d4d8');
      // Categorical (e.g. SCI_PHASE): a fixed colour per category, as its own step plots it.
      const byValue = new Map((col.categories || []).map(([v, meaning, colour]) => [v, { label: `${v} ${meaning}`, colour, rgb: hexRGB(colour) }]));
      const seen = new Set();
      let colourOf;
      if (col.categories) {
        colourOf = (v) => {
          const cat = byValue.get(v);
          if (!cat) return grey;
          seen.add(v);
          return cat.rgb;
        };
        cbar = { label: cv, missing: nNone > 0 };
      } else {
        const range = ManualQC.colourRange(col);
        const stops = col.stops.map(hexRGB);
        const lo = range ? range[0] : 0, span = range ? (range[1] - range[0]) || 1 : 1;
        colourOf = (v) => stops[Math.round(Math.max(0, Math.min(1, (v - lo) / span)) * (stops.length - 1))];
        cbar = { label: cv, lo: range ? range[0] : null, hi: range ? range[1] : null, stops: col.stops, missing: nNone > 0 };
      }
      const none = { x: new Float32Array(nNone), y: new Float32Array(nNone), k: 0 };
      const withValue = { x: new Float32Array(nWith), y: new Float32Array(nWith), k: 0 };
      const rgba = new Uint8Array(nWith * 4);
      for (let k = 0; k < n; k++) {
        const i = idx ? idx[k] : k;
        if (!isFinite(xs[i]) || !isFinite(ys[i])) continue;
        if (ManualQC.hidden.has(flagAt(i))) continue;
        if (!isFinite(cvals[i])) { none.x[none.k] = xs[i]; none.y[none.k] = ys[i]; none.k++; continue; }
        const c = colourOf(cvals[i]), o = withValue.k * 4;
        rgba[o] = c[0]; rgba[o + 1] = c[1]; rgba[o + 2] = c[2]; rgba[o + 3] = 255;
        withValue.x[withValue.k] = xs[i]; withValue.y[withValue.k] = ys[i]; withValue.k++;
      }
      if (col.categories) cbar.categories = [...byValue].filter(([v]) => seen.has(v)).map(([, c]) => [c.label, c.colour]);
      if (nNone) add(none.x, none.y, { label: '_none', color: '#d0d4d8', size: 3 });
      add(withValue.x, withValue.y, { label: '_fill' }, rgba);
    }

    const title = {
      time: plain ? `${yv} vs time` : cv ? `${yv} vs time · coloured by ${cv}` : `${yv} vs time · coloured by ${fv} flag`,
      profile: single ? `${label} ${id} · ${n} samples` : `${yv} vs ${xv} · all profiles`,
      variable: `${yv} vs time · coloured by flag`,
      ts: cv ? `T–S · coloured by ${cv} · grey lines: σ0 (kg m⁻³)` : `T–S · coloured by ${fv} flag · grey lines: σ0 (kg m⁻³)`,
      map: cv ? `Position · coloured by ${cv}` : `Position · coloured by ${fv} flag`,
    }[view];
    const spec = {
      suptitle: '', t0: X.t0, points: n,
      panels: [{
        title, xlabel: xdate ? 'Time' : xv, ylabel: yv, xdate, ydate: false,
        xlim, ylim: ['PRES', 'DEPTH'].includes(yv) ? [ylim[1], ylim[0]] : ylim, xscale: 'linear', yscale: 'linear',
        legend: false, cell: null, share_x: 0, share_y: 0, traces, reflines: [], top_axis: null, cbar,
      }],
    };
    const old = ManualQC.chart;
    const key = [view, p.by, id, xv, yv].join('|');
    const keep = old && old.key === key ? old.panels[0].view : null;
    Plot.purge(stage);
    ManualQC.loading(stage, null);
    const chart = new Chart(stage, spec, data, null, {
      onSelect: (rect) => ManualQC.addBox(rect),
      onPoint: (pt) => ManualQC.addBox({ x0: pt.x, y0: pt.y }),
      onView: () => ManualQC.zoomHint(stage, chart),
      onRemove: (o) => ManualQC.remove(o.index),
      onDoubleClick: () => ManualQC.undoClicks(),
    });
    stage._chart = chart;
    chart.key = key;
    if (keep) { chart.panels[0].view = keep; chart.draw(); }
    ManualQC.zoomHint(stage, chart);
    ManualQC.chart = chart;
    ManualQC.syncOverlays();
    ManualQC.renderList();
  },

  onKey(e) {
    if (!ManualQC.isActive() || !ManualQC.tab().classList.contains('on')) return;
    if (e.target.closest('input, select, textarea, .CodeMirror')) return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) ManualQC.redo(); else ManualQC.undo();
      return;
    }
    if (ManualQC.view !== 'profile' || ManualQC.prof.by === 'ALL' || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      ManualQC.step(e.key === 'ArrowLeft' ? -1 : 1);
    }
  },

  // The chart uses seconds since spec.t0 on a date axis and log10 on a log axis.
  toChart(v, isDate, log) {
    if (isDate) { const ms = Date.parse(v); return isNaN(ms) ? NaN : (ms - ManualQC.chart.spec.t0) / 1000; }
    const n = Number(v);
    return log ? Math.log10(n) : n;
  },
  fromChart(v, isDate, log) {
    if (isDate) return new Date(ManualQC.chart.spec.t0 + v * 1000).toISOString().replace(/\.000Z$/, 'Z');
    return +(log ? Math.pow(10, v) : v).toPrecision(7);
  },
  panelAxes() {
    const p = ManualQC.chart.spec.panels[0];
    return { xd: p.xdate, yd: p.ydate, xl: p.xscale === 'log', yl: p.yscale === 'log' };
  },

  chartBoxes() {
    const { xd, yd, xl, yl } = ManualQC.panelAxes();
    const out = [];
    ManualQC.shownBoxes().forEach(({ b, i }) => {
      if (!Array.isArray(b.x) || !Array.isArray(b.y)) return;
      const x = b.x.map((v) => ManualQC.toChart(v, xd, xl)), y = b.y.map((v) => ManualQC.toChart(v, yd, yl));
      if (![...x, ...y].every(isFinite)) return;
      out.push({
        index: i, box: b, point: x.length === 1,
        x0: Math.min(...x), x1: Math.max(...x), y0: Math.min(...y), y1: Math.max(...y),
      });
    });
    return out;
  },

  zoomHint(stage, chart) {
    let hint = stage.querySelector('.manual-zoom-hint');
    if (!hint) {
      hint = Forms.el('button', { type: 'button', class: 'manual-zoom-hint' });
      stage.appendChild(hint);
    }
    const p = chart.panels[0];
    const zoomed = ['x0', 'x1', 'y0', 'y1'].some((k) => p.view[k] !== p.home[k]);
    hint.textContent = zoomed ? 'Double-click to reset zoom' : `${ManualQC.mod()}- or Shift-drag to zoom`;
    hint.classList.toggle('on', zoomed);
    hint.onclick = zoomed ? () => chart.reset() : null;
  },

  syncOverlays() {
    const chart = ManualQC.chart;
    if (!chart) return;
    chart.setOverlays(ManualQC.chartBoxes().filter((c) => !ManualQC.hidden.has(c.box.flag)).map((c) => ({
      ...c,
      color: ManualQC.border(c.box.flag),
      label: `${c.box.flag}${(c.box.mode || 'inside') === 'outside' ? ' outside' : ''}${c.box.override === false ? ' merge' : ''}`,
      hi: c.index === ManualQC.hi || c.index === ManualQC.sel,
    })));
  },

  // {x0, x1, y0, y1} is a box; {x0, y0} alone is a point (nearest sample).
  addBox(rect) {
    const values = ManualQC.values();
    const group = ManualQC.group();
    if (!values || !group) return;
    const { xd, yd, xl, yl } = ManualQC.panelAxes();
    const { x, y } = ManualQC.axes();
    const point = rect.x1 === undefined;
    if (point) {
      const burst = ManualQC.clicks && Date.now() - ManualQC.clicks.at < 600;
      ManualQC.clicks = { at: Date.now(), before: burst ? ManualQC.clicks.before : JSON.stringify(values.boxes) };
    }
    const box = {
      x: point ? [ManualQC.fromChart(rect.x0, xd, xl)]
        : [ManualQC.fromChart(rect.x0, xd, xl), ManualQC.fromChart(rect.x1, xd, xl)],
      y: point ? [ManualQC.fromChart(rect.y0, yd, yl)]
        : [ManualQC.fromChart(rect.y0, yd, yl), ManualQC.fromChart(rect.y1, yd, yl)],
      flag: ManualQC.tool.flag,
      mode: ManualQC.tool.mode,
      x_variable: x,
      y_variable: y,
      variables: [...group],
    };
    if (ManualQC.view === 'profile' && ManualQC.prof.by !== 'ALL') {
      box[ManualQC.prof.by === 'CYCLE' ? 'cycles' : 'profiles'] = [ManualQC.profileId()];
    }
    if (!ManualQC.tool.override) box.override = false;
    values.boxes.push(box);
    ManualQC.commit();
  },

  remove(index) {
    const values = ManualQC.values();
    if (!values || !values.boxes[index]) return;
    values.boxes.splice(index, 1);
    ManualQC.hi = null; ManualQC.sel = null;
    ManualQC.commit();
  },

  // Nothing runs until Apply: the plot replays the boxes the way the test will.
  commit({ record = true } = {}) {
    if (record) ManualQC.record();
    ManualQC.history();
    STATE.onChange();
    renderPipeline();
    const bubbles = ManualQC.host().querySelector('.manual-plots');
    if (bubbles) bubbles.replaceWith(ManualQC.toolbar());
    ManualQC.draw();
    ManualQC.dirty = true;
    ManualQC.buttons();
  },

  // Re-run reads the YAML pane, which commit() refreshed synchronously.
  apply(thenContinue = false) {
    ManualQC.dirty = false;
    ManualQC.buttons();
    if (ManualQC.chart) ManualQC.chart.busy = true;
    Run.rerunStep({ thenContinue });
  },

  // Re-runs first if needed; Run.showPause continues once that pause lands.
  applyAndContinue() {
    if (!ManualQC.dirty) { Run.continueRun(); return; }
    ManualQC.apply(true);
  },

  setBusy(on) {
    ManualQC.host().classList.toggle('busy', on);
    ManualQC.rail().classList.toggle('busy', on);
  },

  // Unapplied boxes make Apply the main action.
  buttons() {
    const rerun = document.getElementById('btn-rerun');
    const cont = document.getElementById('btn-run');
    rerun.lastChild.textContent = ManualQC.dirty ? 'Apply' : 'Re-run';
    rerun.classList.toggle('primary', ManualQC.dirty);
    cont.classList.toggle('primary', !ManualQC.dirty);
    cont.title = ManualQC.dirty ? 'Applies your boxes first' : '';
  },

  renderList() {
    const list = document.getElementById('manual-list');
    if (!list) return;
    list.innerHTML = '';
    const values = ManualQC.values();
    const boxes = values ? values.boxes : [];
    const here = new Set(ManualQC.chart ? ManualQC.shownBoxes().map(({ i }) => i) : []);
    const mine = boxes.map((b, i) => ({ b, i })).filter(({ b }) => b && ManualQC.inGroup(b, ManualQC.group()));
    if (!mine.length) {
      const hint = document.createElement('div');
      hint.className = 'hint';
      hint.textContent = 'No boxes yet. Drag on the plot to draw one, or click a point to flag just it.';
      list.appendChild(hint);
      return;
    }
    const short = (v) => {
      const s = String(v);
      const m = s.match(/^\d{4}-(\d\d-\d\d)T(\d\d:\d\d)/);
      return m ? `${m[1]} ${m[2]}` : (isFinite(+s) ? String(+(+s).toPrecision(4)) : s);
    };
    mine.forEach(({ b, i }) => {
      const row = document.createElement('div');
      row.className = 'manual-row' + (i === ManualQC.hi ? ' hi' : '') + (i === ManualQC.sel ? ' sel' : '') + (here.has(i) ? '' : ' elsewhere');
      row.style.setProperty('--fc', ManualQC.COLOURS[b.flag] || '#333');
      // Later boxes win on overlap, so order is part of the config.
      row.draggable = true;
      row.ondragstart = (e) => { ManualQC.drag = i; e.dataTransfer.effectAllowed = 'move'; row.classList.add('dragging'); };
      row.ondragend = () => { ManualQC.drag = null; row.classList.remove('dragging'); };
      row.ondragover = (e) => { e.preventDefault(); row.classList.add('over'); };
      row.ondragleave = () => row.classList.remove('over');
      row.ondrop = (e) => {
        e.preventDefault(); row.classList.remove('over');
        const from = ManualQC.drag;
        if (from === null || from === i) return;
        const [moved] = boxes.splice(from, 1);
        boxes.splice(i, 0, moved);
        ManualQC.sel = null;
        ManualQC.commit();
      };
      row.onmouseenter = () => { ManualQC.hi = i; ManualQC.syncOverlays(); };
      row.onmouseleave = () => { ManualQC.hi = null; ManualQC.syncOverlays(); };
      row.onclick = (e) => {
        if (e.target.closest('.manual-row-edit, .manual-row-rm')) return;
        if (!here.has(i)) { ManualQC.goToBox(i); return; }
        ManualQC.sel = ManualQC.sel === i ? null : i;
        ManualQC.renderList(); ManualQC.syncOverlays();
      };

      const head = document.createElement('div');
      head.className = 'manual-row-head';
      const sw = document.createElement('span');
      sw.className = 'sw'; sw.textContent = b.flag;
      head.appendChild(sw);
      const txt = document.createElement('span');
      txt.className = 'manual-row-text';
      const point = b.x.length === 1;
      txt.textContent = (point ? 'point ' : (b.mode === 'outside' ? 'outside ' : '')) +
        (point ? `${short(b.x[0])}, ${short(b.y[0])}` : `${short(b.y[0])}–${short(b.y[1])}`) +
        (b.profiles ? ` · profile ${b.profiles.join(', ')}` : b.cycles ? ` · cycle ${b.cycles.join(', ')}` : '');
      head.appendChild(txt);
      head.appendChild(Forms.el('span', { class: 'manual-row-view', textContent: ManualQC.boxView(b).label }));
      head.appendChild(Forms.button('×', { cls: 'icon-btn reveal manual-row-rm', title: 'remove this box',
        onclick: (e) => { e.stopPropagation(); ManualQC.remove(i); } }));
      row.appendChild(head);

      if (i === ManualQC.sel) {
        const edit = document.createElement('div');
        edit.className = 'manual-row-edit';
        edit.appendChild(Forms.select(ManualQC.FLAGS.map(([f, m]) => [f, `${f} ${m}`]), b.flag,
          (v) => { b.flag = Number(v); ManualQC.commit(); }, { cls: 'sm' }));
        edit.appendChild(Forms.select(['inside', 'outside'], b.mode || 'inside',
          (v) => { b.mode = v; ManualQC.commit(); }, { cls: 'sm' }));
        const ov = document.createElement('label');
        ov.className = 'manual-row-ov'; ov.title = 'override existing flags';
        const ovc = document.createElement('input');
        ovc.type = 'checkbox'; ovc.checked = b.override !== false;
        ovc.onchange = () => { if (ovc.checked) delete b.override; else b.override = false; ManualQC.commit(); };
        ov.appendChild(ovc); ov.appendChild(document.createTextNode('override'));
        edit.appendChild(ov);
        const detail = document.createElement('div');
        detail.className = 'manual-row-detail';
        detail.textContent = `${ManualQC.boxX(b)} ${b.x.map(short).join(' → ')}\n${ManualQC.boxY(b)} ${b.y.map(short).join(' → ')}`;
        edit.appendChild(detail);
        row.appendChild(edit);
      }
      list.appendChild(row);
    });
  },
};

document.addEventListener('keydown', ManualQC.onKey);
// Unapplied boxes live only in this page.
window.addEventListener('beforeunload', (e) => { if (ManualQC.isActive() && ManualQC.dirty) e.preventDefault(); });
