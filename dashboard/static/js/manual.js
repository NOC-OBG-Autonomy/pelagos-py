// Manual QC: its own tab, taking the whole window while a run is paused on the
// `manual qc` test.
//
// The test's plot (y_variable vs x_variable, coloured by flag) is the editor.
// A sidebar holds the tools — axis pickers, the flag to give, inside/outside,
// which variables to flag — and every box dragged (or single point clicked) on
// the chart is added with those settings straight into the test's `boxes`
// parameter (the same values object the builder card and YAML show), then the
// test is re-run so the plot refreshes with the flags applied. The box list
// edits the same values. So the config is the only state: what you drew is
// what a later run repeats.

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
  // Argo merge table, COMBINATRIX[existing][new]: the one in
  // pelagos_py.utils.qc_handling, shipped by /api/registry so there is a single copy.
  get COMBINATRIX() { return STATE.registry.combinatrix; },
  // Box outlines: a darker shade of the flag colour, so a box stays visible
  // over points of the same flag.
  border(flag) {
    const hex = ManualQC.COLOURS[flag] || '#333333';
    const c = parseInt(hex.slice(1), 16);
    const d = (v) => Math.round(v * 0.55).toString(16).padStart(2, '0');
    return '#' + d(c >> 16) + d((c >> 8) & 255) + d(c & 255);
  },

  // Sidebar tool state; survives re-runs and tab switches within one pause.
  tool: { flag: 4, mode: 'inside', override: true, vars: null, draw: true },
  chart: null,
  spec: null,
  fig: null,
  hi: null,      // box index highlighted from the list (hover)
  sel: null,     // box index selected in the list (click) — shows its editor
  dirty: false,  // boxes edited since the last re-run: the plot is a live preview
  continueAfter: false, // Apply & continue: continue once the re-run pauses
  view: 'time',  // 'time': the test's plot; 'profile': one profile (or cycle) at a time
  // Profile view: the time plot cut down to the `by` id at `pos`, coloured by flag or `colour`.
  prof: { by: 'PROFILE_NUMBER', pos: 0, colour: null },
  pchart: null,
  cols: {},      // column name -> promise of {values, date, stops}, fetched once per pause
  index: {},     // PROFILE_NUMBER / CYCLE -> {ids, at: Map id -> sample indices}
  nearest: {},   // point box (JSON) -> its sample, for the profile view's flag replay

  isActive() { return Review.active && Review.test === ManualQC.TEST; },
  tab() { return document.querySelector('.tab[data-tab="manual"]'); },
  host() { return document.getElementById('manual-workspace'); },
  rail() { return document.getElementById('manual-rail'); },

  values() {
    const v = Review.testValues();
    if (v && !Array.isArray(v.boxes)) v.boxes = [];
    return v;
  },

  xVar() { const v = ManualQC.values(); return (v && v.x_variable) || 'TIME'; },
  yVar() { const v = ManualQC.values(); return (v && v.y_variable) || 'PRES'; },
  // The plot a box belongs to: its own y_variable, else the test's.
  boxY(b) { return b.y_variable || ManualQC.yVar(); },
  boxX(b) { return b.x_variable || ManualQC.xVar(); },
  // Boxes from the profile view carry the profiles (or cycles) they are limited to.
  timeBox(b) { return b && !b.profiles && !b.cycles; },
  // Boxes drawn on the plot being shown, with their index in the config list. The profile
  // view shows the time plot's boxes plus those limited to the profile on screen.
  shownBoxes() {
    const values = ManualQC.values();
    const key = ManualQC.prof.by === 'CYCLE' ? 'cycles' : 'profiles';
    const inView = (b) => ManualQC.view === 'time' || ManualQC.timeBox(b)
      || (b[key] || []).map(Number).includes(ManualQC.profileId());
    return (values ? values.boxes : []).map((b, i) => ({ b, i }))
      .filter(({ b }) => b && ManualQC.boxX(b) === ManualQC.xVar() && ManualQC.boxY(b) === ManualQC.yVar() && inView(b));
  },
  // Plots with boxes on them, the current one first.
  plots() {
    const values = ManualQC.values();
    const out = [ManualQC.yVar()];
    for (const b of (values ? values.boxes : [])) if (b && !out.includes(ManualQC.boxY(b))) out.push(ManualQC.boxY(b));
    return out;
  },

  // Switch the plot: boxes on the old one keep it as their y_variable, the
  // tools reset to the new one, and the test re-runs to draw it.
  setPlot(y) {
    const values = ManualQC.values();
    if (!values || y === ManualQC.yVar()) return;
    for (const b of values.boxes) if (b && !b.y_variable) b.y_variable = ManualQC.yVar();
    values.y_variable = y;
    ManualQC.tool.vars = null;
    ManualQC.sel = null; ManualQC.hi = null;
    ManualQC.commit();
    ManualQC.apply();
  },

  // Variables offered by the pickers: what the runner reported at this pause,
  // plus whatever the config already names.
  variables() {
    const out = [...(Run.variables || [])];
    for (const v of [ManualQC.xVar(), ManualQC.yVar()]) if (!out.includes(v)) out.push(v);
    return out;
  },

  targets() {
    if (!ManualQC.tool.vars) ManualQC.tool.vars = new Set([ManualQC.yVar()]);
    return ManualQC.tool.vars;
  },

  // ---- lifecycle ----
  // Called when the run pauses on the test: reveal the tab and jump to it.
  open() {
    ManualQC.tool.vars = null;
    ManualQC.cols = {}; ManualQC.index = {}; ManualQC.nearest = {};
    ManualQC.hi = null; ManualQC.sel = null;
    ManualQC.dirty = false; ManualQC.continueAfter = false;
    ManualQC.tab().classList.remove('hidden');
    ManualQC.host().innerHTML = ''; ManualQC.rail().innerHTML = '';
    Run.showTab('manual');
  },

  close() {
    ManualQC.chart = null; ManualQC.spec = null; ManualQC.fig = null; ManualQC.pchart = null;
    ManualQC.dirty = false; ManualQC.continueAfter = false;
    ManualQC.buttons();
    const tab = ManualQC.tab();
    tab.classList.add('hidden');
    if (tab.classList.contains('on')) Run.showTab('run');
    ManualQC.host().innerHTML = ''; ManualQC.rail().innerHTML = '';
  },

  // A new (or first) figure for this pause: rebuild the chart, keep the tools.
  // Same figure again (attempt strip clicks, "Use these"): just refresh the boxes.
  render(fig) {
    const host = ManualQC.host();
    if (fig && fig === ManualQC.fig && host.children.length) {
      ManualQC.renderList(); ManualQC.syncOverlays(); ManualQC.preview(); return;
    }
    ManualQC.fig = fig;
    ManualQC.dirty = false;
    ManualQC.buttons();
    host.innerHTML = ''; ManualQC.rail().innerHTML = '';
    if (!fig) {
      const hint = document.createElement('div');
      hint.className = 'hint review-empty';
      hint.textContent = Review.busy ? 'Re-running — the plot will appear here.'
        : 'The test produced no figure. Re-run it with diagnostics on, or Continue.';
      host.appendChild(hint);
      return;
    }
    if (!ManualQC.hasProfiles()) ManualQC.view = 'time';
    ManualQC.rail().appendChild(ManualQC.sidebar());
    ManualQC.renderList();
    host.appendChild(ManualQC.view === 'profile' ? ManualQC.profileBar() : ManualQC.plotBar());
    const stage = document.createElement('div');
    stage.className = 'manual-stage' + (ManualQC.view === 'profile' ? ' hidden' : '');
    stage.innerHTML = '<div class="viewer-loading">Loading plot…</div>';
    host.appendChild(stage);
    const pstage = document.createElement('div');
    pstage.className = 'manual-stage manual-pstage' + (ManualQC.view === 'profile' ? '' : ' hidden');
    host.appendChild(pstage);
    if (ManualQC.pchart) ManualQC.pchart.destroy();
    ManualQC.pchart = null;
    if (ManualQC.view === 'profile') ManualQC.drawProfile();

    ManualQC.chart = null;
    Plot.fetchSpec(fig.spec)
      .then((spec) => Plot.render(stage, spec, {
        name: fig.spec,
        manual: {
          draw: () => ManualQC.tool.draw,
          onSelect: (rect) => ManualQC.addBox(rect),
          onPoint: (pt) => ManualQC.addBox({ x0: pt.x, y0: pt.y }),
          onView: () => ManualQC.dimProfile(),
          onRemove: (o) => ManualQC.remove(o.index),
        },
      }).then((chart) => { ManualQC.spec = spec; ManualQC.chart = chart; ManualQC.syncOverlays(); ManualQC.preview(); }))
      .catch((err) => {
        console.error('Manual QC interactive view failed', err);
        Plot.purge(stage);
        stage.innerHTML = '';
        stage.appendChild(Viewer.card([fig], 0, { cls: 'big' }));
        const note = document.createElement('div');
        note.className = 'hint';
        note.textContent = 'Interactive view unavailable' + (err && err.message ? ' — ' + err.message : '') +
          '. Boxes can still be edited in the list.';
        stage.appendChild(note);
      });
  },

  // ---- sidebar ----
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

    const mouse = ManualQC.section('Mouse');
    const hint = document.createElement('div');
    hint.className = 'hint';
    const mod = navigator.platform.toLowerCase().includes('mac') ? '⌘' : 'Ctrl';
    const setHint = () => {
      hint.textContent = t.draw
        ? `Drag to draw a box · click a point to flag just it · ${mod}-drag to zoom · double-click resets zoom`
        : `Drag to zoom · ${mod}-drag to draw a box · ${mod}-click a point to flag just it · double-click resets zoom`;
    };
    setHint();
    mouse.appendChild(Forms.seg([[true, 'Draw boxes'], [false, 'Zoom']], t.draw, (draw) => { t.draw = draw; setHint(); }, 'fill'));
    mouse.appendChild(hint);
    tools.appendChild(mouse);

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

    const vars = ManualQC.section('Apply flag to');
    vars.appendChild(ManualQC.varPicker());
    tools.appendChild(vars);

    const rest = ManualQC.section('Remaining points');
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
    row.appendChild(document.createTextNode(' Flag untouched points 1 (good)'));
    rest.appendChild(row);
    const rh = document.createElement('div');
    rh.className = 'hint';
    rh.textContent = 'Off leaves them at 0 (no QC) for a later step.';
    rest.appendChild(rh);
    tools.appendChild(rest);

    const boxes = ManualQC.section('Boxes');
    boxes.classList.add('manual-boxes');
    const list = document.createElement('div');
    list.className = 'manual-list'; list.id = 'manual-list';
    boxes.appendChild(list);
    side.appendChild(boxes);
    return side;
  },

  // Multi-select dropdown: a summary button that opens a searchable list of
  // checkboxes. Closes on a click anywhere else.
  varPicker() {
    const chosen = ManualQC.targets();
    const wrap = document.createElement('div');
    wrap.className = 'manual-pick';
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'selectish manual-pick-btn';
    const summary = () => {
      const v = [...chosen];
      btn.textContent = v.length ? v.join(', ') : 'Choose variables…';
      btn.classList.toggle('empty', !v.length);
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
    const fill = () => {
      opts.innerHTML = '';
      const q = search.value.trim().toLowerCase();
      for (const v of ManualQC.variables()) {
        if (v === 'TIME' || (q && !v.toLowerCase().includes(q))) continue;
        const l = document.createElement('label');
        const c = document.createElement('input');
        c.type = 'checkbox'; c.checked = chosen.has(v);
        c.onchange = () => { if (c.checked) chosen.add(v); else chosen.delete(v); summary(); };
        l.appendChild(c); l.appendChild(document.createTextNode(v));
        opts.appendChild(l);
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

  // Plot switcher above the chart: one tab per plot with boxes on it (the
  // current one included), plus a picker to open another y variable.
  // x stays TIME: boxes hold ISO timestamps, which a numeric axis cannot read.
  plotBar() {
    const bar = document.createElement('div');
    bar.className = 'manual-plots';
    bar.appendChild(ManualQC.viewSeg());
    const values = ManualQC.values();
    const cur = ManualQC.yVar();
    const seg = Forms.el('div', { class: 'seg' });
    for (const y of ManualQC.plots()) {
      const n = (values ? values.boxes : []).filter((b) => b && ManualQC.boxY(b) === y).length;
      seg.appendChild(Forms.button('', { cls: 'manual-plot' + (y === cur ? ' on' : ''), onclick: () => ManualQC.setPlot(y),
        html: `<span>${y}</span>` + (n ? `<i>${n}</i>` : ''),
        title: `${y} vs ${ManualQC.xVar()}` + (n ? ` · ${n} box${n === 1 ? '' : 'es'}` : '') }));
    }
    bar.appendChild(seg);
    const vars = ManualQC.variables().filter((v) => v !== 'TIME');
    bar.appendChild(Forms.select(vars.filter((v) => !ManualQC.plots().includes(v)), null,
      (v) => { if (v) ManualQC.setPlot(v); }, { placeholder: '+ plot', cls: 'sm' }));
    const x = document.createElement('span');
    x.className = 'hint manual-plot-x'; x.textContent = 'vs ' + ManualQC.xVar();
    bar.appendChild(x);
    // Colour by a third variable: flags then show as rings around non-good points.
    const cl = document.createElement('label');
    cl.className = 'hint manual-plot-cl'; cl.textContent = 'colour by';
    bar.appendChild(cl);
    const curC = (values && values.colour_variable) || '';
    const colour = Forms.select(vars, curC, (v) => {
      const vals = ManualQC.values();
      if (!vals) return;
      if (v) vals.colour_variable = v; else delete vals.colour_variable;
      ManualQC.commit();
      ManualQC.apply();
    }, { placeholder: 'flag', cls: 'sm manual-plot-colour' });
    colour.title = 'Colour points by a variable (flags become rings)';
    cl.onclick = () => colour.focus();
    bar.appendChild(colour);
    // Profile side panel (colour variable vs y), greyed outside the main zoom.
    const prof = document.createElement('button');
    prof.type = 'button';
    const on = !!(values && values.profile_plot && curC);
    prof.className = 'sm manual-plot' + (on ? ' on' : '');
    prof.textContent = 'Profile';
    prof.disabled = !curC;
    prof.title = curC ? 'Side panel: ' + curC + ' vs ' + cur + ', greyed outside the main zoom' : 'Pick a colour variable first';
    prof.onclick = () => {
      const vals = ManualQC.values();
      if (!vals) return;
      if (vals.profile_plot) delete vals.profile_plot; else vals.profile_plot = true;
      ManualQC.commit();
      ManualQC.apply();
    };
    bar.appendChild(prof);
    return bar;
  },

  // ---- profile view ----
  // One profile (or cycle) at a time, drawn here from raw columns the paused run
  // hands over (API.runColumns): stepping and swapping variables never re-run the test.
  hasProfiles() { return (Run.variables || []).includes('PROFILE_NUMBER'); },

  viewSeg() {
    const seg = Forms.seg([['time', 'Time'], ['profile', 'Profiles']], ManualQC.view, (v) => ManualQC.setView(v), 'sm');
    if (!ManualQC.hasProfiles()) {
      seg.lastChild.disabled = true;
      seg.lastChild.title = 'Needs PROFILE_NUMBER: run Find Profiles before this QC step';
    }
    return seg;
  },

  setView(view) {
    if (view === ManualQC.view) return;
    ManualQC.view = view;
    const values = ManualQC.values();
    if (ManualQC.prof.colour === null) ManualQC.prof.colour = (values && values.colour_variable) || '';
    ManualQC.sel = null; ManualQC.hi = null;
    const host = ManualQC.host();
    host.querySelector('.manual-plots').replaceWith(view === 'profile' ? ManualQC.profileBar() : ManualQC.plotBar());
    const [stage, pstage] = host.querySelectorAll('.manual-stage');
    stage.classList.toggle('hidden', view === 'profile');
    pstage.classList.toggle('hidden', view !== 'profile');
    ManualQC.rail().innerHTML = '';
    ManualQC.rail().appendChild(ManualQC.sidebar());
    ManualQC.renderList();
    if (view === 'profile') ManualQC.drawProfile(); else { ManualQC.syncOverlays(); ManualQC.preview(); }
  },

  // Columns by name, each fetched from the run once per pause (missing ones in one request).
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

  // Sample indices of every profile (or cycle) id, built once per pause.
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
    ManualQC.drawProfile();
  },

  // Profile <-> cycle keeps your place: the new id is the one holding the current profile's first sample.
  setBy(by) {
    const p = ManualQC.prof;
    const old = ManualQC.index[p.by];
    const first = old && old.at.get(old.ids[p.pos]);
    p.by = by;
    p.pos = 0;
    ManualQC.columns([by]).then((cols) => {
      const index = ManualQC.profileIndex(by, cols[by]);
      if (first) p.pos = Math.max(0, index.ids.indexOf(cols[by].values[first[0]]));
      ManualQC.drawProfile();
    });
  },

  profileBar() {
    const p = ManualQC.prof;
    const bar = Forms.el('div', { class: 'manual-plots' });
    bar.appendChild(ManualQC.viewSeg());
    const by = Forms.seg([['PROFILE_NUMBER', 'Profile'], ['CYCLE', 'Cycle']], p.by, (v) => ManualQC.setBy(v), 'sm');
    if (!ManualQC.variables().includes('CYCLE')) by.lastChild.disabled = true;
    bar.appendChild(by);
    const nav = Forms.el('div', { class: 'manual-nav' });
    nav.appendChild(Forms.button('◀', { cls: 'sm', title: 'Previous (←)', onclick: () => ManualQC.step(-1) }));
    const id = Forms.el('input', { type: 'number', class: 'sm manual-nav-id', title: 'Jump to an id' });
    id.onchange = () => {
      const index = ManualQC.index[p.by];
      if (!index) return;
      // Nearest id at or after the one typed: ids can have gaps.
      const at = index.ids.findIndex((v) => v >= Number(id.value));
      p.pos = at < 0 ? index.ids.length - 1 : at;
      ManualQC.drawProfile();
    };
    nav.appendChild(id);
    nav.appendChild(Forms.el('span', { class: 'hint manual-nav-of' }));
    nav.appendChild(Forms.button('▶', { cls: 'sm', title: 'Next (→)', onclick: () => ManualQC.step(1) }));
    bar.appendChild(nav);
    bar.appendChild(Forms.el('span', { class: 'hint manual-plot-x', textContent: `${ManualQC.yVar()} vs ${ManualQC.xVar()}` }));
    bar.appendChild(Forms.el('span', { class: 'hint manual-plot-cl', textContent: 'colour by' }));
    const colours = ManualQC.variables().filter((v) => v !== 'TIME');
    bar.appendChild(Forms.select(colours, p.colour, (v) => { p.colour = v; ManualQC.drawProfile(); },
      { placeholder: 'flag', cls: 'sm manual-plot-colour' }));
    return bar;
  },

  // Whether sample i is hit by box b (its mode applied), mirroring manual_qc._box_mask;
  // null when a column it needs is missing.
  boxHit(b, cols) {
    const X = cols[ManualQC.boxX(b)], Y = cols[ManualQC.boxY(b)];
    if (!X || !Y) return null;
    const x = X.values, y = Y.values;
    const scope = b.profiles ? cols.PROFILE_NUMBER : b.cycles ? cols.CYCLE : null;
    const ids = (b.profiles || b.cycles || []).map(Number);
    const valid = (i) => isFinite(x[i]) && isFinite(y[i]) && (!scope || ids.includes(scope.values[i]));
    // Naive ISO times are UTC, as pandas reads them.
    const bound = (v, col) => col.date ? Date.parse(/T/.test(v) && !/(Z|[+-]\d\d:?\d\d)$/i.test(v) ? v + 'Z' : v) : Number(v);
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

  // Flags the test will give `v` at samples `idx`: its flags from before the test with
  // every box replayed in order (mirrors manual_qc.return_qc), so unapplied boxes show too.
  replayFlags(v, idx, cols) {
    const values = ManualQC.values();
    const start = cols[v + '_QC'].values;
    const flags = idx.map((i) => start[i]);
    for (const b of (values ? values.boxes : [])) {
      if (!b || !(b.variables || [ManualQC.boxY(b)]).includes(v)) continue;
      const hit = ManualQC.boxHit(b, cols);
      if (!hit) continue;
      idx.forEach((i, k) => {
        if (flags[k] === 9 || !hit(i)) return;
        flags[k] = b.override === false ? ManualQC.COMBINATRIX[flags[k]][b.flag] : b.flag;
      });
    }
    if (!values || values.flag_remaining_good !== false) flags.forEach((f, k) => { if (f === 0) flags[k] = 1; });
    return flags;
  },

  // 1st-99th percentile, so one spike doesn't wash out the colour scale; same for every profile.
  colourRange(col) {
    if (!col.range) {
      const finite = col.values.filter((v) => isFinite(v)).sort();
      col.range = finite.length ? [finite[Math.floor(finite.length * 0.01)], finite[Math.floor((finite.length - 1) * 0.99)]] : null;
    }
    return col.range;
  },

  async drawProfile() {
    const p = ManualQC.prof;
    const stage = ManualQC.host().querySelector('.manual-pstage');
    if (!stage) return;
    const token = ManualQC.drawToken = (ManualQC.drawToken || 0) + 1;
    const values = ManualQC.values();
    const xv = ManualQC.xVar(), yv = ManualQC.yVar();
    const names = [p.by, xv, yv, yv + '_QC'];
    if (p.colour) names.push(p.colour);
    for (const b of (values ? values.boxes : [])) {
      if (!b || !(b.variables || [ManualQC.boxY(b)]).includes(yv)) continue;
      names.push(ManualQC.boxX(b), ManualQC.boxY(b));
      if (b.profiles) names.push('PROFILE_NUMBER');
      if (b.cycles) names.push('CYCLE');
    }
    if (!ManualQC.pchart) stage.innerHTML = '<div class="viewer-loading">Loading profiles…</div>';
    let cols;
    try {
      cols = await ManualQC.columns(names);
    } catch (err) {
      if (token !== ManualQC.drawToken) return;
      Plot.purge(stage); ManualQC.pchart = null;
      stage.innerHTML = `<div class="hint review-empty">Could not load the data: ${escapeHtml(err.message)}</div>`;
      return;
    }
    if (token !== ManualQC.drawToken) return; // a later step or change superseded this one
    const index = ManualQC.profileIndex(p.by, cols[p.by]);
    const bar = ManualQC.host().querySelector('.manual-plots');
    const label = p.by === 'CYCLE' ? 'Cycle' : 'Profile';
    if (!index.ids.length || !cols[xv] || !cols[yv]) {
      Plot.purge(stage); ManualQC.pchart = null;
      stage.innerHTML = `<div class="hint review-empty">No ${label.toLowerCase()}s to show.</div>`;
      return;
    }
    p.pos = Math.min(p.pos, index.ids.length - 1);
    const id = index.ids[p.pos];
    const idInput = bar && bar.querySelector('.manual-nav-id');
    if (idInput) { idInput.value = id; bar.querySelector('.manual-nav-of').textContent = `${p.pos + 1} of ${index.ids.length}`; }

    const idx = index.at.get(id);
    const xs = cols[xv].values, ys = cols[yv].values;
    // Dates go over as seconds since the profile's first sample, as the chart expects.
    const xdate = cols[xv].date;
    const t0 = xdate ? xs[idx[0]] : 0;
    const pick = (list, arr) => Float32Array.from(list, (i) => (arr === xs ? (arr[i] - t0) / (xdate ? 1000 : 1) : arr[i]));
    const traces = [], data = {};
    const add = (list, t, rgba) => {
      data['0_' + traces.length] = { x: pick(list, xs), y: pick(list, ys), rgba: rgba || null };
      traces.push({ mode: 'markers', size: 4, opacity: 1, ...t });
    };
    // Neighbouring profiles in grey behind, for context.
    const around = [index.ids[p.pos - 1], index.ids[p.pos + 1]].filter((v) => v !== undefined).flatMap((v) => index.at.get(v));
    if (around.length) add(around, { label: 'neighbours', color: '#d0d4d8', size: 3 });
    const flags = ManualQC.replayFlags(yv, idx, cols);
    const col = p.colour && cols[p.colour];
    let cbar = null;
    for (const f of [...new Set(flags)].sort()) {
      const list = idx.filter((_, k) => flags[k] === f);
      const meaning = (ManualQC.FLAGS.find(([n]) => n === f) || [f, ''])[1];
      // Coloured by a variable: flags show as rings under the fill (good ones not at all).
      if (col && f === 1) continue;
      add(list, { label: `${f} (${meaning})`, color: ManualQC.COLOURS[f], size: col ? 7.6 : 4 });
    }
    if (col && col.categories) {
      // A categorical variable (e.g. SCI_PHASE): a fixed colour per category, as its own step plots it.
      const byValue = new Map(col.categories.map(([v, meaning, colour]) => [v, { label: `${v} ${meaning}`, colour }]));
      const rgba = new Uint8Array(idx.length * 4);
      const seen = new Set();
      idx.forEach((i, k) => {
        const cat = byValue.get(col.values[i]);
        if (cat) seen.add(col.values[i]);
        const h = cat ? cat.colour : '#d0d4d8';
        rgba.set([1, 3, 5].map((j) => parseInt(h.slice(j, j + 2), 16)).concat(255), k * 4);
      });
      add(idx, { label: '_fill' }, rgba);
      cbar = { label: p.colour, categories: [...byValue].filter(([v]) => seen.has(v)).map(([, c]) => [c.label, c.colour]),
        missing: idx.some((i) => !byValue.has(col.values[i])) };
    } else if (col) {
      const range = ManualQC.colourRange(col);
      const rgb = col.stops.map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)));
      const rgba = new Uint8Array(idx.length * 4);
      idx.forEach((i, k) => {
        const v = col.values[i];
        const c = !isFinite(v) || !range ? [208, 212, 216]
          : rgb[Math.round(Math.max(0, Math.min(1, (v - range[0]) / ((range[1] - range[0]) || 1))) * (rgb.length - 1))];
        rgba.set([c[0], c[1], c[2], 255], k * 4);
      });
      add(idx, { label: '_fill' }, rgba);
      cbar = { label: p.colour, lo: range ? range[0] : null, hi: range ? range[1] : null, stops: col.stops,
        missing: idx.some((i) => !isFinite(col.values[i])) };
    }
    // Axis limits from this profile alone; depth increases downwards.
    const lim = (arr) => {
      let lo = Infinity, hi = -Infinity;
      for (const v of pick(idx, arr)) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      if (!isFinite(lo)) return [0, 1];
      const pad = (hi - lo || 1) * 0.05;
      return [lo - pad, hi + pad];
    };
    const [y0, y1] = lim(ys);
    const spec = {
      suptitle: '', t0, points: idx.length,
      panels: [{
        title: `${label} ${id} · ${idx.length} samples`, xlabel: xdate ? 'Time' : xv, ylabel: yv, xdate, ydate: false,
        xlim: lim(xs), ylim: ['PRES', 'DEPTH'].includes(yv) ? [y1, y0] : [y0, y1], xscale: 'linear', yscale: 'linear',
        legend: true, legend_title: 'Flag', cell: null, share_x: 0, share_y: 0, traces, reflines: [], top_axis: null, cbar,
      }],
    };
    // A redraw of the same profile (a box added, say) keeps the zoom.
    const old = ManualQC.pchart;
    const key = [p.by, id, xv, yv].join('|');
    const keep = old && old.profileKey === key ? old.panels[0].view : null;
    const hidden = old ? old.panels[0].traces.filter((t) => t.hidden).map((t) => t.spec.label) : [];
    Plot.purge(stage);
    const chart = new Chart(stage, spec, data, null, {
      draw: () => ManualQC.tool.draw,
      onSelect: (rect) => ManualQC.addBox(rect),
      onPoint: (pt) => ManualQC.addBox({ x0: pt.x, y0: pt.y }),
      onRemove: (o) => ManualQC.remove(o.index),
    });
    stage._chart = chart;
    chart.profileKey = key;
    for (const t of chart.panels[0].traces) t.hidden = hidden.includes(t.spec.label);
    if (keep) { chart.panels[0].view = keep; chart.draw(); }
    ManualQC.pchart = chart;
    ManualQC.syncOverlays();
    ManualQC.renderList();
  },

  onKey(e) {
    if (ManualQC.view !== 'profile' || !ManualQC.isActive() || !ManualQC.tab().classList.contains('on')) return;
    if (e.metaKey || e.ctrlKey || e.altKey || e.target.closest('input, select, textarea, .CodeMirror')) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      ManualQC.step(e.key === 'ArrowLeft' ? -1 : 1);
    }
  },

  // ---- config <-> chart coordinates ----
  // The chart works in seconds since the spec's t0 on a date axis and log10 on a
  // log axis; the config holds ISO timestamps / plain numbers.
  toChart(v, isDate, log) {
    if (isDate) { const ms = Date.parse(v); return isNaN(ms) ? NaN : (ms - ManualQC.current().spec.t0) / 1000; }
    const n = Number(v);
    return log ? Math.log10(n) : n;
  },
  fromChart(v, isDate, log) {
    if (isDate) return new Date(ManualQC.current().spec.t0 + v * 1000).toISOString().replace(/\.000Z$/, 'Z');
    return +(log ? Math.pow(10, v) : v).toPrecision(7);
  },
  panelAxes() {
    const p = ManualQC.current().spec.panels[0];
    return { xd: p.xdate, yd: p.ydate, xl: p.xscale === 'log', yl: p.yscale === 'log' };
  },

  // The config's boxes in chart coordinates (unparseable ones skipped).
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

  // The chart of the view being shown.
  current() { return ManualQC.view === 'profile' ? ManualQC.pchart : ManualQC.chart; },

  syncOverlays() {
    const chart = ManualQC.current();
    if (!chart) return;
    chart.setOverlays(ManualQC.chartBoxes().map((c) => ({
      ...c,
      color: ManualQC.border(c.box.flag),
      label: `${c.box.flag}${(c.box.mode || 'inside') === 'outside' ? ' outside' : ''}${c.box.override === false ? ' merge' : ''}`,
      hi: c.index === ManualQC.hi || c.index === ManualQC.sel,
    })));
  },

  // Grey the profile panel's points that fall outside the main panel's zoom.
  // The profile holds every step-th sample of the main fill, so index j there
  // is sample j*step here.
  dimProfile() {
    const chart = ManualQC.chart;
    if (!chart || chart.panels.length < 2) return;
    const main = chart.panels[0], side = chart.panels[1];
    const fill = main.traces.find((t) => t.spec.label === '_fill');
    const prof = side.traces.find((t) => /^profile:\d+$/.test(t.spec.gid || ''));
    if (!fill || !prof) return;
    const step = +prof.spec.gid.split(':')[1];
    const v = main.view, h = main.home;
    const zoomed = v.x0 !== h.x0 || v.x1 !== h.x1 || v.y0 !== h.y0 || v.y1 !== h.y1;
    if (!zoomed) { chart.dim(prof, null); return; }
    const x0 = Math.min(v.x0, v.x1), x1 = Math.max(v.x0, v.x1), y0 = Math.min(v.y0, v.y1), y1 = Math.max(v.y0, v.y1);
    const keep = new Uint8Array(prof.n);
    for (let j = 0; j < prof.n; j++) {
      const i = j * step;
      if (i >= fill.n) break;
      const x = fill.x[i], y = fill.y[i];
      keep[j] = x >= x0 && x <= x1 && y >= y0 && y <= y1 ? 1 : 0;
    }
    chart.dim(prof, keep);
  },

  // Live preview: recolour the points as the test would flag them, without a
  // re-run. Mirrors manual_qc.return_qc for the plotted (y) variable.
  preview() {
    const chart = ManualQC.chart;
    if (!chart || !ManualQC.spec || ManualQC.view !== 'time') return;
    const values = ManualQC.values();
    const yv = ManualQC.yVar();
    // Boxes limited to a profile show here but only preview in the profile view.
    const boxes = ManualQC.chartBoxes().filter((c) => ManualQC.timeBox(c.box) && (!c.box.variables || c.box.variables.includes(yv)));
    const rem = !values || values.flag_remaining_good !== false;
    const p = chart.panels[0];
    const sx = (p.home.x1 - p.home.x0) || 1, sy = (p.home.y1 - p.home.y0) || 1;
    // A point box hits the single nearest sample across every trace.
    for (const c of boxes) {
      if (!c.point) continue;
      let best = Infinity; c.hit = null;
      for (const t of p.traces) {
        if (!t.spec.mode.includes('markers')) continue;
        for (let i = 0; i < t.n; i++) {
          const dx = (t.x[i] - c.x0) / sx, dy = (t.y[i] - c.y0) / sy, d = dx * dx + dy * dy;
          if (d < best) { best = d; c.hit = { t, i }; }
        }
      }
    }
    const rgb = {};
    for (const f of Object.keys(ManualQC.COLOURS)) {
      const h = ManualQC.COLOURS[f];
      rgb[f] = [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
    }
    chart.recolour((t, arr) => {
      // Pre-test flag from the trace gid ("flag:N", or "ring:N" when the fill is a
      // colour variable and only non-good flags show, as a ring), else the legend label.
      const m = /^(flag|ring):(\d)$/.exec(t.spec.gid || '');
      const ring = !!m && m[1] === 'ring';
      const base = m ? +m[2] : parseInt(t.spec.label);
      const h = t.spec.color || '#9aa5ad';
      const fallback = isNaN(base) ? [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)] : rgb[base];
      for (let i = 0; i < t.n; i++) {
        let f = base;
        if (!isNaN(f) && f !== 9) {
          const x = t.x[i], y = t.y[i];
          for (const c of boxes) {
            const inside = c.point ? (c.hit && c.hit.t === t && c.hit.i === i)
              : (x >= c.x0 && x <= c.x1 && y >= c.y0 && y <= c.y1);
            if ((c.box.mode || 'inside') === 'outside' ? inside : !inside) continue;
            f = c.box.override === false ? ManualQC.COMBINATRIX[f][c.box.flag] : c.box.flag;
          }
          if (rem && f === 0) f = 1;
        }
        const col = isNaN(f) ? fallback : rgb[f];
        const a = ring && f === 1 ? 0 : 255;
        // Premultiplied alpha: an invisible ring is all zeros.
        arr[i * 4] = a ? col[0] : 0; arr[i * 4 + 1] = a ? col[1] : 0; arr[i * 4 + 2] = a ? col[2] : 0; arr[i * 4 + 3] = a;
      }
    });
  },

  // ---- editing ----
  // rect {x0, x1, y0, y1} is a box; {x0, y0} alone is a point (nearest sample).
  addBox(rect) {
    const values = ManualQC.values();
    if (!values) return;
    const chosen = [...ManualQC.targets()];
    if (!chosen.length) { alert('Tick at least one variable under "Apply flag to".'); ManualQC.current().clearPending(); return; }
    const { xd, yd, xl, yl } = ManualQC.panelAxes();
    const point = rect.x1 === undefined;
    const box = {
      x: point ? [ManualQC.fromChart(rect.x0, xd, xl)]
        : [ManualQC.fromChart(rect.x0, xd, xl), ManualQC.fromChart(rect.x1, xd, xl)],
      y: point ? [ManualQC.fromChart(rect.y0, yd, yl)]
        : [ManualQC.fromChart(rect.y0, yd, yl), ManualQC.fromChart(rect.y1, yd, yl)],
      flag: ManualQC.tool.flag,
      mode: ManualQC.tool.mode,
      y_variable: ManualQC.yVar(),
    };
    if (ManualQC.view === 'profile') {
      const p = ManualQC.prof;
      box[p.by === 'CYCLE' ? 'cycles' : 'profiles'] = [ManualQC.profileId()];
    }
    if (!ManualQC.tool.override) box.override = false;
    if (!(chosen.length === 1 && chosen[0] === box.y_variable)) box.variables = chosen;
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

  // Push the edited values into the builder + YAML and preview them on the
  // chart. Nothing runs until Apply: the preview colours points the way the
  // test will, so several boxes can be drawn in one go.
  commit() {
    STATE.onChange();
    renderPipeline();
    if (ManualQC.chart) { ManualQC.chart.clearPending(); ManualQC.syncOverlays(); ManualQC.preview(); }
    if (ManualQC.view === 'profile') ManualQC.drawProfile();
    ManualQC.renderList();
    ManualQC.dirty = true;
    ManualQC.buttons();
  },

  // Re-run the test with the boxes as they stand. Re-run reads the YAML pane,
  // which commit() refreshed synchronously.
  apply() {
    ManualQC.dirty = false;
    ManualQC.buttons();
    if (ManualQC.chart) ManualQC.chart.busy = true;
    Run.rerunStep();
  },

  // Continue must carry the drawn boxes, so it re-runs first when needed and
  // Run.showPause continues once that pause lands.
  applyAndContinue() {
    if (!ManualQC.dirty) { Run.continueRun(); return; }
    ManualQC.continueAfter = true;
    ManualQC.apply();
  },

  // Re-run in flight: grey and lock the editor until the new plot lands.
  setBusy(on) {
    ManualQC.host().classList.toggle('busy', on);
    ManualQC.rail().classList.toggle('busy', on);
  },

  // The top bar's Re-run/Continue: unapplied boxes make Apply the main action.
  buttons() {
    const rerun = document.getElementById('btn-rerun');
    const cont = document.getElementById('btn-run');
    rerun.lastChild.textContent = ManualQC.dirty ? 'Apply' : 'Re-run';
    rerun.classList.toggle('primary', ManualQC.dirty);
    cont.classList.toggle('primary', !ManualQC.dirty);
    cont.title = ManualQC.dirty ? 'Applies your boxes first' : '';
  },

  // Compact box list: one line each, hover/click highlights it on the chart;
  // the clicked one opens its editor (flag, mode, override).
  renderList() {
    const list = document.getElementById('manual-list');
    if (!list) return;
    list.innerHTML = '';
    const values = ManualQC.values();
    const boxes = values ? values.boxes : [];
    const shown = ManualQC.shownBoxes();
    if (!shown.length) {
      const hint = document.createElement('div');
      hint.className = 'hint';
      hint.textContent = 'No boxes on this plot yet — draw one.';
      list.appendChild(hint);
      return;
    }
    const short = (v) => {
      const s = String(v);
      const m = s.match(/^\d{4}-(\d\d-\d\d)T(\d\d:\d\d)/);
      return m ? `${m[1]} ${m[2]}` : (isFinite(+s) ? String(+(+s).toPrecision(4)) : s);
    };
    shown.forEach(({ b, i }) => {
      const row = document.createElement('div');
      row.className = 'manual-row' + (i === ManualQC.hi ? ' hi' : '') + (i === ManualQC.sel ? ' sel' : '');
      row.style.setProperty('--fc', ManualQC.COLOURS[b.flag] || '#333');
      // Drag to reorder: later boxes win on overlap, so order is part of the config.
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
        (b.variables ? ` · ${b.variables.join(', ')}` : '') +
        (b.profiles ? ` · profile ${b.profiles.join(', ')}` : b.cycles ? ` · cycle ${b.cycles.join(', ')}` : '');
      head.appendChild(txt);
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
