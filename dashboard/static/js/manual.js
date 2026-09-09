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
  // Argo merge table, as ApplyQC.organise_flags: COMBINATRIX[existing][new].
  COMBINATRIX: [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 2, 3, 4, 5, 1, 1, 8, 9], [2, 2, 2, 3, 4, 5, 2, 2, 8, 9],
    [3, 3, 3, 3, 4, 3, 3, 3, 3, 9], [4, 4, 4, 4, 4, 4, 4, 4, 4, 9], [5, 5, 5, 3, 4, 5, 5, 5, 8, 9],
    [6, 1, 2, 3, 4, 5, 6, 6, 8, 9], [7, 1, 2, 3, 4, 5, 6, 7, 8, 9], [8, 8, 8, 3, 4, 8, 8, 8, 8, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
  ],
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
  // Boxes drawn on the plot being shown, with their index in the config list.
  shownBoxes() {
    const values = ManualQC.values();
    const yv = ManualQC.yVar();
    return (values ? values.boxes : []).map((b, i) => ({ b, i })).filter(({ b }) => b && ManualQC.boxY(b) === yv);
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
    ManualQC.hi = null; ManualQC.sel = null;
    ManualQC.dirty = false; ManualQC.continueAfter = false;
    ManualQC.tab().classList.remove('hidden');
    ManualQC.host().innerHTML = ''; ManualQC.rail().innerHTML = '';
    Run.showTab('manual');
  },

  close() {
    ManualQC.chart = null; ManualQC.spec = null; ManualQC.fig = null;
    ManualQC.dirty = false; ManualQC.continueAfter = false;
    const tab = ManualQC.tab();
    tab.classList.add('hidden');
    if (tab.classList.contains('active')) Run.showTab('run');
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
    ManualQC.rail().appendChild(ManualQC.sidebar());
    ManualQC.renderList();
    host.appendChild(ManualQC.plotBar());
    const stage = document.createElement('div');
    stage.className = 'manual-stage';
    stage.innerHTML = '<div class="viewer-loading">Loading plot…</div>';
    host.appendChild(stage);

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
    const seg = document.createElement('div');
    seg.className = 'manual-seg';
    const hint = document.createElement('div');
    hint.className = 'hint';
    const mod = navigator.platform.toLowerCase().includes('mac') ? '⌘' : 'Ctrl';
    const setHint = () => {
      hint.textContent = t.draw
        ? `Drag to draw a box · click a point to flag just it · ${mod}-drag to zoom · double-click resets zoom`
        : `Drag to zoom · ${mod}-drag to draw a box · ${mod}-click a point to flag just it · double-click resets zoom`;
    };
    const segBtns = [];
    for (const [draw, text] of [[true, 'Draw boxes'], [false, 'Zoom']]) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'manual-seg-btn' + (t.draw === draw ? ' on' : ''); b.textContent = text;
      b.onclick = () => { t.draw = draw; segBtns.forEach((x) => x.classList.toggle('on', x === b)); setHint(); };
      segBtns.push(b); seg.appendChild(b);
    }
    setHint();
    mouse.appendChild(seg); mouse.appendChild(hint);
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
    btn.type = 'button'; btn.className = 'manual-pick-btn';
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
    const values = ManualQC.values();
    const cur = ManualQC.yVar();
    for (const y of ManualQC.plots()) {
      const n = (values ? values.boxes : []).filter((b) => b && ManualQC.boxY(b) === y).length;
      const t = document.createElement('button');
      t.type = 'button'; t.className = 'manual-plot' + (y === cur ? ' on' : '');
      t.innerHTML = `<span>${y}</span>` + (n ? `<i>${n}</i>` : '');
      t.title = `${y} vs ${ManualQC.xVar()}` + (n ? ` · ${n} box${n === 1 ? '' : 'es'}` : '');
      t.onclick = () => ManualQC.setPlot(y);
      bar.appendChild(t);
    }
    const add = document.createElement('select');
    add.className = 'manual-plot-add';
    const ph = document.createElement('option');
    ph.value = ''; ph.textContent = '+ plot'; ph.selected = true;
    add.appendChild(ph);
    for (const v of ManualQC.variables()) {
      if (v === 'TIME' || ManualQC.plots().includes(v)) continue;
      const o = document.createElement('option');
      o.value = v; o.textContent = v;
      add.appendChild(o);
    }
    add.onchange = () => { if (add.value) ManualQC.setPlot(add.value); };
    bar.appendChild(add);
    const x = document.createElement('span');
    x.className = 'hint manual-plot-x'; x.textContent = 'vs ' + ManualQC.xVar();
    bar.appendChild(x);
    // Colour by a third variable: flags then show as rings around non-good points.
    const colour = document.createElement('select');
    colour.className = 'manual-plot-add manual-plot-colour';
    colour.title = 'Colour points by a variable (flags become rings)';
    const none = document.createElement('option');
    none.value = ''; none.textContent = 'colour: flag';
    colour.appendChild(none);
    const curC = (values && values.colour_variable) || '';
    for (const v of ManualQC.variables()) {
      if (v === 'TIME') continue;
      const o = document.createElement('option');
      o.value = v; o.textContent = 'colour: ' + v; if (v === curC) o.selected = true;
      colour.appendChild(o);
    }
    colour.onchange = () => {
      const vals = ManualQC.values();
      if (!vals) return;
      if (colour.value) vals.colour_variable = colour.value; else delete vals.colour_variable;
      ManualQC.commit();
      ManualQC.apply();
    };
    bar.appendChild(colour);
    // Profile side panel (colour variable vs y), greyed outside the main zoom.
    const prof = document.createElement('button');
    prof.type = 'button';
    const on = !!(values && values.profile_plot && curC);
    prof.className = 'manual-plot manual-plot-profile' + (on ? ' on' : '');
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

  // ---- config <-> chart coordinates ----
  // The chart works in seconds since the spec's t0 on a date axis and log10 on a
  // log axis; the config holds ISO timestamps / plain numbers.
  toChart(v, isDate, log) {
    if (isDate) { const ms = Date.parse(v); return isNaN(ms) ? NaN : (ms - ManualQC.spec.t0) / 1000; }
    const n = Number(v);
    return log ? Math.log10(n) : n;
  },
  fromChart(v, isDate, log) {
    if (isDate) return new Date(ManualQC.spec.t0 + v * 1000).toISOString().replace(/\.000Z$/, 'Z');
    return +(log ? Math.pow(10, v) : v).toPrecision(7);
  },
  panelAxes() {
    const p = ManualQC.spec.panels[0];
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

  syncOverlays() {
    if (!ManualQC.chart || !ManualQC.spec) return;
    ManualQC.chart.setOverlays(ManualQC.chartBoxes().map((c) => ({
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
    if (!chart || !ManualQC.spec) return;
    const values = ManualQC.values();
    const yv = ManualQC.yVar();
    const boxes = ManualQC.chartBoxes().filter((c) => !c.box.variables || c.box.variables.includes(yv));
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
      const fallback = rgb[isNaN(base) ? 0 : base];
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
    if (!chosen.length) { alert('Tick at least one variable under "Apply flag to".'); ManualQC.chart.clearPending(); return; }
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
    if (!ManualQC.tool.override) box.override = false;
    if (!(chosen.length === 1 && chosen[0] === ManualQC.yVar())) box.variables = chosen;
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

  buttons() {
    const rerun = document.getElementById('btn-manual-rerun');
    const cont = document.getElementById('btn-manual-continue');
    if (!rerun || !cont) return;
    rerun.lastChild.textContent = ManualQC.dirty ? 'Apply & re-run' : 'Re-run test';
    rerun.classList.toggle('primary', ManualQC.dirty);
    rerun.classList.toggle('ghost', !ManualQC.dirty);
    cont.lastChild.textContent = ManualQC.dirty ? 'Apply & continue' : 'Continue';
    cont.classList.toggle('primary', !ManualQC.dirty);
    cont.classList.toggle('ghost', ManualQC.dirty);
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
        (b.variables ? ` · ${b.variables.join(', ')}` : '');
      head.appendChild(txt);
      const rm = document.createElement('button');
      rm.className = 'manual-row-rm'; rm.textContent = '×'; rm.title = 'remove this box';
      rm.onclick = (e) => { e.stopPropagation(); ManualQC.remove(i); };
      head.appendChild(rm);
      row.appendChild(head);

      if (i === ManualQC.sel) {
        const edit = document.createElement('div');
        edit.className = 'manual-row-edit';
        const flag = document.createElement('select');
        for (const [f, meaning] of ManualQC.FLAGS) {
          const o = document.createElement('option');
          o.value = f; o.textContent = `${f} ${meaning}`; if (f === b.flag) o.selected = true;
          flag.appendChild(o);
        }
        flag.onchange = () => { b.flag = Number(flag.value); ManualQC.commit(); };
        edit.appendChild(flag);
        const mode = document.createElement('select');
        for (const m of ['inside', 'outside']) {
          const o = document.createElement('option');
          o.value = m; o.textContent = m; if ((b.mode || 'inside') === m) o.selected = true;
          mode.appendChild(o);
        }
        mode.onchange = () => { b.mode = mode.value; ManualQC.commit(); };
        edit.appendChild(mode);
        const ov = document.createElement('label');
        ov.className = 'manual-row-ov'; ov.title = 'override existing flags';
        const ovc = document.createElement('input');
        ovc.type = 'checkbox'; ovc.checked = b.override !== false;
        ovc.onchange = () => { if (ovc.checked) delete b.override; else b.override = false; ManualQC.commit(); };
        ov.appendChild(ovc); ov.appendChild(document.createTextNode('override'));
        edit.appendChild(ov);
        const detail = document.createElement('div');
        detail.className = 'manual-row-detail';
        detail.textContent = `${ManualQC.xVar()} ${b.x.map(short).join(' → ')}\n${ManualQC.yVar()} ${b.y.map(short).join(' → ')}`;
        edit.appendChild(detail);
        row.appendChild(edit);
      }
      list.appendChild(row);
    });
  },
};
