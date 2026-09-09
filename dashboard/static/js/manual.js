// Manual QC: the paused-step panel for the `manual qc` test.
//
// The test's plot (y_variable vs x_variable, coloured by flag) is embedded here
// as a live chart. ⌘/Ctrl-drag selects a region; a popover then asks which flag
// to give it (inside or outside the box) and on which variables. Each box is
// written straight into the test's `boxes` parameter — the same values object
// the builder card and YAML show — and the test is re-run so the plot refreshes
// with the flags applied. Boxes on the chart carry an × that removes them the
// same way. The axis pickers change x_variable/y_variable and re-run too. So
// the config is the only state: what you drew is what a later run repeats.

const ManualQC = {
  TEST: 'manual qc',
  FLAGS: [
    [0, 'no QC performed'], [1, 'good'], [2, 'probably good'], [3, 'probably bad (correctable)'],
    [4, 'bad'], [5, 'value changed'], [6, 'not used'], [7, 'not used'],
    [8, 'estimated / interpolated'], [9, 'missing value'],
  ],
  // Same palette as pelagos_py.utils.fig_spec.FLAG_COLOURS.
  COLOURS: { 0: '#9aa5ad', 1: '#1f6fd6', 2: '#7fb2e5', 3: '#e8912b', 4: '#d6392f',
    5: '#9aa5ad', 6: '#9aa5ad', 7: '#9aa5ad', 8: '#17b6c4', 9: '#111111' },

  chart: null,
  spec: null,

  isActive() { return Review.active && Review.test === ManualQC.TEST; },

  values() {
    const v = Review.testValues();
    if (v && !Array.isArray(v.boxes)) v.boxes = [];
    return v;
  },

  xVar() { const v = ManualQC.values(); return (v && v.x_variable) || 'TIME'; },
  yVar() { const v = ManualQC.values(); return (v && v.y_variable) || 'PRES'; },

  // Variables offered by the axis/target pickers: what the runner reported at
  // this pause, plus whatever the config already names.
  variables() {
    const out = [...(Run.variables || [])];
    for (const v of [ManualQC.xVar(), ManualQC.yVar()]) if (!out.includes(v)) out.push(v);
    return out;
  },

  // ---- panel ----
  panel(fig) {
    const wrap = document.createElement('div');
    wrap.className = 'manual-panel';

    const bar = document.createElement('div');
    bar.className = 'manual-bar';
    // x stays TIME: boxes hold ISO timestamps, which a numeric axis cannot read.
    const xl = document.createElement('span');
    xl.className = 'manual-axis'; xl.textContent = 'x ' + ManualQC.xVar();
    bar.appendChild(xl);
    bar.appendChild(ManualQC.axisPicker('y', ManualQC.yVar()));
    const hint = document.createElement('span');
    hint.className = 'hint manual-hint';
    const mod = navigator.platform.toLowerCase().includes('mac') ? '⌘' : 'Ctrl';
    hint.textContent = `${mod} + drag to select a region and flag it · drag to zoom · double-click to reset · × on a box removes it`;
    bar.appendChild(hint);
    wrap.appendChild(bar);

    const host = document.createElement('div');
    host.className = 'manual-chart';
    host.innerHTML = '<div class="viewer-loading">Loading plot…</div>';
    wrap.appendChild(host);

    const list = document.createElement('div');
    list.className = 'manual-boxes';
    wrap.appendChild(list);
    ManualQC.renderList(list);

    ManualQC.chart = null;
    Plot.fetchSpec(fig.spec)
      .then((spec) => Plot.render(host, spec, {
        name: fig.spec,
        manual: {
          onSelect: (rect, at) => ManualQC.popover(host, rect, at),
          onRemove: (o) => ManualQC.remove(o.index),
        },
      }).then((chart) => { ManualQC.spec = spec; ManualQC.chart = chart; ManualQC.syncOverlays(); }))
      .catch((err) => {
        Plot.purge(host);
        host.innerHTML = '';
        host.appendChild(Viewer.card([fig], 0, { cls: 'big' }));
        const note = document.createElement('div');
        note.className = 'hint';
        note.textContent = 'Interactive view unavailable' + (err && err.message ? ' — ' + err.message : '') +
          '. Boxes can still be edited in the builder card.';
        wrap.appendChild(note);
      });
    return wrap;
  },

  axisPicker(axis, current) {
    const lbl = document.createElement('label');
    lbl.className = 'manual-axis';
    lbl.textContent = axis + ' ';
    const sel = document.createElement('select');
    for (const v of ManualQC.variables()) {
      const o = document.createElement('option');
      o.value = v; o.textContent = v; if (v === current) o.selected = true;
      sel.appendChild(o);
    }
    sel.onchange = () => {
      const values = ManualQC.values();
      if (!values) return;
      values[axis + '_variable'] = sel.value;
      ManualQC.commit();
    };
    lbl.appendChild(sel);
    return lbl;
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

  syncOverlays() {
    if (!ManualQC.chart || !ManualQC.spec) return;
    const values = ManualQC.values();
    const { xd, yd, xl, yl } = ManualQC.panelAxes();
    const overlays = [];
    (values ? values.boxes : []).forEach((b, i) => {
      if (!b || !Array.isArray(b.x) || !Array.isArray(b.y)) return;
      const x = b.x.map((v) => ManualQC.toChart(v, xd, xl)), y = b.y.map((v) => ManualQC.toChart(v, yd, yl));
      if (![...x, ...y].every(isFinite)) return;
      overlays.push({
        index: i, x0: Math.min(...x), x1: Math.max(...x), y0: Math.min(...y), y1: Math.max(...y),
        color: ManualQC.COLOURS[b.flag] || '#333',
        label: `${b.flag}${(b.mode || 'inside') === 'outside' ? ' outside' : ''}`,
      });
    });
    ManualQC.chart.setOverlays(overlays);
  },

  // ---- the flag popover ----
  popover(host, rect, at) {
    ManualQC.closePopover();
    const values = ManualQC.values();
    if (!values) return;
    const pop = document.createElement('div');
    pop.className = 'manual-pop';
    pop.onpointerdown = (e) => e.stopPropagation();
    const state = { flag: 4, mode: 'inside', vars: new Set([ManualQC.yVar()]) };

    const title = document.createElement('div');
    title.className = 'manual-pop-title';
    title.textContent = 'Flag the selected region';
    pop.appendChild(title);

    const grid = document.createElement('div');
    grid.className = 'manual-flags';
    const btns = [];
    for (const [f, meaning] of ManualQC.FLAGS) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'manual-flag' + (f === state.flag ? ' on' : '');
      b.innerHTML = `<span class="sw" style="background:${ManualQC.COLOURS[f]}"></span><b>${f}</b> ${meaning}`;
      b.onclick = () => { state.flag = f; btns.forEach((x) => x.classList.toggle('on', x === b)); };
      btns.push(b); grid.appendChild(b);
    }
    pop.appendChild(grid);

    const modes = document.createElement('div');
    modes.className = 'manual-modes';
    for (const [m, text] of [['inside', 'Inside the box'], ['outside', 'Outside the box']]) {
      const l = document.createElement('label');
      const r = document.createElement('input');
      r.type = 'radio'; r.name = 'manual-mode'; r.checked = m === state.mode;
      r.onchange = () => { state.mode = m; };
      l.appendChild(r); l.appendChild(document.createTextNode(' ' + text));
      modes.appendChild(l);
    }
    pop.appendChild(modes);

    const vars = document.createElement('div');
    vars.className = 'manual-vars';
    const vt = document.createElement('div');
    vt.className = 'hint'; vt.textContent = 'Apply flag to';
    vars.appendChild(vt);
    const vlist = document.createElement('div');
    vlist.className = 'manual-var-list';
    for (const v of ManualQC.variables()) {
      if (v === 'TIME') continue;
      const l = document.createElement('label');
      const c = document.createElement('input');
      c.type = 'checkbox'; c.checked = state.vars.has(v);
      c.onchange = () => { if (c.checked) state.vars.add(v); else state.vars.delete(v); };
      l.appendChild(c); l.appendChild(document.createTextNode(' ' + v));
      vlist.appendChild(l);
    }
    vars.appendChild(vlist);
    pop.appendChild(vars);

    const row = document.createElement('div');
    row.className = 'manual-pop-actions';
    const cancel = document.createElement('button');
    cancel.className = 'ghost'; cancel.textContent = 'Cancel';
    cancel.onclick = () => ManualQC.closePopover();
    const ok = document.createElement('button');
    ok.className = 'primary'; ok.textContent = 'Add box & re-run';
    ok.onclick = () => {
      if (!state.vars.size) { alert('Pick at least one variable to flag.'); return; }
      const { xd, yd, xl, yl } = ManualQC.panelAxes();
      const box = {
        x: [ManualQC.fromChart(rect.x0, xd, xl), ManualQC.fromChart(rect.x1, xd, xl)],
        y: [ManualQC.fromChart(rect.y0, yd, yl), ManualQC.fromChart(rect.y1, yd, yl)],
        flag: state.flag,
        mode: state.mode,
      };
      const chosen = [...state.vars];
      if (!(chosen.length === 1 && chosen[0] === ManualQC.yVar())) box.variables = chosen;
      values.boxes.push(box);
      ManualQC.closePopover();
      ManualQC.commit();
    };
    row.appendChild(cancel); row.appendChild(ok);
    pop.appendChild(row);

    host.appendChild(pop);
    const hw = host.clientWidth, hh = host.clientHeight;
    pop.style.left = Math.max(4, Math.min(hw - pop.offsetWidth - 4, at.x + 10)) + 'px';
    pop.style.top = Math.max(4, Math.min(hh - pop.offsetHeight - 4, at.y)) + 'px';
    ManualQC._pop = pop;
  },

  closePopover() {
    if (ManualQC._pop) { ManualQC._pop.remove(); ManualQC._pop = null; }
    if (ManualQC.chart) ManualQC.chart.clearPending();
  },

  remove(index) {
    const values = ManualQC.values();
    if (!values || !values.boxes[index]) return;
    values.boxes.splice(index, 1);
    ManualQC.commit();
  },

  // Push the edited values into the builder + YAML, then re-run the test so the
  // plot shows the boxes applied. Re-run reads the YAML pane, which onChange
  // refreshes synchronously.
  commit() {
    STATE.onChange();
    renderPipeline();
    if (ManualQC.chart) { ManualQC.chart.busy = true; ManualQC.syncOverlays(); }
    Run.rerunStep();
  },

  // Boxes as a list under the chart — a readable copy of what the YAML holds.
  renderList(list) {
    list.innerHTML = '';
    const values = ManualQC.values();
    const boxes = values ? values.boxes : [];
    if (!boxes.length) {
      list.className = 'manual-boxes hint';
      list.textContent = 'No boxes yet — this test flags nothing until you draw one.';
      return;
    }
    list.className = 'manual-boxes';
    boxes.forEach((b, i) => {
      const chip = document.createElement('span');
      chip.className = 'manual-box-chip';
      const sw = document.createElement('span');
      sw.className = 'sw'; sw.style.background = ManualQC.COLOURS[b.flag] || '#333';
      chip.appendChild(sw);
      const fmt = (v) => Array.isArray(v) ? v.map((x) => String(x).replace('T', ' ').replace(/Z$/, '')).join(' → ') : String(v);
      chip.appendChild(document.createTextNode(
        `flag ${b.flag} ${b.mode || 'inside'} · ${ManualQC.xVar()} ${fmt(b.x)} · ${ManualQC.yVar()} ${fmt(b.y)}` +
        (b.variables ? ` · on ${b.variables.join(', ')}` : '')));
      const rm = document.createElement('button');
      rm.className = 'icon-btn'; rm.innerHTML = Icon.svg('close'); rm.title = 'remove this box and re-run';
      rm.onclick = () => ManualQC.remove(i);
      chip.appendChild(rm);
      list.appendChild(chip);
    });
  },
};
