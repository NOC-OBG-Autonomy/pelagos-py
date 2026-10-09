// Review of the paused step (or QC test) at the top of the Plots tab: its figures and earlier attempts.

const Review = {
  active: false,
  index: null,     // step index in the running pipeline
  name: null,      // step name, used to match figures and guard the re-run
  test: null,      // QC test, when the runner split this step test by test
  key: null,       // Run.unitKey(index, test), what figures are grouped under
  busy: false,     // true between "Re-run" and the next pause
  selected: null,  // attempt shown in the main view; null = the latest one

  host() { return document.getElementById('step-review'); },

  show(index, name, test) {
    Review.active = true;
    Review.index = index;
    Review.name = name;
    Review.test = test || null;
    Review.key = Run.unitKey(index, test);
    Review.busy = false;
    Review.selected = null;
    Review.build();
    Review.host().classList.remove('hidden');
    if (ManualQC.isActive()) ManualQC.open();
    else Run.showTab('plots');
    Review.renderPlots();
    Run.renderGallery();
  },

  hide() {
    Review.active = false;
    Review.busy = false;
    Review.host().innerHTML = '';
    Review.host().classList.add('hidden');
    ManualQC.close();
    Run.renderGallery();
  },

  // If steps were reordered since the run, fall back to the first step of the same name.
  item() {
    const items = STATE.pipeline.items;
    const byIndex = items[Review.index];
    if (byIndex && byIndex.name === Review.name) return byIndex;
    return items.find((i) => i.name === Review.name) || byIndex || null;
  },

  // Only a viewing choice; it doesn't change what Continue does.
  selectedGroup() {
    const attempts = Run.groupsFor(Review.key);
    if (!attempts.length) return null;
    return attempts.includes(Review.selected)
      ? Review.selected : attempts[attempts.length - 1];
  },

  select(group) {
    Review.selected = group;
    Review.renderPlots();
  },

  // A QC unit's parameters are `{qc_settings: {<test>: …}}`.
  unwrap(params) {
    if (!Review.test || !params) return params;
    return (params.qc_settings || {})[Review.test] || null;
  },

  // Same reference the builder card edits.
  testValues() {
    const item = Review.item();
    if (!item || !Review.test) return null;
    const settings = item.values.qc_settings;
    return settings ? settings[Review.test] || null : null;
  },

  // Builder, YAML and form all get the attempt's parameters.
  applyParams(rawParams) {
    const item = Review.item();
    const params = Review.unwrap(rawParams);
    if (!item || !params) return;
    if (Review.test) {
      const values = Review.testValues();
      if (!values) return;
      const def = STATE.qcByName[Review.test];
      for (const spec of (def && def.parameters) || []) {
        values[spec.name] = spec.name in params
          ? Forms.clone(params[spec.name]) : Forms.defaultValue(spec);
      }
    } else {
      for (const spec of item.def.parameters || []) {
        item.values[spec.name] = spec.name in params
          ? Forms.clone(params[spec.name]) : Forms.defaultValue(spec);
      }
    }
    STATE.onChange();
    renderPipeline();
  },

  // Built once per pause; plots and title refresh on their own.
  build() {
    const host = Review.host();
    host.innerHTML = '';
    const title = document.createElement('h4');
    title.className = 'plot-step-title review-title'; title.id = 'review-title';
    const hint = document.createElement('div');
    hint.className = 'review-hint hidden'; hint.id = 'review-hint';
    const plots = document.createElement('div');
    plots.id = 'review-plots';
    host.appendChild(title);
    host.appendChild(hint);
    host.appendChild(plots);
    Review.renderTitle();
  },

  renderTitle() {
    const title = document.getElementById('review-title');
    if (!title) return;
    const where = Review.test ? `${Review.name}, step ${Review.index + 1}` : `step ${Review.index + 1}`;
    title.textContent = `${Review.test ? testLabel(Review.test) : Review.name} · ${Run.pauseFailed ? 'failed' : 'paused'} (${where})`;
    title.classList.toggle('failed', Run.pauseFailed);
  },

  renderHint(text) {
    const hint = document.getElementById('review-hint');
    if (!hint) return;
    hint.textContent = text;
    hint.classList.toggle('hidden', !text);
  },

  syncUseButtons() {
    const form = Run.paramsAt(Review.index, Review.test);
    for (const btn of document.querySelectorAll('#review-plots .review-use')) {
      btn.classList.toggle('hidden', Forms.equal(btn.attemptParams, form));
    }
  },

  renderPlots() {
    const host = document.getElementById('review-plots');
    if (!host) return;
    host.innerHTML = '';
    const attempts = Run.groupsFor(Review.key);
    const current = Review.selectedGroup();
    // No open group means the last re-run drew nothing, so the figure shown is stale.
    const isLatest = current && current === attempts[attempts.length - 1];
    const stale = !Review.busy && isLatest && attempts.length > 0 &&
      (!Run.activeGroup || Run.activeGroup.key !== Review.key);

    if (!current || !current.figs.length) {
      const hint = document.createElement('div');
      hint.className = 'hint review-empty';
      hint.textContent = Review.busy
        ? 'Re-running — the new figure will appear here.'
        : 'This step produced no figure. Re-run it with diagnostics on, or Continue.';
      host.appendChild(hint);
    } else {
      const no = Run.attemptNo(current);
      const main = document.createElement('div');
      main.className = 'review-main' + (current.figs.some((f) => f.isError) ? ' failed' : '');
      const label = document.createElement('div');
      label.className = 'review-attempt-label';
      label.textContent = stale
        ? `Attempt ${no} — the last re-run produced no new figure`
        : (attempts.length > 1
          ? `Attempt ${no}${isLatest ? ' (latest — this is what Continue carries forward)' : ''}`
          : 'Result');
      main.appendChild(label);
      main.appendChild(Review.paramSummary(current.params));
      const cards = document.createElement('div');
      cards.className = 'review-main-cards';
      current.figs.forEach((_, i) =>
        cards.appendChild(Viewer.card(current.figs, i, { cls: 'big' })));
      main.appendChild(cards);
      host.appendChild(main);
    }
    // Manual QC edits the latest attempt in its own tab.
    if (ManualQC.isActive()) {
      const latest = attempts[attempts.length - 1];
      ManualQC.render(latest && latest.figs.length && latest.figs[0].spec ? latest.figs[0] : null);
    }

    if (attempts.length > 1) {
      const strip = document.createElement('div');
      strip.className = 'review-strip';
      const title = document.createElement('div');
      title.className = 'review-strip-title';
      title.textContent = 'Attempts — click to compare';
      strip.appendChild(title);
      const row = document.createElement('div');
      row.className = 'review-strip-row';
      for (let k = attempts.length - 1; k >= 0; k--) {
        const g = attempts[k];
        const failed = g.figs.some((f) => f.isError);
        const cell = document.createElement('div');
        cell.className = 'review-thumb' + (failed ? ' failed' : '') + (g === current ? ' selected' : '');
        if (g.figs.length && g.figs[0].isLog) {
          const pre = document.createElement('pre');
          pre.className = 'log-card-text log-card-text-compact' +
            (g.figs[0].isError ? ' error-card-text' : '');
          pre.textContent = g.figs[0].text;
          cell.appendChild(pre);
        } else if (g.figs.length) {
          const img = document.createElement('img');
          img.src = Viewer.src(g.figs[0]);
          img.alt = `attempt ${Run.attemptNo(g)}`;
          img.loading = 'lazy';
          cell.appendChild(img);
        }
        const cap = document.createElement('div');
        cap.className = 'review-thumb-cap';
        cap.textContent = `Attempt ${Run.attemptNo(g)}` + (g === current ? ' ·  shown' : '');
        if (failed) {
          const mark = document.createElement('span');
          mark.className = 'review-thumb-failed';
          mark.textContent = ' · failed';
          cap.appendChild(mark);
        }
        cell.appendChild(cap);
        cell.appendChild(Review.paramSummary(g.params, { compact: true }));
        // Only fills the form in; running is still an explicit Re-run.
        if (g.params) {
          const use = document.createElement('button');
          use.className = 'sm review-use';
          use.textContent = 'Use these';
          use.attemptParams = g.params;
          use.title = 'Load these parameters into the form (does not re-run)';
          use.onclick = (e) => {
            e.stopPropagation();
            Review.applyParams(g.params);
            Review.select(g);
          };
          cell.appendChild(use);
        }
        cell.onclick = () => Review.select(g);
        cell.title = 'Show this attempt';
        row.appendChild(cell);
      }
      strip.appendChild(row);
      host.appendChild(strip);
      Review.syncUseButtons();
    }
  },

  // Only values that differ from the schema defaults.
  paramSummary(rawParams, { compact = false } = {}) {
    const params = Review.unwrap(rawParams);
    const wrap = document.createElement('div');
    wrap.className = 'review-diff' + (compact ? '' : ' review-diff-main');
    if (!params) {
      wrap.classList.add('hint');
      wrap.textContent = 'parameters not recorded';
      return wrap;
    }
    const keys = Object.keys(params);
    if (!keys.length) {
      wrap.classList.add('hint');
      wrap.textContent = 'step defaults';
      return wrap;
    }
    for (const k of keys.slice(0, compact ? 3 : 8)) {
      const chip = document.createElement('span');
      chip.className = 'review-chip';
      chip.textContent = `${k}: ${Review.short(params[k])}`;
      wrap.appendChild(chip);
    }
    if (keys.length > (compact ? 3 : 8)) {
      const more = document.createElement('span');
      more.className = 'review-chip';
      more.textContent = `+${keys.length - (compact ? 3 : 8)} more`;
      wrap.appendChild(more);
    }
    return wrap;
  },

  short(v) {
    if (v === undefined) return '—';
    const s = typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v);
    return s.length > 18 ? s.slice(0, 17) + '…' : s;
  },

  setBusy(busy) {
    Review.busy = busy;
    ManualQC.setBusy(busy);
  },
};
