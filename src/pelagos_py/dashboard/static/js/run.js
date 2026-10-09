// Run the current config and stream the pipeline's logs via Server-Sent Events.

const Run = {
  source: null,
  progressEl: null,
  stopping: false,  // so the end event reads as "stopped"
  runBtnMode: 'idle', // see setRunButton
  // {index, name, test, key}; figures group by index so a repeated step name still gets its own group.
  currentStep: null,
  plotCount: 0,

  // One group per (unit, re-run): {index, key, step, test, figs: [{fname, caption, spec}], params}
  groups: [],
  activeGroup: null,
  pendingParams: null,

  pausedStep: null,
  pausedName: null, // guards a re-run against edits
  pausedTest: null,
  reportName: null,
  pendingStart: false,

  // Marker prefixes run_bootstrap.py prints on stdout; formats in its module docstring.
  FIG_MARKER: '__PELAGOS_FIG__ ',
  LOG_MARKER: '__PELAGOS_LOG__ ',
  FAIL_MARKER: '__PELAGOS_FAIL__ ',
  STEP_MARKER: '__PELAGOS_STEP__ ',
  PAUSE_MARKER: '__PELAGOS_PAUSE__ ',
  RERUN_MARKER: '__PELAGOS_RERUN__ ',
  MEM_MARKER: '__PELAGOS_MEM__ ',
  REPORT_MARKER: '__PELAGOS_REPORT__ ',
  VARS_MARKER: '__PELAGOS_VARS__ ',
  TIME_MARKER: '__PELAGOS_TIME__ ',
  SAMPLE_MARKER: '__PELAGOS_SAMPLE__ ',
  variables: [], // for Manual QC's axis pickers
  emptyVariables: [], // those all NaN or 0

  // Reset on every __PELAGOS_STEP__, set by __PELAGOS_FAIL__.
  pauseFailed: false,

  // The paused unit's attempts and the result the runner holds, so Continue can follow the form.
  outcomes: [],
  heldParams: null,
  sentParams: null,
  continueAfter: false,  // carry on once the re-run passes

  // The server forwards only SGR colour codes; it strips other escapes.
  ANSI_SGR: /\x1b\[([0-9;]*)m/g,
  ANSI_COLORS: {
    30: 'ansi-black', 31: 'ansi-red', 32: 'ansi-green', 33: 'ansi-yellow',
    34: 'ansi-blue', 35: 'ansi-magenta', 36: 'ansi-cyan', 37: 'ansi-white',
    90: 'ansi-bright-black', 91: 'ansi-bright-red', 92: 'ansi-bright-green',
    93: 'ansi-bright-yellow', 94: 'ansi-bright-blue', 95: 'ansi-bright-magenta',
    96: 'ansi-bright-cyan', 97: 'ansi-bright-white',
  },

  stripAnsi(text) {
    return text.replace(Run.ANSI_SGR, '');
  },

  // Text is escaped first, so log output can't inject markup.
  ansiToHtml(text) {
    let out = '', fg = null, bold = false, dim = false, open = false, last = 0;
    const close = () => { if (open) { out += '</span>'; open = false; } };
    const openSpan = () => {
      const cls = [fg, bold ? 'ansi-bold' : null, dim ? 'ansi-dim' : null].filter(Boolean).join(' ');
      if (cls) { out += `<span class="${cls}">`; open = true; }
    };
    for (const m of text.matchAll(Run.ANSI_SGR)) {
      out += escapeHtml(text.slice(last, m.index));
      last = m.index + m[0].length;
      close();
      const parts = (m[1] || '0').split(';');
      for (let i = 0; i < parts.length; i++) {
        const n = Number(parts[i] || 0);
        if (n === 0) { fg = null; bold = false; dim = false; }
        else if (n === 1) bold = true;
        else if (n === 2) dim = true;
        else if (n === 22) { bold = false; dim = false; }
        else if (n === 39) fg = null;
        // 256-colour: only the SEVERE amber (202) the pipeline emits is mapped.
        else if (n === 38 && parts[i + 1] === '5') {
          if (Number(parts[i + 2]) === 202) fg = 'ansi-amber';
          i += 2;
        }
        else if (Run.ANSI_COLORS[n]) fg = Run.ANSI_COLORS[n];
      }
      openSpan();
    }
    out += escapeHtml(text.slice(last));
    close();
    return out;
  },

  levelClass(line) {
    if (/ - ERROR - | ERROR:| Traceback/.test(line)) return 'lvl-error';
    if (/ - WARNING - | WARN/.test(line)) return 'lvl-warn';
    if (/ - SEVERE - /.test(line)) return 'lvl-severe';
    if (/STOP|Pipeline stopped/.test(line)) return 'lvl-stop';
    return '';
  },

  looksLikeProgress(line) {
    return /\d+%\|/.test(line);
  },

  // Scrolling up detaches from the tail; the jump-to-latest button re-attaches.
  stick: true,

  atBottom(c) {
    // slack for sub-pixel rounding
    return c.scrollHeight - c.scrollTop - c.clientHeight < 8;
  },

  autoScroll() {
    const c = document.getElementById('log-console');
    if (Run.stick) c.scrollTop = c.scrollHeight;
    document.getElementById('log-to-bottom').classList.toggle('hidden', Run.stick);
  },

  scrollToBottom() {
    const c = document.getElementById('log-console');
    Run.stick = true;
    c.scrollTop = c.scrollHeight;
    document.getElementById('log-to-bottom').classList.add('hidden');
  },

  initScroll() {
    const c = document.getElementById('log-console');
    c.addEventListener('scroll', () => {
      Run.stick = Run.atBottom(c);
      document.getElementById('log-to-bottom').classList.toggle('hidden', Run.stick);
    });
    document.getElementById('log-to-bottom')
      .addEventListener('click', () => Run.scrollToBottom());
  },

  append(line) {
    Run.settleStart();
    const c = document.getElementById('log-console');
    const span = document.createElement('span');
    // Fallback colour; real ANSI colour still wins, being set on a descendant span.
    const cls = Run.levelClass(Run.stripAnsi(line));
    if (cls) span.className = cls;
    span.innerHTML = Run.ansiToHtml(line) + '\n';
    c.appendChild(span);
    Run.autoScroll();
  },

  note(tag, text, cls = '') {
    const c = document.getElementById('log-console');
    const span = document.createElement('span');
    span.className = 'lvl-note' + (cls ? ' note-' + cls : '');
    span.innerHTML = `<span class="note-tag">${escapeHtml(tag)}</span>${escapeHtml(text)}`;
    c.appendChild(span);
    Run.autoScroll();
  },

  banner(kind, title, sub = '') {
    const c = document.getElementById('log-console');
    const span = document.createElement('span');
    span.className = 'lvl-banner banner-' + kind;
    span.innerHTML = (kind === 'starting' ? ''
      : `<span class="banner-ico">${Icon.svg(kind === 'ok' ? 'check' : 'alert', 13)}</span>`)
      + `<strong>${escapeHtml(title)}</strong>`
      + (sub ? `<span class="banner-sub">${escapeHtml(sub)}</span>` : '');
    c.appendChild(span);
    Run.autoScroll();
    return span;
  },

  settle(kind, title, sub = '') {
    const els = document.querySelectorAll('#log-console .banner-starting');
    const el = els[els.length - 1];
    if (!el) return;
    el.className = 'lvl-banner banner-' + kind;
    el.querySelector('strong').textContent = title;
    let subEl = el.querySelector('.banner-sub');
    if (sub && !subEl) {
      subEl = document.createElement('span');
      subEl.className = 'banner-sub';
      el.appendChild(subEl);
    }
    if (subEl) subEl.textContent = sub;
  },

  settleStart() {
    if (!Run.stopping) Run.settle('started', 'Pipeline started');
  },

  // "<PREFIX><index>\t<name>[\t<qc test>]" -> [index, name, test|null].
  splitMarker(plain, prefix) {
    const parts = plain.slice(prefix.length).split('\t');
    return [
      parseInt(parts[0], 10),
      (parts[1] || '').trim(),
      parts.length > 2 ? parts[2].trim() : null,
    ];
  },

  // One pausable unit: a step, or one QC test within a split step.
  unitKey(index, test) {
    return test ? index + ' ' + test : String(index);
  },

  // A leave=False tqdm bar ends with a bare "\r", so a marker can land mid-line.
  markerAt(plain) {
    let best = null;
    for (const marker of [Run.FIG_MARKER, Run.LOG_MARKER, Run.FAIL_MARKER, Run.STEP_MARKER,
      Run.PAUSE_MARKER, Run.RERUN_MARKER, Run.MEM_MARKER, Run.REPORT_MARKER, Run.VARS_MARKER,
      Run.TIME_MARKER, Run.SAMPLE_MARKER]) {
      const at = plain.indexOf(marker);
      if (at >= 0 && (best === null || at < best.at)) best = { marker, at };
    }
    return best;
  },

  handleLine(line) {
    // Uncoloured, so a leading colour code can't hide a marker.
    const plain = Run.stripAnsi(line);
    const hit = Run.markerAt(plain);
    if (hit) {
      if (hit.at > 0) Run.renderLine(plain.slice(0, hit.at));
      // RAM samples arrive mid-step, so they must not freeze a live bar.
      if (hit.marker !== Run.SAMPLE_MARKER) Run.finalizeProgress();
      Run.handleMarker(hit.marker, plain.slice(hit.at));
      return;
    }
    Run.renderLine(line);
  },

  handleMarker(marker, plain) {
    if (marker === Run.MEM_MARKER) {
      // "<rss>\t<peak>\t<data>\t<label>" for the RAM meter, not the console.
      Mem.add(plain.slice(marker.length));
      return;
    }
    if (marker === Run.SAMPLE_MARKER) {
      Mem.sample(plain.slice(marker.length));
      return;
    }
    if (marker === Run.TIME_MARKER) {
      RunClock.update(plain.slice(marker.length));
      return;
    }
    if (marker === Run.VARS_MARKER) {
      try {
        const vars = JSON.parse(plain.slice(marker.length));
        Run.variables = vars.names; Run.emptyVariables = vars.empty;
      } catch (e) { /* malformed */ }
      return;
    }
    if (marker === Run.REPORT_MARKER) {
      // "<abspath>\t<filename>"
      const parts = plain.slice(marker.length).split('\t');
      const path = (parts[0] || '').trim();
      const name = (parts[1] || '').trim() || path;
      if (path) {
        Run.showReport(path, name);
        Run.reportName = name;
        Run.note('report', name + ' — open it in the Output tab', 'ok');
      }
      return;
    }
    if (marker === Run.LOG_MARKER || marker === Run.FAIL_MARKER) {
      // "<index>\t<step>\t<qc test>\t<base64 text>"; base64 so newlines can't break the line.
      const parts = plain.slice(marker.length).split('\t');
      const idx = parseInt(parts[0], 10);
      const name = (parts[1] || '').trim();
      const test = (parts[2] || '').trim() || null;
      const isError = marker === Run.FAIL_MARKER;
      let text = '';
      try { text = decodeURIComponent(escape(atob(parts[3] || ''))); } catch (e) { /* malformed payload */ }
      if (Number.isInteger(idx) && text) {
        Run.addLog(idx, name, test, text, { isError });
        if (isError) {
          Run.pauseFailed = true;
          Run.note('failed', text.split('\n')[0], 'err');
        } else {
          Run.note('diagnostics', (test || name) + ' (log)');
        }
      }
      return;
    }
    const [idx, rest, test] = Run.splitMarker(plain, marker);
    if (marker === Run.FIG_MARKER) {
      // "<filename>\t<caption>\t<spec>\t<reason>"; an empty spec means PNG-only, <reason> says why.
      const parts = plain.slice(marker.length).split('\t');
      const fname = (parts[0] || '').trim();
      const caption = (parts[1] || '').trim();
      const spec = (parts[2] || '').trim();
      const reason = (parts[3] || '').trim();
      Run.addPlot(fname, caption, spec);
      Run.note('plot', (caption || fname) +
        (spec ? ' (interactive)' : reason ? ` (image only — ${reason})` : ''));
    } else if (marker === Run.STEP_MARKER) {
      // A garbled index would make every figure its own group (NaN !== NaN).
      if (!Number.isInteger(idx)) return;
      Run.currentStep = { index: idx, name: rest, test, key: Run.unitKey(idx, test) };
      Run.activeGroup = null; // each execution opens a fresh attempt
      Run.pauseFailed = false;
      RunLock.stepStarted(idx, test);
    } else if (marker === Run.PAUSE_MARKER) {
      if (!Number.isInteger(idx)) return;
      Run.showPause(idx, rest, test);
    } else if (marker === Run.RERUN_MARKER) {
      Run.setStatus('re-running step…', 'running');
    }
  },

  renderLine(line) {
    // The final bar frame arrives newline-terminated: finalise it rather than duplicate it.
    if (Run.progressEl && Run.looksLikeProgress(Run.stripAnsi(line))
        && Run.barDesc(line) === Run.barDesc(Run.progressEl.textContent || '')) {
      Run.progressEl.innerHTML = Run.ansiToHtml(line) + '\n';
      Run.progressEl = null;
      Run.autoScroll();
      return;
    }
    Run.finalizeProgress();
    Run.append(line);
  },

  // The text before the percentage: "<time>  <step>  <desc>", minus the time.
  barDesc(plain) {
    const m = Run.stripAnsi(plain).match(/^\S+\s+(.*?)\s*\d+%\|/);
    return m ? m[1] : Run.stripAnsi(plain);
  },

  // A leave=False bar erases itself without a 100% frame, so it could freeze mid-way.
  finalizeProgress() {
    if (!Run.progressEl) return;
    const plain = Run.stripAnsi(Run.progressEl.textContent || '').replace(/\n+$/, '');
    const m = plain.match(/^(.*?)\d+%\|([^|]*)\|\s*\d+\/(\d+)(.*)$/);
    if (m) {
      const [, desc, bar, total, tail] = m;
      const doneTail = tail.replace(/<.*?\]/, '<00:00]');
      const line = `${desc}100%|${'█'.repeat(bar.length)}| ${total}/${total}${doneTail}`;
      Run.progressEl.innerHTML = Run.ansiToHtml(line) + '\n';
    }
    Run.progressEl = null;
  },

  handleProgress(line) {
    Run.settleStart();
    const c = document.getElementById('log-console');
    const plain = Run.stripAnsi(line);
    if (!plain.trim()) return; // a closing bar's blank erase frame
    // A new description means a new loop: keep the finished bar, start another line.
    if (Run.progressEl && Run.looksLikeProgress(plain)
        && Run.barDesc(plain) !== Run.barDesc(Run.progressEl.textContent || '')) {
      Run.finalizeProgress();
    }
    if (!Run.progressEl) {
      Run.progressEl = document.createElement('span');
      Run.progressEl.className = 'lvl-progress';
      c.appendChild(Run.progressEl);
    }
    Run.progressEl.innerHTML = Run.ansiToHtml(line) + '\n';
    Run.autoScroll();
  },

  _groupFor(cur) {
    if (!Run.activeGroup || Run.activeGroup.key !== cur.key) {
      Run.activeGroup = {
        index: cur.index, key: cur.key, step: cur.name, test: cur.test,
        figs: [], params: Run.pendingParams,
      };
      Run.pendingParams = null;
      Run.groups.push(Run.activeGroup);
    }
    return Run.activeGroup;
  },

  addPlot(fname, caption, spec) {
    if (!fname) return;
    const cur = Run.currentStep ||
      { index: -1, name: 'Diagnostics', test: null, key: Run.unitKey(-1, null) };
    Run._groupFor(cur).figs.push({
      fname, caption, spec: spec || null, url: Viewer.freshUrl(fname),
    });
    Run.plotCount += 1;
    Run.renderGallery();
    Run.updatePlotTab();
    if (Review.active && Review.key === cur.key) Review.renderPlots();
  },

  // Stored as a pseudo-figure (isLog) so attempts and re-runs work unchanged; only Viewer.card() differs.
  addLog(idx, name, test, text, { isError = false } = {}) {
    if (!text) return;
    const cur = Run.currentStep && Run.currentStep.index === idx
      ? Run.currentStep
      : { index: idx, name, test, key: Run.unitKey(idx, test) };
    Run._groupFor(cur).figs.push({ isLog: true, isError, text, caption: '' });
    Run.renderGallery();
    if (Review.active && Review.key === cur.key) Review.renderPlots();
  },

  groupsFor(key) {
    return Run.groups.filter((g) => g.key === key);
  },

  // Derived from position: a stored counter drifts when groups are dropped or re-keyed.
  attemptNo(group) {
    return Run.groupsFor(group.key).indexOf(group) + 1;
  },

  dropGroup(group) {
    const at = Run.groups.indexOf(group);
    if (at < 0) return;
    Run.groups.splice(at, 1);
    Run.plotCount -= group.figs.filter((f) => !f.isLog).length;
    if (Run.activeGroup === group) Run.activeGroup = null;
    Run.renderGallery();
    Run.updatePlotTab();
  },

  // The Plots tab keeps only the accepted attempt, not every experiment.
  keepOnlyAttempt(key, group) {
    for (const g of Run.groupsFor(key)) {
      if (g !== group) Run.dropGroup(g);
    }
  },

  // Groups captured before any step marker have index -1.
  adoptOrphans(index, name, test) {
    const orphans = Run.groups.filter((g) => g.index === -1);
    if (!orphans.length) return;
    for (const g of orphans) {
      g.index = index; g.step = name; g.test = test;
      g.key = Run.unitKey(index, test);
    }
    Run.renderGallery();
  },

  // The paused step's figures are shown by Review, so they're left out here.
  renderGallery() {
    const gallery = document.getElementById('plots-gallery');
    gallery.innerHTML = '';
    const shown = Run.groups.filter((g) => !(Review.active && g.key === Review.key));
    document.getElementById('plots-empty').classList.toggle('hidden', !!shown.length || Review.active);
    for (const g of shown) {
      const total = Run.groupsFor(g.key).length;
      const sec = document.createElement('section');
      sec.className = 'plot-step';
      const h = document.createElement('h4');
      h.className = 'plot-step-title';
      const title = g.test ? `${g.step} · ${g.test}` : g.step;
      h.textContent = total > 1 ? `${title} · attempt ${Run.attemptNo(g)}` : title;
      sec.appendChild(h);
      const cards = document.createElement('div');
      cards.className = 'plot-cards';
      g.figs.forEach((_, i) => cards.appendChild(Viewer.card(g.figs, i)));
      sec.appendChild(cards);
      gallery.appendChild(sec);
    }
  },

  updatePlotTab() {
    const tab = document.querySelector('.tab[data-tab="plots"]');
    if (tab) tab.textContent = Run.plotCount ? `Plots (${Run.plotCount})` : 'Plots';
  },

  clearPlots() {
    Run.groups = [];
    Run.activeGroup = null;
    Run.pendingParams = null;
    Run.plotCount = 0;
    Run.currentStep = null;
    // Spec filenames restart at fig_001.json each run, so a cached spec would be stale.
    Plot._cache = {};
    Run.renderGallery();
    Run.updatePlotTab();
    Run.clearReport();
  },

  showReport(path, name) {
    const url = '/api/run/report?path=' + encodeURIComponent(path);
    const view = document.getElementById('report-view');
    view.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'report-head';
    const meta = document.createElement('div');
    meta.className = 'report-meta';
    meta.innerHTML = `<strong>${escapeHtml(name)}</strong>` +
      `<span class="report-path">${escapeHtml(path)}</span>`;
    const open = document.createElement('a');
    open.className = 'btn primary';
    open.href = url;
    open.target = '_blank';
    open.rel = 'noopener';
    open.innerHTML = `${Icon.svg('external', 14)}Open PDF`;
    head.appendChild(meta);
    head.appendChild(open);
    const frame = document.createElement('iframe');
    frame.className = 'report-frame';
    frame.title = name;
    frame.src = url;
    view.appendChild(head);
    view.appendChild(frame);
    view.classList.remove('hidden');
    document.getElementById('report-empty').classList.add('hidden');
    const tab = document.querySelector('.tab[data-tab="report"]');
    if (tab) tab.classList.add('has-new');
  },

  clearReport() {
    const view = document.getElementById('report-view');
    if (view) { view.innerHTML = ''; view.classList.add('hidden'); }
    const empty = document.getElementById('report-empty');
    if (empty) empty.classList.remove('hidden');
    const tab = document.querySelector('.tab[data-tab="report"]');
    if (tab) tab.classList.remove('has-new');
  },

  showPause(idx, name, test) {
    Run.pausedStep = idx;
    Run.pausedName = name;
    Run.pausedTest = test || null;
    const key = Run.unitKey(idx, test);
    // Orphan figures can only be this step's; attempt 1's params come from the config.
    Run.adoptOrphans(idx, name, test);
    const group = Run.groupsFor(key).slice(-1)[0];
    if (group && !group.params) group.params = Run.paramsAt(idx, test);
    // A re-run that drew nothing leaves this set.
    Run.pendingParams = null;
    const ran = Run.sentParams || Run.paramsAt(idx, test);
    Run.sentParams = null;
    Run.outcomes.push({ params: ran, failed: Run.pauseFailed });
    if (!Run.pauseFailed) Run.heldParams = ran;
    Run.setStatus(Run.pauseFailed ? 'step failed' : 'paused', Run.pauseFailed ? 'err' : 'running');
    Run.setRunButton('paused');
    RunLock.pauseAt(idx, test);
    if (Review.active && Review.key === key) {
      Review.setBusy(false);   // a re-run finished
      Review.select(null);
      Review.renderTitle();
    } else {
      Review.show(idx, name, test);
    }
    if (Run.continueAfter) {
      Run.continueAfter = false;
      if (!Run.pauseFailed) Run.continueRun();
    }
  },

  hidePause() {
    Run.pausedStep = null;
    Run.pausedName = null;
    Run.pausedTest = null;
    Run.outcomes = [];
    Run.heldParams = null;
    Run.continueAfter = false;
    if (RunLock.running) RunLock.pauseAt(null, null);
    Run.setRunButton(RunLock.running ? 'running' : 'idle');
    Review.hide();
  },

  // Read from the YAML pane, which holds both builder and hand edits.
  paramsAt(idx, test) {
    let steps;
    try {
      steps = (jsyaml.load(editor.getValue()) || {}).steps || [];
    } catch (e) {
      return null;
    }
    const step = steps[idx];
    if (!step) return null;
    const params = step.parameters || {};
    if (!test) return params;
    const settings = (params.qc_settings || {})[test];
    return settings === undefined ? null : { qc_settings: { [test]: settings } };
  },

  // 'continue', 'skip' (form failed), 'rerun' (passed in an older attempt) or 'untested'.
  continueAction() {
    const form = Run.paramsAt(Run.pausedStep, Run.pausedTest);
    if (!form) return Run.heldParams ? 'continue' : 'skip';
    if (Run.heldParams && Forms.equal(form, Run.heldParams)) return 'continue';
    const known = Run.outcomes.filter((o) => Forms.equal(o.params, form)).pop();
    if (!known) return 'untested';
    return known.failed ? 'skip' : 'rerun';
  },

  async continueRun() {
    if (Run.pausedStep === null) return;
    const action = Run.continueAction();
    if (action === 'rerun' || action === 'untested') {
      await Run.rerunStep({ thenContinue: true });
      return;
    }
    const key = Run.unitKey(Run.pausedStep, Run.pausedTest);
    const form = Run.paramsAt(Run.pausedStep, Run.pausedTest);
    try {
      await (action === 'skip' ? API.skipStep() : API.continueRun());
    } catch (e) {
      Run.note('error', e.message, 'err');
      return;
    }
    const attempts = Run.groupsFor(key);
    const kept = attempts.filter((g) => g.params && Forms.equal(g.params, form)).pop();
    Run.keepOnlyAttempt(key, kept || attempts[attempts.length - 1]);
    // The next pause may already have arrived while the request was in flight.
    if (Run.unitKey(Run.pausedStep, Run.pausedTest) !== key) return;
    Run.hidePause();
    Run.setStatus('running…', 'running');
  },

  // Refuse if an edit moved this step, since the index must match the running pipeline.
  async rerunStep({ thenContinue = false } = {}) {
    if (Run.pausedStep === null) return;
    const idx = Run.pausedStep;
    const test = Run.pausedTest;
    let steps;
    try {
      steps = (jsyaml.load(editor.getValue()) || {}).steps || [];
    } catch (e) {
      alert('Cannot parse the YAML to re-run: ' + e.message);
      return;
    }
    const step = steps[idx];
    if (!step) {
      alert('Could not find step ' + (idx + 1) + ' in the current config.');
      return;
    }
    if (step.name !== Run.pausedName) {
      alert(`Step ${idx + 1} is now '${step.name}', not '${Run.pausedName}'. ` +
        'Undo the reordering, or Continue and start a fresh run.');
      return;
    }
    const params = Run.paramsAt(idx, test);
    if (!params) {
      alert(test
        ? `QC test '${test}' is no longer configured on step ${idx + 1}.`
        : `Could not read the parameters of step ${idx + 1}.`);
      return;
    }
    // Unchanged params give an identical figure, so replace that attempt instead of stacking one.
    const latest = Run.groupsFor(Run.unitKey(idx, test)).slice(-1)[0];
    if (latest && latest.params && Forms.equal(latest.params, params)) {
      Run.note('re-run', 'unchanged parameters — replacing attempt ' + Run.attemptNo(latest));
      Run.dropGroup(latest);
    }
    Run.note('re-run', 'step ' + (idx + 1) + (test ? ` (${test})` : '') +
      ' with ' + JSON.stringify(params));
    Run.pendingParams = params;
    Run.sentParams = params;
    Run.continueAfter = thenContinue;
    Run.activeGroup = null;
    Review.setBusy(true);
    Run.setStatus('re-running…', 'running');
    Run.setRunButton('busy');
    try {
      await API.rerunStep(params, editor.getValue());
    } catch (e) {
      Run.note('error', e.message, 'err');
      Run.pendingParams = null;
      Run.sentParams = null;
      Run.continueAfter = false;
      Review.setBusy(false);
      Run.setStatus('paused', 'running');
      Run.setRunButton('paused');
    }
  },

  setStatus(text, cls) {
    const el = document.getElementById('run-status');
    el.textContent = text;
    el.className = 'run-status' + (cls ? ' ' + cls : '');
    if (text) document.getElementById('mem-meter').classList.remove('hidden');
  },

  // Doubles as Continue while paused ('busy': a re-run is in flight); reads Skip if the form is known to fail.
  setRunButton(mode) {
    const btn = document.getElementById('btn-run');
    Run.runBtnMode = mode;
    const paused = mode === 'paused' || mode === 'busy';
    const action = paused ? Run.continueAction() : null;
    btn.classList.toggle('primary', action !== 'skip');
    btn.classList.toggle('warn', action === 'skip');
    btn.title = action === 'untested' ? 'These values have not been run yet: Continue re-runs the step first'
      : action === 'rerun' ? 'Continue re-runs the step with these values first' : '';
    Review.renderHint(action === 'untested'
      ? "These values haven't been run yet. Re-run first? Continue will re-run the step, then carry on if it passes."
      : '');
    if (paused) {
      btn.disabled = mode === 'busy';
      btn.innerHTML = Icon.svg('play') + (action === 'skip' ? 'Skip step' : 'Continue');
    } else {
      btn.disabled = mode === 'running';
      btn.innerHTML = Icon.svg('play') + 'Run';
    }
    Run.syncRunButtons();
  },

  syncToForm() {
    if (Run.pausedStep === null || ManualQC.isActive()) return;
    Run.setRunButton(Run.runBtnMode);
    Review.syncUseButtons();
  },

  // Clear/Stop/Run always show so the bar doesn't shift; only a paused run adds Re-run.
  syncRunButtons() {
    const paused = Run.runBtnMode === 'paused' || Run.runBtnMode === 'busy';
    const rerun = document.getElementById('btn-rerun');
    rerun.classList.toggle('hidden', !paused);
    rerun.disabled = Run.runBtnMode === 'busy';
  },

  setStopClear(mode) {
    document.getElementById('btn-stop').disabled = mode !== 'stop';
    document.getElementById('btn-clear').disabled = mode !== 'clear';
  },

  async start(yamlContent) {
    Run.showTab();
    Run.setRunButton('running');
    Run.setStopClear('stop');
    Run.setStatus('starting…', 'running');
    Run.pendingStart = true;
    try {
      await API.run(yamlContent);
    } catch (e) {
      // Usually a dropped stream (laptop sleep) left the buttons stale: attach to the real run.
      if (/already running/i.test(e.message)) {
        Run.setStatus('re-attaching to the running pipeline…', 'running');
        Run.connect(true);
        return;
      }
      Run.setStatus('failed to start: ' + e.message, 'err');
      Run.pendingStart = false;
      Run.banner('err', 'Could not start', e.message);
      Run.setRunButton('idle');
      Run.setStopClear('idle');
      return;
    }
    Run.connect(true);
  },

  // The stream replays the backlog first, so a reconnect repaints the whole log.
  connect(clearConsole) {
    if (clearConsole) {
      document.getElementById('log-console').textContent = '';
      Run.scrollToBottom();
      Run.clearPlots();
      Run.reportName = null;
      Mem.reset();
      RunClock.reset();
      if (Run.pendingStart) Run.banner('starting', 'Starting pipeline');
    }
    Run.pendingStart = false;
    Run.hidePause();
    RunLock.begin();
    Run.setRunButton('running');
    Run.setStopClear('stop');
    Run.setStatus('running…', 'running');
    Run.progressEl = null;
    Run.stopping = false;
    if (Run.source) Run.source.close();
    Run.source = new EventSource('/api/run/stream');
    Run.source.onmessage = (ev) => Run.handleLine(ev.data);
    Run.source.addEventListener('progress', (ev) => Run.handleProgress(ev.data));
    Run.source.addEventListener('end', (ev) => {
      Run.finalizeProgress();
      RunClock.stop();
      const code = Number(ev.data);
      const took = RunClock.epoch === null ? '' : RunClock.fmt(RunClock.seconds());
      if (Run.stopping) {
        Run.setStatus('stopped', 'err');
        Run.settle('stopped', 'Pipeline stopped', took);
      } else if (code === 0) {
        Run.setStatus('finished', 'ok');
        Run.banner('ok', 'Pipeline finished', [took,
          Run.plotCount ? `${Run.plotCount} plot${Run.plotCount === 1 ? '' : 's'}` : '',
          Run.reportName ? 'report ready in the Output tab' : ''].filter(Boolean).join(' · '));
      } else if (code < 0) {
        // Killed by a signal we didn't send, most often the OS reclaiming memory.
        Run.setStatus('killed', 'err');
        Run.banner('err', 'Pipeline killed', ['possibly out of memory', took].filter(Boolean).join(' · '));
      } else {
        Run.setStatus('failed', 'err');
        Run.banner('err', 'Pipeline failed');
      }
      Run.cleanup();
      Outputs.refresh();
    });
    Run.source.onerror = () => Run.handleDrop();
  },

  // A dropped stream (laptop sleep) doesn't end the run: re-attach and let the backlog replay.
  handleDrop() {
    if (Run.source) { Run.source.close(); Run.source = null; }
    clearTimeout(Run._retry);
    if (Run.stopping) return; // Stop is in flight; the end event settles it
    Run.setStatus('reconnecting…', 'running');
    API.runStatus().then((s) => {
      if (s.running) {
        Run._retry = setTimeout(() => Run.connect(true), 1500);
      } else {
        Run.setStatus('run ended while disconnected', '');
        Run.cleanup();
      }
    }).catch(() => {
      Run._retry = setTimeout(() => Run.handleDrop(), 3000);
    });
  },

  // Called when the tab is shown again, e.g. after laptop sleep.
  async ensureConnected() {
    if (Run.source || Run.stopping) return;
    try {
      const s = await API.runStatus();
      if (s.running) Run.connect(true);
    } catch (e) { /* server not reachable yet; the next event will retry */ }
  },

  async resumeIfRunning() {
    try {
      const s = await API.runStatus();
      if (s.running) {
        Run.showTab();
        Run.connect(true);
      }
    } catch (e) { /* server not ready; ignore */ }
  },

  showTab(name = 'run') {
    document.querySelectorAll('.tab').forEach((t) =>
      t.classList.toggle('on', t.dataset.tab === name));
    document.querySelectorAll('.tab-panel').forEach((p) =>
      p.classList.toggle('hidden', p.dataset.panel !== name));
    if (name === 'report') Outputs.refresh();
    Run.onTabChange();
  },

  onTabChange() {
    const which = document.querySelector('.tab.on')?.dataset.tab;
    document.body.classList.toggle('manual-full', which === 'manual');
  },

  // Only on a finished/stopped run, never a transient stream drop (see handleDrop).
  cleanup() {
    Run.progressEl = null;
    clearTimeout(Run._retry);
    Run.hidePause();
    RunLock.end();
    if (Run.source) { Run.source.close(); Run.source = null; }
    Run.setRunButton('idle');
    Run.setStopClear('clear');
  },

  async stop() {
    Run.settleStart();
    Run.stopping = true;
    Run.setStatus('stopping…', 'running');
    Run.banner('starting', 'Stopping pipeline');
    try {
      await API.stopRun();
    } catch (e) {
      Run.stopping = false;
      Run.setStatus('stop failed: ' + e.message, 'err');
      return;
    }
    Run.clearPlots();
  },

  clearRun() {
    document.getElementById('log-console').textContent = '';
    Run.clearPlots();
    Run.setStatus('', '');
    Run.setStopClear('idle');
  },
};
