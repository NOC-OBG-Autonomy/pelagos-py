// Bootstrap: load the registry, build the UI, wire controls together.

let editor = null;         // CodeMirror YAML pane
let syncingFromBuilder = false;
let errorLineHandle = null;
let activeStepId = null;
let stepHighlightLines = null;
let activeFieldPath = null;
let fieldHighlightLines = null;
let lastFocusedStep = null;
let lastFocusedKey = null;

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// Line spans (0-based, inclusive) of each step under `steps:`, indexed like STATE.pipeline.items.
function stepLineRanges() {
  const lines = editor.getValue().split('\n');
  let inSteps = false, stepsIndent = null;
  const starts = [];
  for (let i = 0; i < lines.length; i++) {
    if (!inSteps) { if (/^steps:\s*$/.test(lines[i])) inSteps = true; continue; }
    const m = lines[i].match(/^(\s*)-\s/);
    if (!m) continue;
    if (stepsIndent === null) stepsIndent = m[1].length;
    if (m[1].length === stepsIndent) starts.push(i);
  }
  return starts.map((start, k) => ({
    start,
    end: (k + 1 < starts.length ? starts[k + 1] - 1 : lines.length - 1),
  }));
}

function stepIndexAtLine(line) {
  const ranges = stepLineRanges();
  for (let i = 0; i < ranges.length; i++) {
    if (line >= ranges[i].start && line <= ranges[i].end) return i;
  }
  return null;
}

// Counts a list dash as one more level, so `- DEPTH` nests under `to_derive:` at the same column.
function yamlIndent(line) {
  const m = line.match(/^(\s*)(-\s)?/);
  return m[1].length + (m[2] ? 1 : 0);
}

function yamlKey(line) {
  const m = line.match(/^\s*(?:-\s+)?(['"]?)(.+?)\1\s*:(?:\s|$)/);
  return m ? m[2] : null;
}

const yamlBlank = (line) => !line.trim() || line.trim().startsWith('#');

// e.g. ['parameters', 'qc_settings', 'range qc', 'variable_ranges'].
function yamlPathAtLine(line) {
  const lines = editor.getValue().split('\n');
  if (line >= lines.length || yamlBlank(lines[line])) return [];
  const stepRange = stepLineRanges().find((r) => line >= r.start && line <= r.end);
  const top = stepRange ? stepRange.start : -1;
  const path = [];
  let indent = Infinity;
  for (let i = line; i > top; i--) {
    if (yamlBlank(lines[i]) || yamlIndent(lines[i]) >= indent) continue;
    indent = yamlIndent(lines[i]);
    const key = yamlKey(lines[i]);
    if (key != null) path.unshift(key);
    if (indent === 0) break;
  }
  return path;
}

// Only matches keys at the block's own depth, not a same-named key deeper down.
function yamlBlockForPath(from, to, path) {
  const lines = editor.getValue().split('\n');
  let block = { start: from, end: Math.min(to, lines.length - 1) };
  let inner = from;
  for (const key of path) {
    const depth = Math.min(...lines.slice(inner, block.end + 1).filter((l) => !yamlBlank(l)).map(yamlIndent));
    let found = null;
    for (let i = inner; i <= block.end; i++) {
      if (yamlBlank(lines[i]) || yamlIndent(lines[i]) !== depth || yamlKey(lines[i]) !== key) continue;
      let end = i;
      for (let j = i + 1; j <= block.end; j++) {
        if (yamlBlank(lines[j])) continue;
        if (yamlIndent(lines[j]) <= depth) break;
        end = j;
      }
      found = { start: i, end };
      break;
    }
    if (!found) return null;
    block = found;
    inner = found.start + 1;
  }
  return block;
}

function clearStepHighlight() {
  if (editor && stepHighlightLines) {
    for (let ln = stepHighlightLines.start; ln <= stepHighlightLines.end && ln < editor.lineCount(); ln++) {
      editor.removeLineClass(ln, 'background', 'cm-step-highlight');
    }
  }
  if (editor && fieldHighlightLines) {
    for (let ln = fieldHighlightLines.start; ln <= fieldHighlightLines.end && ln < editor.lineCount(); ln++) {
      editor.removeLineClass(ln, 'background', 'cm-field-highlight');
    }
  }
  stepHighlightLines = null;
  fieldHighlightLines = null;
}

// A null id with a ['pipeline', key] path is a Pipeline Settings box.
function highlightYamlForStep(id, path = null) {
  activeStepId = id;
  activeFieldPath = path;
  applyStepHighlight();
}

function applyStepHighlight() {
  if (!editor) return;
  clearStepHighlight();
  let r = { start: 0, end: editor.lineCount() - 1 };
  if (activeStepId != null) {
    const idx = STATE.pipeline.items.findIndex((i) => i.id === activeStepId);
    r = idx < 0 ? null : stepLineRanges()[idx];
    if (!r) return;
    for (let ln = r.start; ln <= r.end && ln < editor.lineCount(); ln++) {
      editor.addLineClass(ln, 'background', 'cm-step-highlight');
    }
    stepHighlightLines = r;
  } else if (!activeFieldPath) {
    return;
  }
  const from = activeStepId != null ? r.start + 1 : r.start;
  const f = activeFieldPath && yamlBlockForPath(from, r.end, activeFieldPath);
  if (f) {
    for (let ln = f.start; ln <= f.end; ln++) editor.addLineClass(ln, 'background', 'cm-field-highlight');
    fieldHighlightLines = f;
  }
  const view = f || stepHighlightLines;
  if (view) editor.scrollIntoView({ from: { line: view.start, ch: 0 }, to: { line: view.end, ch: 0 } });
}

function refreshYAML() {
  if (!editor) return;
  Config.noteEdit();
  syncingFromBuilder = true;
  editor.setValue(Config.toYAML());
  syncingFromBuilder = false;
  clearYamlError();
  applyStepHighlight(); // setValue drops line classes; re-mark the active step
  Config.checkDataFile();
  Config.updateStatus();
  scheduleValidate();
  Inspect.schedule();
  Run.syncToForm();
}

let validateToken = 0;
async function runValidate() {
  if (!editor) return;
  const token = ++validateToken;
  showValidating();
  let result;
  try {
    result = await API.validate(editor.getValue());
  } catch (e) {
    result = { ok: false, error: e.message };
  }
  if (token !== validateToken) return; // a newer edit has been sent since
  if (result.error) {
    const statusHost = document.getElementById('validation-status');
    statusHost.replaceChildren(statusBar('err', 'Could not validate', result.error));
    return;
  }
  renderValidation(result);
}
const scheduleValidate = debounce(runValidate, 400);

// Leaves the editor text alone so hand-typed formatting is kept.
function syncYamlToBuilder() {
  if (!editor) return;
  clearYamlError();
  let cfg;
  try {
    cfg = jsyaml.load(editor.getValue());
  } catch (e) {
    markYamlError(e);
    return;
  }
  try {
    Config.fromObject(cfg || {}, Config.sectionsFromYAML(editor.getValue()));
    Config.checkDataFile();
    // fromObject makes new item objects, so re-apply the lock to the paused step.
    if (RunLock.running && RunLock.index !== null) {
      RunLock.pauseAt(RunLock.index, RunLock.test);
    }
  } catch (e) { /* structurally odd but parseable: leave builder, Validate reports it */ }
}
const scheduleYamlToBuilder = debounce(syncYamlToBuilder, 400);

function clearYamlError() {
  if (editor && errorLineHandle != null) {
    editor.removeLineClass(errorLineHandle, 'background', 'cm-error-line');
    editor.removeLineClass(errorLineHandle, 'gutter', 'cm-error-line');
  }
  errorLineHandle = null;
}

function markYamlError(e) {
  clearYamlError();
  const line = e && e.mark && typeof e.mark.line === 'number' ? e.mark.line : null;
  if (line == null || line >= editor.lineCount()) return;
  errorLineHandle = editor.addLineClass(line, 'background', 'cm-error-line');
  editor.addLineClass(errorLineHandle, 'gutter', 'cm-error-line');
}

// No-op once a result is showing (no flicker while typing) unless force is set.
function showValidating(force = false) {
  const statusHost = document.getElementById('validation-status');
  if (!statusHost || (statusHost.firstChild && !force)) return;
  statusHost.innerHTML = '';
  statusHost.appendChild(statusBar('pending', 'Validating…', 'Checking the config against the step schemas'));
  if (force) document.getElementById('top-slot').classList.remove('has-error');
}

function renderValidation(result) {
  const statusHost = document.getElementById('validation-status');
  statusHost.innerHTML = '';
  setIssueMarks(issueMarksFor(result.issues || []));
  document.getElementById('top-slot').classList.toggle('has-error', !result.ok);

  if (result.yaml_error) {
    const y = formatYamlError(result.yaml_error);
    const bar = statusBar('err', 'YAML syntax error', y.location || 'Check the YAML pane');
    let body = `<div class="v-msg">${escapeHtml(y.message)}</div>`;
    if (y.snippet) body += `<pre class="v-snippet">${escapeHtml(y.snippet)}</pre>`;
    bar.appendChild(issueBody(body));
    statusHost.appendChild(bar);
    return;
  }

  if (result.ok) {
    statusHost.appendChild(statusBar('ok', 'Valid', "Matches every step's parameter schema"));
    return;
  }

  // Missing data is the most fundamental problem, so show it first.
  const ranked = [...result.issues].sort((a, b) =>
    (parseIssue(b.error).critical ? 1 : 0) - (parseIssue(a.error).critical ? 1 : 0));

  if (ranked.length === 1) {
    const issue = ranked[0];
    const parsed = parseIssue(issue.error);
    const bar = statusBar('err', parsed.tag, issueWhere(issue));
    bar.classList.add('v-single');
    bar.appendChild(issueBody(parsed.html));
    linkToStep(bar, issue);
    statusHost.appendChild(bar);
    return;
  }
  const bar = statusBar('err', `${ranked.length} issues`, 'Fix before running');
  for (const issue of ranked) {
    const parsed = parseIssue(issue.error);
    const row = issueBody(
      `<div class="v-msg"><strong>${escapeHtml(parsed.tag)}</strong> ` +
      `<span class="v-muted">· ${issueWhere(issue)}</span></div>${parsed.html}`);
    linkToStep(row, issue);
    bar.appendChild(row);
  }
  statusHost.appendChild(bar);
}

function issueMarksFor(issues) {
  const marks = new Map();
  for (const issue of issues) {
    const item = issue.index == null ? null : STATE.pipeline.items[issue.index];
    if (!item) continue;
    const fields = marks.get(item.id) || [];
    fields.push(...(parseIssue(issue.error).fields || []));
    marks.set(item.id, fields);
  }
  return marks;
}

function statusBar(kind, title, sub) {
  const bar = document.createElement('div');
  bar.className = `v-status v-${kind}`;
  const icon = kind === 'ok' ? 'check' : kind === 'pending' ? 'rerun' : 'alert';
  bar.innerHTML = `<span class="v-icon">${Icon.svg(icon, 14)}</span>` +
    `<div class="v-text"><strong>${escapeHtml(title)}</strong>` +
    `<span>${escapeHtml(sub)}</span></div>`;
  return bar;
}

function issueBody(html) {
  const body = document.createElement('div');
  body.className = 'v-body';
  body.innerHTML = html;
  return body;
}

function issueWhere(issue) {
  if (issue.index == null) return 'Pipeline';
  return `Step ${issue.index + 1}${issue.name ? ' · ' + escapeHtml(issue.name) : ''}`;
}

function linkToStep(el, issue) {
  const item = issue.index == null ? null : STATE.pipeline.items[issue.index];
  if (!item) return;
  el.classList.add('v-clickable');
  el.title = 'Show in YAML and the builder';
  el.onclick = (e) => {
    if (e.target.closest('[data-tab]')) {
      e.preventDefault();
      Run.showTab(e.target.closest('[data-tab]').dataset.tab);
      return;
    }
    highlightYamlForStep(item.id);
    focusStepInBuilder(issue.index);
  };
}

// Message shapes come from utils/parameter_spec.py resolve().
function parseIssue(raw) {
  const msg = String(raw).replace(/^\[[^\]]*\]\s*/, '');

  let m = msg.match(/^Multiple data-loading steps found:\s*(.+?)\.\s*(.*)$/s);
  if (m) {
    return { kind: 'load', tag: 'Multiple data sources', critical: true,
      html: `<div class="v-msg">Only one step should load or generate the ` +
        `pipeline's base data.</div><div class="v-detail v-muted">${escapeHtml(m[1])}</div>` };
  }
  m = msg.match(/^'Load OG1' has no 'file_path' set[^.]*\.\s*(.*)$/s);
  if (m) {
    return { kind: 'load', tag: 'No file set', critical: true,
      fields: [{ param: 'file_path', text: 'No file set' }],
      html: `<div class="v-msg">Choose an input file, or get a demo from ` +
        `<a href="#" data-tab="files">Files</a>.</div>` };
  }
  m = msg.match(/^Data file '(.*)' is not a NetCDF \(\.nc\) file\.$/s);
  if (m) {
    return { kind: 'load', tag: 'Not a NetCDF file', critical: true,
      fields: [{ param: 'file_path', text: 'Not a .nc file' }],
      html: `<div class="v-msg">Choose a .nc file, or get a demo from ` +
        `<a href="#" data-tab="files">Files</a>.</div>` };
  }
  m = msg.match(/^Could not find data file '(.*)'\.$/s);
  if (m) {
    return { kind: 'load', tag: 'File not found', critical: true,
      fields: [{ param: 'file_path', text: 'File not found' }],
      html: `<div class="v-msg">Check the path, or get a demo from ` +
        `<a href="#" data-tab="files">Files</a>.</div>` };
  }
  m = msg.match(/^Missing variables for(?: QC test)? '([^']+)':\s*([^.]+)\.\s*(?:[^.]*\.\s*)*No data-loading step[^.]*\.\s*(.*)$/s);
  if (m) {
    return { kind: 'load', tag: 'No data source', critical: true,
      html: `<div class="v-msg"><span class="chip">${escapeHtml(m[1])}</span> needs ` +
        `${chips(splitNames(m[2]), 'bad')}, but nothing in the pipeline loads data.</div>` +
        `<div class="v-detail">${escapeHtml(m[3])}</div>` };
  }

  m = msg.match(/^invalid parameter value\(s\):\s*(.+)$/s);
  if (m) {
    return { kind: 'value', tag: 'Invalid value',
      fields: segmentParams(m[1], 'Value not allowed'),
      html: m[1].split(';').map(formatValueSegment).join('') };
  }
  m = msg.match(/^invalid parameter type\(s\):\s*(.+)$/s);
  if (m) {
    return { kind: 'type', tag: 'Wrong type',
      fields: segmentParams(m[1], 'Wrong type'),
      html: m[1].split(';').map(formatTypeSegment).join('') };
  }
  m = msg.match(/^missing required parameter\(s\):\s*(.+)$/s);
  if (m) {
    return { kind: 'missing', tag: 'Missing',
      fields: splitNames(m[1]).map((param) => ({ param, text: 'Required' })),
      html: `<div class="v-detail">Add ${chips(splitNames(m[1]))}</div>` };
  }
  m = msg.match(/^unknown parameter\(s\):\s*([^.]+)\.\s*Valid parameters:\s*(.+?)\.?$/s);
  if (m) {
    return { kind: 'unknown', tag: 'Unknown',
      html: `<div class="v-detail">${chips(splitNames(m[1]), 'bad')} not recognised.</div>` +
        `<div class="v-detail v-muted">Valid: ${chips(splitNames(m[2]))}</div>` };
  }
  return { kind: 'other', tag: 'Error', html: `<div class="v-msg">${escapeHtml(msg)}</div>` };
}

// "a (expected ...); b (expected ...)" -> [{param: 'a', text}, {param: 'b', text}].
function segmentParams(segments, text) {
  return segments.split(';').map((seg) => ({ param: seg.trim().split(/[\s(]/)[0], text }));
}

// "to_derive (expected one of ['A','B'], got ['X'])" → param + allowed chips + bad value.
function formatValueSegment(seg) {
  const m = seg.match(/^\s*(\S+)\s*\(expected one of\s*(.+?),\s*got\s*(.+?)\)\s*$/s);
  if (!m) return `<div class="v-detail">${escapeHtml(seg.trim())}</div>`;
  const options = pyTokens(m[2]);
  const got = pyTokens(m[3]);
  return `<div class="v-detail"><span class="chip">${escapeHtml(m[1])}</span>: ` +
    `${chips(got, 'bad')} not allowed.</div>` +
    `<div class="v-detail v-muted">Allowed: ${chips(options)}</div>`;
}

// "temperature (expected float, got str value 'x')" → param + expected/got.
function formatTypeSegment(seg) {
  const m = seg.match(/^\s*(\S+)\s*\(expected\s*(.+?),\s*got\s*(.+?)\)\s*$/s);
  if (!m) return `<div class="v-detail">${escapeHtml(seg.trim())}</div>`;
  return `<div class="v-detail"><span class="chip">${escapeHtml(m[1])}</span>: ` +
    `expected <span class="v-good">${escapeHtml(m[2])}</span>, ` +
    `got <span class="v-got">${escapeHtml(m[3])}</span></div>`;
}

function chips(items, cls = '') {
  if (!items.length) return '<span class="v-muted">(none)</span>';
  return `<span class="v-chips">` +
    items.map((t) => `<span class="chip ${cls === 'bad' ? 'bad' : ''}">${escapeHtml(t)}</span>`).join('') +
    `</span>`;
}

function splitNames(s) {
  return s.split(',').map((x) => x.trim()).filter(Boolean);
}

// "['A', 'B']" or "'x'" -> values.
function pyTokens(s) {
  const quoted = [...s.matchAll(/'([^']*)'|"([^"]*)"/g)].map((m) => m[1] ?? m[2]);
  if (quoted.length) return quoted;
  return [s.replace(/^[\[\(]|[\]\)]$/g, '').trim()];
}

// PyYAML's error dump -> {message, location, snippet}.
function formatYamlError(raw) {
  const text = String(raw);
  const marks = [...text.matchAll(/line (\d+), column (\d+)/g)];
  const last = marks.length ? marks[marks.length - 1] : null;
  const location = last ? `Line ${last[1]}, column ${last[2]}` : '';

  const lines = text.split('\n');
  let snippet = '';
  for (let i = lines.length - 1; i > 0; i--) {
    if (/^\s*\^\s*$/.test(lines[i])) {
      snippet = `${lines[i - 1].trim()}\n${'^'.padStart(caretCol(lines[i], lines[i - 1]))}`;
      break;
    }
  }

  const problem = (text.match(/expected (.+?), but found (.+?)(?:\n|$)/) || [])[0];
  return { message: friendlyYaml(text, problem), location, snippet };
}

// Re-align the caret to the trimmed snippet line (PyYAML indents both).
function caretCol(caretLine, codeLine) {
  const lead = codeLine.length - codeLine.trimStart().length;
  const col = caretLine.indexOf('^');
  return Math.max(1, col - lead + 1);
}

function friendlyYaml(text, problem) {
  if (/expected <block end>, but found/.test(text))
    return 'Unexpected extra content — a value continues where a list or block should have ended. Look for a stray character, or a missing comma or closing bracket.';
  if (/could not find expected ':'/.test(text))
    return "Missing colon — a key needs a ':' after it before its value.";
  if (/mapping values are not allowed here/.test(text))
    return "Unexpected ':' — check the indentation, or wrap the value in quotes if it genuinely contains a colon.";
  if (/found unexpected end of (stream|document)/.test(text))
    return 'The document ended early — a bracket, brace or quote is probably left unclosed.';
  if (/found character '\\t'|found a tab character/.test(text))
    return 'Tab character in indentation — YAML must be indented with spaces, not tabs.';
  if (/found duplicate key/.test(text))
    return 'Duplicate key — the same key is defined twice in this mapping.';
  if (problem) return problem.replace(/\n.*/s, '').trim();
  return (text.split('\n')[0] || 'Could not parse the YAML.').trim();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function boot() {
  Icon.hydrate();
  STATE.registry = await API.registry();
  for (const s of STATE.registry.steps) STATE.stepsByName[s.name] = s;
  for (const q of STATE.registry.qc) STATE.qcByName[q.name] = q;
  STATE.onChange = refreshYAML;

  // Themes are defined in style.css, not loaded from a CDN.
  let savedTheme = 'vscode-dark';
  try { savedTheme = localStorage.getItem('yamlTheme') || savedTheme; } catch (e) { /* private mode */ }
  editor = CodeMirror.fromTextArea(document.getElementById('yaml-editor'), {
    mode: 'yaml',
    theme: savedTheme,
    lineNumbers: true,
    lineWrapping: true,
  });

  const themeSel = document.getElementById('theme-select');
  themeSel.value = savedTheme;
  themeSel.addEventListener('change', () => {
    editor.setOption('theme', themeSel.value);
    try { localStorage.setItem('yamlTheme', themeSel.value); } catch (e) { /* ignore */ }
  });

  editor.on('change', () => {
    if (syncingFromBuilder) return;
    Config.noteEdit();
    Config.updateStatus();
    Run.syncToForm();
    scheduleYamlToBuilder();
    scheduleValidate();
    Inspect.schedule();
  });

  // While editing the YAML, the builder follows the cursor rather than the other way round.
  editor.on('focus', () => { activeStepId = null; activeFieldPath = null; clearStepHighlight(); });
  editor.on('cursorActivity', () => {
    if (syncingFromBuilder) return;
    const line = editor.getCursor().line;
    const idx = stepIndexAtLine(line);
    const path = yamlPathAtLine(line);
    const key = idx + '|' + path.join('/');
    if (key === lastFocusedKey) return;
    lastFocusedKey = key;
    if (idx != null && idx !== lastFocusedStep) focusStepInBuilder(idx);
    lastFocusedStep = idx;
    focusFieldInBuilder(idx, path);
  });

  showValidating();
  renderSettings();
  renderPipeline();
  initBuilderDnD();
  Config.loading = true;
  refreshYAML();
  Config.loading = false;
  await Config.refreshList();
  Demos.resume();
  await Defaults.load();

  // After a refresh mid-run, open the running config so pause markers index into the right steps.
  try {
    let opening = Config.known.includes('default.yaml') ? 'default.yaml' : null;
    const running = await API.runStatus().then((s) => s.running).catch(() => false);
    if (running && Config.known.includes('_last_run.yaml')) opening = '_last_run.yaml';
    if (opening) await Config.load(opening);
  } catch (e) { /* no default; start empty */ }

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      Run.showTab(tab.dataset.tab);
      if (tab.dataset.tab === 'yaml') setTimeout(() => editor.refresh(), 0);
    });
  });

  document.getElementById('btn-copy').addEventListener('click', () => {
    navigator.clipboard.writeText(editor.getValue());
  });

  const picker = document.getElementById('config-select');
  picker.querySelector('.cfg-trigger').addEventListener('click', () => {
    if (picker.classList.contains('open')) Config.closePicker();
    else Config.openPicker();
  });
  document.addEventListener('click', (e) => {
    if (!picker.contains(e.target)) Config.closePicker();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') Config.closePicker();
  });

  document.getElementById('btn-reveal').addEventListener('click', async () => {
    try {
      const { path } = await API.revealConfigs();
      Config.notice('Opened ' + path);
    } catch (e) {
      Config.notice('Could not open the configs folder: ' + e.message);
    }
    await Config.refreshList();
  });
  document.getElementById('config-name').addEventListener('input', () => Config.updateSaveLabel());
  document.getElementById('btn-save').addEventListener('click', async () => {
    const input = document.getElementById('config-name');
    let name = input.value.trim();
    if (!name) { alert('Enter a config name to save.'); return; }
    if (Config.isLocked(name)) {
      name = Config.nextCustomName();
      input.value = name;
      Config.notice(`That name is locked — saved as ${name}.yaml instead.`);
    }
    try {
      await API.saveConfig(name, editor.getValue());
    } catch (e) {
      alert(e.message);
      return;
    }
    Config.savedText = editor.getValue();
    Config.builtFor = null;
    await Config.refreshList(Config.withExt(name));
    Config.setCurrent(name);
  });
  document.getElementById('btn-delete').addEventListener('click', async () => {
    const name = Config.selected;
    if (!name || Config.isLocked(name)) return;
    if (!confirm('Delete ' + name + '?')) return;
    try {
      await API.deleteConfig(name);
    } catch (e) {
      alert(e.message);
      return;
    }
    if (Config.current === name) Config.current = null;
    await Config.refreshList();
  });

  Run.initScroll();
  Outputs.init();
  // Doubles as Continue while paused — see Run.setRunButton().
  document.getElementById('btn-run').addEventListener('click', () => {
    if (Build.active) return;
    if (Run.pausedStep === null) Run.start(editor.getValue());
    else if (ManualQC.isActive()) ManualQC.applyAndContinue();
    else Run.continueRun();
  });
  document.getElementById('btn-stop').addEventListener('click', () => Run.stop());
  document.getElementById('btn-clear').addEventListener('click', () => Run.clearRun());
  document.getElementById('btn-rerun').addEventListener('click', () =>
    ManualQC.isActive() ? ManualQC.apply() : Run.rerunStep());

  // Reattach after a refresh mid-run instead of offering a Run that would 409.
  Run.resumeIfRunning();

  // Sleep or a background tab kills the SSE stream while the run continues.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) return;
    Run.ensureConnected();
    checkServer();
  });
  window.addEventListener('online', () => Run.ensureConnected());
  setInterval(checkServer, 5000);
}

// The page keeps working once the server is gone, so say so rather than failing silently.
async function checkServer() {
  let up = true;
  try {
    await fetch('/api/run/status');
  } catch (e) {
    up = false;
  }
  document.getElementById('server-down').classList.toggle('hidden', up);
}

boot().catch((e) => {
  document.body.innerHTML =
    `<pre style="padding:20px;color:#dc2626">Failed to start dashboard:\n${escapeHtml(e.stack || e)}</pre>`;
});
