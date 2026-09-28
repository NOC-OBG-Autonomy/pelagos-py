// What differs from the shipped default pipeline (default.yaml), step by step
// and QC test by QC test, so the builder can flag edits, extra steps and
// removed steps, and offer to put the default back. Diagnostics switches, the
// manual QC test and the per-file paths never count as edits.
const Defaults = {
  steps: null, // default.yaml's steps, or null when it isn't available
  sectionOf: [], // per default step, the banner title it sits under (or null)
  filePath: null, // input file the decisions below were fetched for
  decisions: {},  // build decisions with a choice, keyed by the section they can drop
  IGNORE: new Set(['file_path', 'output_path']),
  IGNORE_TESTS: new Set(['manual qc']),

  async load() {
    try {
      const { yaml_content } = await API.loadConfig('default.yaml');
      const steps = (jsyaml.load(yaml_content) || {}).steps || [];
      const banners = Config.sectionsFromYAML(yaml_content);
      Defaults.steps = []; Defaults.sectionOf = [];
      steps.forEach((s, i) => {
        if (!s || !s.name) return;
        const b = banners.filter((x) => x.index <= i).pop();
        Defaults.steps.push(s);
        Defaults.sectionOf.push(b ? b.title : null);
      });
    } catch (e) { Defaults.steps = null; }
  },

  // Every described parameter (spec default when unset) plus undescribed
  // extras. A cleared field ('') counts as unset, as it does in the YAML.
  canonical(specs, params, ignore) {
    const out = {};
    const described = new Set();
    const norm = (v) => (v === '' || v === undefined ? null : v);
    for (const spec of specs) {
      described.add(spec.name);
      if (ignore.has(spec.name) || spec.name === 'qc_settings') continue;
      out[spec.name] = norm(spec.name in params ? params[spec.name] : Forms.defaultValue(spec));
    }
    for (const [k, v] of Object.entries(params)) {
      if (!described.has(k) && !ignore.has(k) && k !== 'qc_settings' && k !== 'diagnostics') out[k] = norm(v);
    }
    return out;
  },

  diffParams(specs, cur, base, ignore) {
    const a = Defaults.canonical(specs, cur, ignore);
    const b = Defaults.canonical(specs, base, ignore);
    const out = [];
    for (const name of new Set([...Object.keys(a), ...Object.keys(b)])) {
      if (!Forms.equal(a[name], b[name])) out.push({ name, current: a[name], base: b[name] });
    }
    return out;
  },

  diffQc(cur, base) {
    const qc = { changed: {}, added: [], removed: [] };
    for (const t of Object.keys(cur)) {
      if (Defaults.IGNORE_TESTS.has(t)) continue;
      if (!(t in base)) { qc.added.push(t); continue; }
      const specs = (STATE.qcByName[t] || {}).parameters || [];
      const d = Defaults.diffParams(specs, cur[t], base[t], new Set(['diagnostics']));
      if (d.length) qc.changed[t] = d;
    }
    for (const t of Object.keys(base)) {
      if (!Defaults.IGNORE_TESTS.has(t) && !(t in cur)) qc.removed.push(t);
    }
    return qc;
  },

  diffStep(item, base) {
    const params = Defaults.diffParams(
      item.def.parameters || [], Object.assign({}, item.values, item.extras || {}),
      base.parameters || {}, Defaults.IGNORE);
    let qc = null, count = params.length;
    if (isQcContainer(item.def)) {
      qc = Defaults.diffQc(item.values.qc_settings || {}, (base.parameters || {}).qc_settings || {});
      count += Object.keys(qc.changed).length + qc.added.length + qc.removed.length;
    }
    return { params, qc, count };
  },

  // Align the pipeline with default.yaml (longest common subsequence on step
  // names, preferring the least-edited pairing when a name repeats), then
  // classify every step: edited / extra / unchanged, and place each missing
  // default step: inside its section when the pipeline still has that section
  // (`sectionFirst`/`after`), else grouped into a missing section
  // (`sectionsFirst`/`sectionsAfter`, keyed by the root node it follows).
  compute() {
    const byId = new Map(), after = new Map(), first = [];
    const sectionFirst = new Map(), sectionsAfter = new Map(), sectionsFirst = [];
    const view = { byId, first, after, sectionFirst, sectionsAfter, sectionsFirst };
    const items = STATE.pipeline.items, base = Defaults.steps;
    if (!base || !items.length) return view;
    Defaults.syncDecisions(items);
    const baseNames = base.map((s) => Config.stepName(s.name));
    const n = items.length, m = base.length;
    const score = Array.from({ length: n + 1 }, () => new Float64Array(m + 1));
    const diffs = new Map();
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        let best = Math.max(score[i - 1][j], score[i][j - 1]);
        if (items[i - 1].name === baseNames[j - 1]) {
          const d = Defaults.diffStep(items[i - 1], base[j - 1]);
          diffs.set(i + ',' + j, d);
          best = Math.max(best, score[i - 1][j - 1] + 1 + 1 / (1 + d.count));
        }
        score[i][j] = best;
      }
    }
    const matched = new Array(m).fill(null);
    for (let i = n, j = m; i > 0 && j > 0;) {
      const d = diffs.get(i + ',' + j);
      if (d && score[i][j] === score[i - 1][j - 1] + 1 + 1 / (1 + d.count)) {
        byId.set(items[i - 1].id, { diff: d, base: base[j - 1] });
        matched[j - 1] = items[i - 1].id;
        i--; j--;
      } else if (score[i - 1][j] >= score[i][j - 1]) i--;
      else j--;
    }
    for (const item of items) if (!byId.has(item.id)) byId.set(item.id, { extra: true });

    const norm = (t) => String(t || '').trim().toLowerCase();
    const sections = STATE.pipeline.nodes.filter(isSection);
    const push = (map, key, v) => { if (!map.has(key)) map.set(key, []); map.get(key).push(v); };
    // The root node (loose step or section) holding a matched item.
    const rootOf = (id) => { const loc = locateStep(id); return loc.section ? loc.section.id : id; };
    let anchor = null, ghostSec = null;
    for (let j = 0; j < m; j++) {
      if (matched[j] != null) { anchor = matched[j]; ghostSec = null; continue; }
      const title = Defaults.sectionOf[j];
      const sec = title == null ? null : sections.find((s) => norm(s.title) === norm(title));
      if (sec) {
        const inSec = anchor != null && sec.steps.some((s) => s.id === anchor);
        if (inSec) push(after, anchor, base[j]); else push(sectionFirst, sec.id, base[j]);
      } else if (title != null) {
        if (!ghostSec || ghostSec.title !== title) {
          ghostSec = { title, steps: [], anchor: anchor == null ? null : rootOf(anchor) };
          if (ghostSec.anchor == null) sectionsFirst.push(ghostSec); else push(sectionsAfter, ghostSec.anchor, ghostSec);
        }
        ghostSec.steps.push(base[j]);
      } else if (anchor == null) first.push(base[j]);
      else push(after, anchor, base[j]);
    }
    return view;
  },

  // Re-fetch the build decisions for the Load step's file when it changes, so a
  // missing section can offer the choice that would bring it back.
  syncDecisions(items) {
    const load = items.find((i) => (i.def.parameters || []).some((p) => p.name === 'file_path'));
    const path = (load && load.values.file_path) || null;
    if (path === Defaults.filePath) return;
    Defaults.filePath = path;
    Defaults.decisions = {};
    if (!path) return;
    API.buildDecisions(path).then(({ decisions }) => {
      if (path !== Defaults.filePath) return;
      for (const d of decisions) if (d.section && d.options.length) Defaults.decisions[d.section] = d;
      renderPipeline();
    }).catch(() => {});
  },

  // Rebuild the template for the current file with one decision changed and
  // bring back the steps of `ghost`'s section from it (plus what Prepare OG1
  // must rename for them). Other sections are left as they are.
  async applyChoice(ghost, d, key) {
    const { yaml_content } = await API.build(Defaults.filePath, { [d.id]: key });
    const cfg = jsyaml.load(yaml_content) || {};
    const banners = Config.sectionsFromYAML(yaml_content);
    const norm = (t) => String(t || '').trim().toLowerCase();
    const at = banners.findIndex((b) => norm(b.title) === norm(ghost.title));
    const steps = (cfg.steps || []).slice(
      at < 0 ? 0 : banners[at].index, at < 0 || !banners[at + 1] ? undefined : banners[at + 1].index);
    const built = (cfg.steps || []).find((s) => s && s.name === 'Prepare OG1');
    const prep = STATE.pipeline.items.find((i) => i.name === 'Prepare OG1');
    if (built && prep) {
      for (const k of ['renames', 'bbp700_is_beta']) {
        if (k in (built.parameters || {})) prep.values[k] = Forms.clone(built.parameters[k]);
      }
    }
    Defaults.addSectionBack(Object.assign({}, ghost, { steps }));
  },

  fmt(v) {
    if (v === null || v === undefined || v === '') return 'unset';
    const s = typeof v === 'object' ? Forms.dump(v).trim().replace(/\s*\n\s*/g, ' ') : String(v);
    return s.length > 48 ? s.slice(0, 47) + '…' : s;
  },

  // "Changed from the default (x) — Restore default" under a field, or any
  // other one-line warning with a link-styled action.
  note(text, action, onAction, cls = '') {
    const el = document.createElement('div');
    el.className = 'default-note ' + cls;
    el.appendChild(document.createTextNode(text + ' '));
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = action;
    btn.onclick = (e) => { e.stopPropagation(); onAction(); };
    el.appendChild(btn);
    return el;
  },

  changedNote(base, onRestore) {
    return Defaults.note(`Changed from the default (${Defaults.fmt(base)}) —`, 'Restore default', onRestore);
  },

  badge(text, cls = '') {
    return Forms.el('span', { class: `tag ${cls === 'extra' ? 'accent' : 'warn'} default-badge`, textContent: text });
  },

  // ---- restore actions (each re-renders and syncs the YAML) ----
  commit() { renderPipeline(); STATE.onChange(); },

  restoreParam(values, name, base) {
    values[name] = Forms.clone(base);
    Defaults.commit();
  },

  restoreStep(item, base) {
    const params = base.parameters || {};
    const described = new Set();
    for (const spec of item.def.parameters || []) {
      described.add(spec.name);
      if (Defaults.IGNORE.has(spec.name)) continue;
      if (spec.name === 'qc_settings') {
        item.values.qc_settings = Defaults.restoredQc(item.values.qc_settings || {}, params.qc_settings || {});
        continue;
      }
      item.values[spec.name] = spec.name in params ? Forms.clone(params[spec.name]) : Forms.defaultValue(spec);
    }
    item.extras = {};
    for (const [k, v] of Object.entries(params)) if (!described.has(k)) item.extras[k] = Forms.clone(v);
    Defaults.commit();
  },

  // The default's tests in its order, keeping each test's current diagnostics
  // override and any manual QC the user has set up.
  restoredQc(cur, base) {
    const out = {};
    for (const t of Object.keys(base)) {
      if (Defaults.IGNORE_TESTS.has(t)) continue;
      out[t] = Defaults.testFrom(base[t], cur[t]);
    }
    for (const t of Object.keys(cur)) if (Defaults.IGNORE_TESTS.has(t)) out[t] = cur[t];
    return out;
  },

  testFrom(baseTest, curTest) {
    const v = Forms.clone(baseTest) || {};
    delete v.diagnostics;
    if (curTest && 'diagnostics' in curTest) v.diagnostics = curTest.diagnostics;
    return v;
  },

  restoreTest(item, t, baseQc) {
    item.values.qc_settings[t] = Defaults.testFrom(baseQc[t], item.values.qc_settings[t]);
    Defaults.commit();
  },

  addTestBack(item, t, baseQc) {
    const qc = item.values.qc_settings;
    qc[t] = Defaults.testFrom(baseQc[t]);
    const ordered = {};
    for (const k of Object.keys(baseQc)) if (k in qc) ordered[k] = qc[k];
    for (const k of Object.keys(qc)) if (!(k in ordered)) ordered[k] = qc[k];
    item.values.qc_settings = ordered;
    Defaults.commit();
  },

  // Put a missing default step back: after the step it followed, at the start
  // of its section (`sectionId`), or at the very top.
  addStepBack(base, anchorId, sectionId) {
    const item = Config.itemFromStep(base);
    if (!item) return;
    const loc = anchorId == null ? null : locateStep(anchorId);
    const sec = sectionId == null ? null : STATE.pipeline.nodes.find((n) => isSection(n) && n.id === sectionId);
    if (loc) loc.list.splice(loc.index + 1, 0, item);
    else if (sec) sec.steps.unshift(item);
    else STATE.pipeline.nodes.unshift(item);
    Defaults.commit();
  },

  // Put a whole missing default section back after the root node it followed.
  addSectionBack(ghost) {
    const sec = makeSection(ghost.title);
    sec.steps = ghost.steps.map(Config.itemFromStep).filter(Boolean);
    const nodes = STATE.pipeline.nodes;
    const at = ghost.anchor == null ? -1 : nodes.findIndex((n) => n.id === ghost.anchor);
    nodes.splice(at + 1, 0, sec);
    Defaults.commit();
  },

  // Dashed placeholder row: something in default.yaml that this pipeline lacks.
  // `control` is the way back: an Add back button or a decision select.
  ghostRow(cls, name, why, detail, control) {
    const row = document.createElement('div');
    row.className = 'ghost-step ' + cls;
    const n = document.createElement('span');
    n.className = 'step-name';
    n.textContent = name;
    row.appendChild(n);
    const text = document.createElement('span');
    text.className = 'ghost-text';
    text.textContent = why + (detail ? ' · ' + detail : '');
    text.title = text.textContent;
    row.appendChild(text);
    row.appendChild(control);
    return row;
  },

  addButton(label, onAdd) {
    return Forms.button(label, { icon: 'plus', onclick: onAdd });
  },

  ghostStep(base, anchorId, sectionId) {
    const tests = Object.keys(((base.parameters || {}).qc_settings) || {});
    return Defaults.ghostRow('', base.name, 'removed', tests.join(', '),
      Defaults.addButton('Add back', () => Defaults.addStepBack(base, anchorId, sectionId)));
  },

  // A section the builder skipped for this file offers the build decision's
  // choices; any other missing section just comes back as in default.yaml.
  ghostSection(ghost) {
    const d = Defaults.decisions[ghost.title];
    const names = ghost.steps.map((s) => s.name).join(', ');
    if (!d) {
      return Defaults.ghostRow('ghost-section', ghost.title, 'removed', names,
        Defaults.addButton('Add back container', () => Defaults.addSectionBack(ghost)));
    }
    const sel = Forms.select(d.options.map((o) => [o.key, o.label]), null, null, { placeholder: 'Choose…' });
    sel.onchange = () => {
      sel.disabled = true;
      Defaults.applyChoice(ghost, d, sel.value).catch((e) => {
        sel.disabled = false;
        Config.notice(`Could not rebuild ${ghost.title}: ${e.message}`, { sticky: true, err: true });
      });
    };
    return Defaults.ghostRow('ghost-section', ghost.title, 'removed: variable missing from the file', d.title, sel);
  },

  ghostTest(item, t, baseQc) {
    return Defaults.ghostRow('ghost-test', t, 'removed', '',
      Defaults.addButton('Add back', () => Defaults.addTestBack(item, t, baseQc)));
  },
};
