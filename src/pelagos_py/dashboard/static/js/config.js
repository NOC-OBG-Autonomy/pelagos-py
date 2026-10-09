// Builder STATE <-> pipeline YAML. The builder is the source of truth.

const Config = {
  toObject() {
    const cfg = {};

    const pipeline = {};
    for (const spec of STATE.registry.pipeline_fields) {
      const val = STATE.pipeline.settings[spec.name];
      if (val !== null && val !== undefined && val !== '') pipeline[spec.name] = val;
    }
    if (Object.keys(pipeline).length) cfg.pipeline = pipeline;

    cfg.steps = STATE.pipeline.items.map((item) => {
      const params = {};
      for (const spec of item.def.parameters || []) {
        const val = item.values[spec.name];
        if (spec.name === 'qc_settings' && isQcContainer(item.def)) {
          if (val && Object.keys(val).length) params[spec.name] = val;
          continue;
        }
        const required = spec.required;
        const changed = !('default' in spec) || !Forms.equal(val, spec.default);
        const empty = val === null || val === undefined || val === '';
        if (required || (changed && !empty)) params[spec.name] = val;
      }
      // Params the builder can't edit (e.g. qc_handling_settings) must still survive the round trip.
      for (const [k, v] of Object.entries(item.extras || {})) params[k] = v;
      const step = { name: item.name };
      if (Object.keys(params).length) step.parameters = params;
      step.diagnostics = item.diagnostics;
      return step;
    });

    return cfg;
  },

  // Forms.dump keeps lists inline and integer keys (QC flags) unquoted, like the hand-written configs.
  toYAML() {
    const cfg = Config.toObject();
    const steps = cfg.steps || [];
    let out = '';
    if (cfg.pipeline) out += Forms.dump({ pipeline: cfg.pipeline }) + '\n';
    if (!steps.length) return out + 'steps: []\n';

    const emit = (obj) => Forms.dump([obj]).replace(/^(?=.)/gm, '  ');
    out += 'steps:\n';
    let i = 0;
    for (const node of STATE.pipeline.nodes) {
      if (isSection(node)) {
        out += Config.banner(node.title);
        for (let k = 0; k < node.steps.length; k++) out += emit(steps[i++]);
      } else {
        out += emit(steps[i++]);
      }
    }
    return out;
  },

  banner(title) {
    const bar = '# ' + '='.repeat(59);
    const text = String(title == null ? '' : title).trim();
    const pad = Math.max(0, Math.floor((59 - text.length) / 2));
    return `\n${bar}\n# ${' '.repeat(pad)}${text}\n${bar}\n`;
  },

  // Comments are lost in the object round trip, so banners are read from the text.
  sectionsFromYAML(text) {
    const lines = String(text).split('\n');
    const isBar = (l) => /^\s*#\s*[=~-]{5,}\s*$/.test(l || '');
    const oneLine = /^\s*#\s*[-=]{4,}\s*(\S.*?)\s*[-=]{4,}\s*$/;
    const found = [];
    let inSteps = false, stepsIndent = null, count = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!inSteps) { if (/^steps:\s*$/.test(line)) inSteps = true; continue; }
      const m = line.match(/^(\s*)-\s/);
      if (m) {
        if (stepsIndent === null) stepsIndent = m[1].length;
        if (m[1].length === stepsIndent) count++;
        continue;
      }
      const one = line.match(oneLine);
      if (one && /[A-Za-z0-9]/.test(one[1])) { found.push({ title: one[1], index: count }); continue; }
      if (isBar(line) && !isBar(lines[i + 1]) && isBar(lines[i + 2])) {
        const t = (lines[i + 1] || '').match(/^\s*#\s*(\S.*?)\s*$/);
        if (t) { found.push({ title: t[1], index: count }); i += 2; }
      }
    }
    return found;
  },

  fromYAML(text) {
    Config.fromObject(jsyaml.load(text) || {}, Config.sectionsFromYAML(text));
  },

  stepName(name) {
    if (STATE.stepsByName[name]) return name;
    const low = String(name).toLowerCase();
    return Object.keys(STATE.stepsByName).find((n) => n.toLowerCase() === low);
  },

  itemFromStep(s) {
    if (!s || !s.name) return null;
    const canonical = Config.stepName(s.name);
    const def = STATE.stepsByName[canonical];
    if (!def) return null;
    const item = {
      id: ++STATE._seq,
      name: canonical,
      def,
      values: initValues(def),
      diagnostics: (s.diagnostics === 'all' || Array.isArray(s.diagnostics)) ? s.diagnostics : !!s.diagnostics,
      collapsed: true,
    };
    const params = s.parameters || {};
    const described = new Set((def.parameters || []).map((p) => p.name));
    item.extras = {};
    for (const [k, v] of Object.entries(params)) {
      if (described.has(k)) item.values[k] = Forms.clone(v);
      else item.extras[k] = Forms.clone(v);
    }
    if (isQcContainer(def) && !item.values.qc_settings) item.values.qc_settings = {};
    return item;
  },

  fromObject(cfg, sections = []) {
    STATE.pipeline = { settings: {}, nodes: [], items: [] };

    const pipeline = (cfg && cfg.pipeline) || {};
    for (const spec of STATE.registry.pipeline_fields) {
      STATE.pipeline.settings[spec.name] =
        spec.name in pipeline ? pipeline[spec.name] : Forms.defaultValue(spec);
    }

    const steps = (cfg && cfg.steps) || [];

    // Steps before the first banner stay at the top level.
    const nodes = STATE.pipeline.nodes;
    let si = 0;
    const openSectionsAt = (k) => {
      while (si < sections.length && sections[si].index <= k) {
        nodes.push(makeSection(sections[si].title));
        si++;
      }
    };

    steps.forEach((s, idx) => {
      openSectionsAt(idx);
      const item = Config.itemFromStep(s);
      if (!item) return; // surfaced by Validate
      const last = nodes[nodes.length - 1];
      if (isSection(last)) last.steps.push(item);
      else nodes.push(item);
    });
    openSectionsAt(steps.length);

    renderSettings();
    renderPipeline();
  },

  // Locked configs (default.yaml, demos) save as a new custom_run_N; the server enforces this too.
  known: [],
  locked: [],
  demo: [],
  missions: {},
  labels: {},       // glider names repeat across missions
  gliders: {},
  modes: {},        // 'nrt' | 'delayed' per demo config
  reference: [],
  downloaded: [],
  sizes: {},
  current: null,
  selected: '',
  loading: false,   // so loading isn't seen as an edit
  picks: 0,         // a finished download only opens if nothing else was picked since

  isLocked(name) {
    return !!name && Config.locked.includes(Config.withExt(name));
  },

  withExt(name) {
    return /\.ya?ml$/.test(name) ? name : name + '.yaml';
  },

  nextCustomName() {
    let n = 0;
    for (const name of Config.known) {
      const m = name.match(/^custom_run_(\d+)\.ya?ml$/);
      if (m) n = Math.max(n, Number(m[1]));
    }
    return `custom_run_${n + 1}`;
  },

  builtFor: null,   // data file the pipeline was generated for, until saved
  savedText: '',
  updateStatus() {
    const el = document.getElementById('config-status');
    if (!el || !editor) return;
    const onDisk = !!Config.current && Config.known.includes(Config.current);
    const edited = editor.getValue() !== Config.savedText;
    let text;
    if (Config.builtFor) text = `Built automatically for ${Config.builtFor} · not saved`;
    else if (onDisk && Config.isLocked(Config.current)) text = 'Reference pipeline · edits save as a copy';
    else if (!onDisk) text = 'New pipeline · not saved';
    else text = edited ? 'Edited · not saved' : 'Saved';
    el.textContent = text;
  },

  // Manual QC boxes belong to one data file, so offer to clear them when it changes.
  dataFile: null,
  checkDataFile() {
    const load = STATE.pipeline.items.find((i) => (i.def.parameters || []).some((p) => p.name === 'file_path'));
    const path = (load && load.values.file_path) || '';
    const previous = Config.dataFile;
    Config.dataFile = path;
    if (Config.loading || !previous || path === previous) return;
    const manual = STATE.pipeline.items
      .filter((i) => isQcContainer(i.def))
      .map((i) => (i.values.qc_settings || {})['manual qc'])
      .filter((t) => t && (t.boxes || []).length);
    if (!manual.length) return;
    Config.notice('Manual QC boxes were drawn on the previous data file.', { sticky: true, action: {
      label: 'Clear boxes',
      onClick: () => {
        for (const t of manual) t.boxes = [];
        Config.notice('');
        Defaults.commit();
      },
    } });
  },

  setCurrent(name) {
    Config.current = name ? Config.withExt(name) : null;
    const nameInput = document.getElementById('config-name');
    if (nameInput && name) nameInput.value = name.replace(/\.ya?ml$/, '');
    Config.selected = Config.current || '';
    Config.renderPicker();
    Config.updateControls();
    Config.updateSaveLabel();
    Config.updateStatus();
    Demos.render();
  },

  // Save writes to the name in the field, so a new name makes a new config.
  updateSaveLabel() {
    const btn = document.getElementById('btn-save');
    const nameInput = document.getElementById('config-name');
    if (!btn || !nameInput) return;
    const typed = nameInput.value.trim();
    const label = btn.querySelector('.btn-label') || btn;
    const overwriting = !typed || Config.withExt(typed) === Config.current;
    label.textContent = overwriting ? 'Save' : 'Save as new';
    btn.title = !typed
      ? 'Enter a name to save this pipeline'
      : overwriting
      ? `Save changes to ${Config.current}`
      : `Save as a new pipeline: ${Config.withExt(typed)}`;
  },

  updateControls() {
    const del = document.getElementById('btn-delete');
    if (!del) return;
    const chosen = Config.selected;
    del.disabled = !chosen || Config.isLocked(chosen) || RunLock.running;
    if (RunLock.running) del.title = 'Locked while the pipeline is running';
    else if (Config.isLocked(chosen)) del.title = `${chosen} is a locked reference config and cannot be deleted`;
    else del.title = 'Delete the selected config';
  },

  apply(text) {
    showValidating(true);
    Config.loading = true;
    // fromYAML already synced; a second sync from the change event would lose a paused step's state.
    syncingFromBuilder = true;
    editor.setValue(text);
    syncingFromBuilder = false;
    Config.dataFile = null;
    Config.builtFor = null;
    try { Config.fromYAML(text); refreshYAML(); }
    catch (e) { Config.notice('Loaded as raw YAML — the builder could not read it: ' + e.message); }
    Config.savedText = editor.getValue();
    Config.loading = false;
    Config.updateStatus();
  },

  async load(name) {
    const pick = ++Config.picks;
    if (Config.demo.includes(name) && !Config.downloaded.includes(name)) {
      await Demos.download(name);
      if (pick !== Config.picks || !Config.downloaded.includes(name)) return;
    }
    const { yaml_content, build } = await API.loadConfig(name);
    if (pick !== Config.picks) return; // a later pick has taken over
    Config.notice('');
    if (build) {
      await Build.start({
        name, filePath: build.file_path, description: build.description,
      });
    } else {
      Config.apply(yaml_content);
      Config.setCurrent(name);
    }
  },

  // "demo_nelson.yaml" -> "Nelson"; prefers the server label since glider names repeat.
  demoLabel(name) {
    return Config.labels[name] || name.replace(/^demo_/, '').replace(/\.ya?ml$/, '')
      .replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  },

  renderPicker() {
    const root = document.getElementById('config-select');
    if (!root) return;
    const menu = root.querySelector('.cfg-menu');
    menu.innerHTML = '';
    if (!Config.known.length) {
      const li = document.createElement('li');
      li.className = 'cfg-opt';
      li.textContent = 'no saved pipelines';
      menu.appendChild(li);
      return;
    }

    const makeOpt = (name, { demo = false, hint, label, title, unsaved = false } = {}) => {
      const locked = Config.locked.includes(name);
      const needsDownload = demo && !Config.downloaded.includes(name);
      const li = document.createElement('li');
      li.className = 'cfg-opt' + (locked ? ' ref' : '') + (demo ? ' demo' : '') +
        (needsDownload ? ' needs-download' : '') + (name === Config.selected || unsaved ? ' selected' : '');
      li.setAttribute('role', 'option');
      if (title) li.title = title;
      if (needsDownload) {
        const dl = document.createElement('span');
        dl.className = 'cfg-opt-download';
        dl.title = 'Not downloaded yet — picking this fetches its data file first';
        dl.appendChild(Icon.el('download', 12));
        li.appendChild(dl);
      }
      const text = document.createElement('span');
      text.className = 'cfg-opt-text';
      text.textContent = label ?? (demo ? Config.demoLabel(name) : name);
      li.appendChild(text);
      const edited = name === Config.current && editor && editor.getValue() !== Config.savedText;
      const hintText = hint ?? (needsDownload ? 'download' : edited ? 'edited · not saved' : '');
      if (hintText) {
        const h = document.createElement('span');
        h.className = 'cfg-hint';
        h.textContent = hintText;
        li.appendChild(h);
      }
      li.addEventListener('click', async () => {
        Config.closePicker();
        if (unsaved) return;
        try { await Config.load(name); }
        catch (e) {
          Config.notice('Could not load ' + name + ': ' + e.message, { sticky: true, err: true });
        }
      });
      return li;
    };

    const group = (title) => {
      const h = document.createElement('li');
      h.className = 'cfg-group';
      h.textContent = title;
      menu.appendChild(h);
    };

    const LAST_RUN = '_last_run.yaml';
    const referenceNames = Config.known.filter((n) => Config.reference.includes(n));
    const otherNames = Config.known.filter(
      (n) => !Config.demo.includes(n) && !Config.reference.includes(n) && n !== LAST_RUN
    );
    const unsaved = Config.current && !Config.known.includes(Config.current);

    // Demo configs live on the Files tab (demos.js), not in this menu.
    if (referenceNames.length) {
      group('Template');
      for (const name of referenceNames) menu.appendChild(makeOpt(name));
    }
    if (otherNames.length || unsaved) {
      group('Your pipelines');
      if (unsaved) menu.appendChild(makeOpt(Config.current, { hint: 'not saved', unsaved: true }));
      for (const name of otherNames) menu.appendChild(makeOpt(name));
    }
    if (Config.known.includes(LAST_RUN)) {
      group('Automatic');
      menu.appendChild(makeOpt(LAST_RUN, { label: 'Last run', hint: 'copy of what last ran',
        title: 'The exact pipeline of your most recent run, saved automatically each time you press Run' }));
    }
  },

  // Re-read on open so files changed outside the dashboard show up.
  async openPicker() {
    const root = document.getElementById('config-select');
    await Config.refreshList(Config.selected);
    root.classList.add('open');
    root.querySelector('.cfg-menu').classList.remove('hidden');
    root.querySelector('.cfg-trigger').setAttribute('aria-expanded', 'true');
  },

  closePicker() {
    const root = document.getElementById('config-select');
    if (!root) return;
    root.classList.remove('open');
    root.querySelector('.cfg-menu').classList.add('hidden');
    root.querySelector('.cfg-trigger').setAttribute('aria-expanded', 'false');
  },

  noteEdit() {
    if (Config.loading || !Config.isLocked(Config.current)) return;
    const forked = Config.nextCustomName();
    const from = Config.current;
    Config.current = Config.withExt(forked);
    document.getElementById('config-name').value = forked;
    Config.selected = '';
    Config.renderPicker();
    Config.updateControls();
    Config.updateSaveLabel();
    Config.notice('Editing a copy — save it in the', { action: {
      label: 'YAML tab', onClick: () => document.querySelector('.tab[data-tab="yaml"]').click() } });
  },

  // sticky: stays until the next notice(), so errors and downloads aren't missed.
  notice(text, { sticky = false, err = false, action = null } = {}) {
    const el = document.getElementById('config-note');
    if (!el) return;
    el.textContent = text;
    el.title = text;
    if (action) {
      el.appendChild(Forms.button(action.label, { onclick: action.onClick }));
      el.title = text + ' ' + action.label;
    }
    el.classList.toggle('hidden', !text);
    el.classList.toggle('err', !!err);
    clearTimeout(Config._noticeTimer);
    if (text && !sticky) Config._noticeTimer = setTimeout(() => Config.notice(''), 8000);
  },

  async refreshList(selected) {
    const info = await API.listConfigs();
    Config.known = info.configs || [];
    Config.locked = info.protected || [];
    Config.demo = info.demo || [];
    Config.missions = info.missions || {};
    Config.labels = info.labels || {};
    Config.gliders = info.gliders || {};
    Config.modes = info.modes || {};
    Config.reference = info.reference || [];
    Config.downloaded = info.downloaded || [];
    Config.sizes = info.sizes || {};
    if (selected !== undefined) Config.selected = selected || '';
    if (Config.selected && !Config.known.includes(Config.selected)) Config.selected = '';
    Config.renderPicker();
    Config.updateControls();
    Config.updateStatus();
    Demos.render();
  },
};
