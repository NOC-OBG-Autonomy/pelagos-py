// Turn the builder STATE into pipeline YAML and back. The builder is the source
// of truth; the YAML pane is a live preview whose hand edits sync back automatically.

const Config = {
  // Build the plain config object (pre-YAML) from STATE.
  toObject() {
    const cfg = {};

    // pipeline block: only include fields the user actually set.
    const pipeline = {};
    for (const spec of STATE.registry.pipeline_fields) {
      const val = STATE.pipeline.settings[spec.name];
      if (val !== null && val !== undefined && val !== '') pipeline[spec.name] = val;
    }
    if (Object.keys(pipeline).length) cfg.pipeline = pipeline;

    // steps
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
        // Skip empty optional values (leave them to the step's own default).
        const empty = val === null || val === undefined || val === '';
        if (required || (changed && !empty)) params[spec.name] = val;
      }
      // Parameters the registry doesn't describe (framework ones such as
      // qc_handling_settings) aren't editable in the builder, but must survive
      // the round trip — dropping them silently changes what the step does.
      for (const [k, v] of Object.entries(item.extras || {})) params[k] = v;
      const step = { name: item.name };
      if (Object.keys(params).length) step.parameters = params;
      step.diagnostics = item.diagnostics;
      return step;
    });

    return cfg;
  },

  // Serialise via Forms.dump so scalar lists stay inline ([20, 45]) and integer
  // mapping keys (Argo QC flags in `variable_ranges`, `flag_mapping`, …) emit
  // unquoted, matching the hand-written config style. Steps are emitted node by
  // node so each section can be preceded by its banner comment.
  toYAML() {
    const cfg = Config.toObject();
    const steps = cfg.steps || [];
    let out = '';
    if (cfg.pipeline) out += Forms.dump({ pipeline: cfg.pipeline }) + '\n';
    if (!steps.length) return out + 'steps: []\n';

    // Indent one serialised step ("- name: X\n  parameters:…") under `steps:`.
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

  // A section header, in the banner-comment style the hand-written configs use.
  banner(title) {
    const bar = '# ' + '='.repeat(59);
    const text = String(title == null ? '' : title).trim();
    const pad = Math.max(0, Math.floor((59 - text.length) / 2));
    return `\n${bar}\n# ${' '.repeat(pad)}${text}\n${bar}\n`;
  },

  // Find section banners in raw YAML text. Comments are lost by the object
  // round-trip, so sections are recovered from the text instead. Each result is
  // {title, index} where index counts the steps that precede the banner.
  // Recognises the three-line `# ====` banner this emits, and the one-line
  // `# ---- TITLE ----` style also used in the example configs.
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

  // Load YAML text into the builder, sections included.
  fromYAML(text) {
    Config.fromObject(jsyaml.load(text) || {}, Config.sectionsFromYAML(text));
  },

  // Registry name for a config's step name (matched case-insensitively).
  stepName(name) {
    if (STATE.stepsByName[name]) return name;
    const low = String(name).toLowerCase();
    return Object.keys(STATE.stepsByName).find((n) => n.toLowerCase() === low);
  },

  // A builder item for one config step entry, or null for an unknown step.
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
      // true/false, 'all', or a list of figure names (steps with a figure registry)
      diagnostics: (s.diagnostics === 'all' || Array.isArray(s.diagnostics)) ? s.diagnostics : !!s.diagnostics,
      collapsed: true,
    };
    const params = s.parameters || {};
    const described = new Set((def.parameters || []).map((p) => p.name));
    item.extras = {};
    for (const [k, v] of Object.entries(params)) {
      if (described.has(k)) item.values[k] = Forms.clone(v);
      else item.extras[k] = Forms.clone(v); // kept verbatim, re-emitted by toObject
    }
    if (isQcContainer(def) && !item.values.qc_settings) item.values.qc_settings = {};
    return item;
  },

  // Best-effort load of a config object back into the builder. `sections` is
  // the banner list from sectionsFromYAML (absent when loading a bare object).
  fromObject(cfg, sections = []) {
    STATE.pipeline = { settings: {}, nodes: [], items: [] };

    const pipeline = (cfg && cfg.pipeline) || {};
    for (const spec of STATE.registry.pipeline_fields) {
      STATE.pipeline.settings[spec.name] =
        spec.name in pipeline ? pipeline[spec.name] : Forms.defaultValue(spec);
    }

    const steps = (cfg && cfg.steps) || [];

    // Banner index -> section, opened as the step at that index is reached. A
    // section stays open until the next banner, so steps before the first
    // banner sit loose at the top level.
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
      if (!item) return; // unknown step: skip (surfaced by Validate)
      const last = nodes[nodes.length - 1];
      if (isSection(last)) last.steps.push(item);
      else nodes.push(item);
    });
    openSectionsAt(steps.length); // trailing banners with no steps under them

    renderSettings();
    renderPipeline();
  },

  // ---- persistence ----
  //
  // The shipped reference config (default.yaml) is read-only. Editing it
  // doesn't overwrite it: the pending save target switches to a fresh
  // custom_run_N, so the reference stays pristine and your work becomes a
  // config of its own. Editing any other config is ordinary — changes belong
  // to that file. The server enforces the same rule, so this is convenience,
  // not the lock itself.
  known: [],        // every config name on the server
  locked: [],       // the protected subset
  demo: [],         // the demo subset (also locked) — shown as their own group
  missions: {},     // demo config names grouped by deployment mission, in display order
  labels: {},       // display label per demo config name (glider names repeat across missions)
  gliders: {},      // glider name per demo config
  modes: {},        // 'nrt' | 'delayed' per demo config
  reference: [],    // non-demo protected configs (default.yaml)
  downloaded: [],   // demo configs whose NetCDF file is already on disk
  sizes: {},        // bytes on disk per downloaded demo config
  current: null,    // name of the loaded config, or null for an unsaved one
  selected: '',     // name shown in the picker ('' once the config is unsaved)
  loading: false,   // true while loading/booting, so that isn't seen as an edit
  picks: 0,         // bumped on every pick, so a finished download only opens if nothing else was picked since

  isLocked(name) {
    return !!name && Config.locked.includes(Config.withExt(name));
  },

  withExt(name) {
    return /\.ya?ml$/.test(name) ? name : name + '.yaml';
  },

  // First unused custom_run_N.yaml.
  nextCustomName() {
    let n = 0;
    for (const name of Config.known) {
      const m = name.match(/^custom_run_(\d+)\.ya?ml$/);
      if (m) n = Math.max(n, Number(m[1]));
    }
    return `custom_run_${n + 1}`;
  },

  // One line under the pipeline controls: where the open pipeline came from and
  // whether it matches what's saved.
  builtFor: null,   // data file the open pipeline was generated for, until it's saved
  savedText: '',    // editor text as last loaded from / saved to disk
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

  // Manual QC boxes are drawn on one file's data: when Load OG1 moves to another
  // file, offer to clear them (only in the open copy; the saved .yaml is untouched).
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

  // Note which config is loaded, and reflect it in the toolbar.
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

  // The Save button always writes to whatever name is in the field — typing a
  // different name creates a new config rather than renaming the current one.
  // Swap its label/title so that's clear without a separate "rename" control.
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

  // Delete is only offered for a config that exists and isn't locked.
  updateControls() {
    const del = document.getElementById('btn-delete');
    if (!del) return;
    const chosen = Config.selected;
    del.disabled = !chosen || Config.isLocked(chosen) || RunLock.running;
    del.title = RunLock.running ? 'Locked while the pipeline is running' : Config.isLocked(chosen)
      ? `${chosen} is a locked reference config and cannot be deleted`
      : 'Delete the selected config';
  },

  // ---- picker ----
  //
  // Choosing a config loads it straight away — there is no separate Load step.
  // Reference configs are drawn in the accent colour rather than tagged.

  // Put `text` into the config editor and the builder, keeping the raw YAML if
  // it doesn't map cleanly onto the registry.
  apply(text) {
    // Replace the previous config's validation result rather than leave it showing.
    showValidating(true);
    Config.loading = true;
    // fromYAML syncs the builder itself; a second, deferred sync from the editor's
    // change event would rebuild every step object and lose e.g. a paused step's state.
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

  // A demo whose data isn't on disk downloads first while the dashboard stays
  // usable; it only opens afterwards if nothing else was picked in the meantime.
  async load(name) {
    const pick = ++Config.picks;
    if (Config.demo.includes(name) && !Config.downloaded.includes(name)) {
      await Demos.download(name);
      if (pick !== Config.picks || !Config.downloaded.includes(name)) return;
    }
    const { yaml_content, build } = await API.loadConfig(name);
    Config.notice('');
    if (build) {
      // Demo: the config is generated for its file once the user confirms.
      await Build.start({
        name, filePath: build.file_path, description: build.description,
      });
    } else {
      Config.apply(yaml_content);
      Config.setCurrent(name);
    }
  },

  // "demo_nelson.yaml" -> "Nelson", from the server-supplied label where
  // available (glider names repeat across missions, e.g. two Churchills, so
  // the key alone can't always be title-cased back into the right label).
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

    // `unsaved`: the open pipeline, not a file yet, so picking it just closes the menu.
    const makeOpt = (name, { demo = false, hint, label, title, unsaved = false } = {}) => {
      const locked = Config.locked.includes(name);
      const needsDownload = demo && !Config.downloaded.includes(name);
      const li = document.createElement('li');
      li.className = 'cfg-opt' + (locked ? ' ref' : '') + (demo ? ' demo' : '') +
        (needsDownload ? ' needs-download' : '') + (name === Config.selected || unsaved ? ' selected' : '');
      li.setAttribute('role', 'option');
      if (title) li.title = title;
      // Not-yet-downloaded demos get a download icon instead of the plain
      // reference dot, so picking one visibly means "fetch its data first".
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

  // Re-read the folder on open, so files added/removed outside the dashboard
  // (via the Folder button) show up without a page reload.
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

  // Called on every edit. Editing a locked config forks it: the save target
  // becomes a new custom_run_N so the original can't be written over.
  noteEdit() {
    if (Config.loading || !Config.isLocked(Config.current)) return;
    const forked = Config.nextCustomName();
    const from = Config.current;
    Config.current = Config.withExt(forked);
    document.getElementById('config-name').value = forked;
    // Nothing on the server is selected any more — this is a new, unsaved config.
    Config.selected = '';
    Config.renderPicker();
    Config.updateControls();
    Config.updateSaveLabel();
    Config.notice('Editing a copy — save it in the', { action: {
      label: 'YAML tab', onClick: () => document.querySelector('.tab[data-tab="yaml"]').click() } });
  },

  // sticky: stays up until the next notice() call instead of auto-clearing —
  // used for errors and while a demo download is in progress, so neither is
  // ever missed because it faded out.
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
