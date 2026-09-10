// Report tab, lower half: every file runs have left in the output folders
// (reports, exports, logs, kept report figures), with open/delete. The folders
// come from the loaded config (out_directory + each output_path) plus the
// demo data folder, so nothing depends on knowing where pip put the package.
const Outputs = {
  files: [],
  dirs: [],
  KINDS: { report: 'Reports', data: 'Exported data', log: 'Logs', figures: 'Report figures' },

  // Folders and input files named by the YAML pane.
  fromConfig() {
    let cfg = {};
    try { cfg = jsyaml.load(editor.getValue()) || {}; } catch (e) { /* mid-edit */ }
    const dirs = [], inputs = [];
    const out = (cfg.pipeline || {}).out_directory;
    if (out) dirs.push(String(out));
    for (const step of cfg.steps || []) {
      const p = step && step.parameters ? step.parameters : {};
      if (p.output_path) dirs.push(String(p.output_path).replace(/[^/\\]*$/, '') || '.');
      if (p.file_path) inputs.push(String(p.file_path));
    }
    return { dirs, inputs };
  },

  async refresh() {
    const { dirs, inputs } = Outputs.fromConfig();
    try {
      const res = await API.listOutputs(dirs, inputs);
      Outputs.files = res.files || [];
      Outputs.dirs = res.dirs || [];
    } catch (e) {
      Outputs.files = []; Outputs.dirs = [];
    }
    Outputs.render();
  },

  render() {
    const list = document.getElementById('outputs-list');
    const sub = document.getElementById('outputs-sub');
    list.innerHTML = '';
    const n = Outputs.files.length;
    const bytes = Outputs.files.reduce((t, f) => t + f.size, 0);
    sub.textContent = n ? `${n} file${n === 1 ? '' : 's'} · ${fmtBytes(bytes)}` : 'No output files yet';
    document.getElementById('btn-outputs-clean').classList.toggle('hidden', !n);
    for (const [kind, title] of Object.entries(Outputs.KINDS)) {
      const files = Outputs.files.filter((f) => f.kind === kind);
      if (!files.length) continue;
      const h = document.createElement('h4');
      h.className = 'outputs-kind';
      h.textContent = title;
      list.appendChild(h);
      for (const f of files) list.appendChild(Outputs.row(f));
    }
  },

  row(f) {
    const el = document.createElement('div');
    el.className = 'output-row' + (f.kind === 'report' && /\.pdf$/i.test(f.name) ? ' output-pdf' : '');
    const ico = document.createElement('span');
    ico.className = 'output-ico output-' + f.kind;
    ico.appendChild(Icon.el(f.kind === 'figures' ? 'folder' : 'file', 14));
    el.appendChild(ico);
    const text = document.createElement('div');
    text.className = 'output-text';
    text.innerHTML = `<strong>${escapeHtml(f.name)}</strong>` +
      `<span class="output-meta">${escapeHtml(f.dir)}</span>`;
    el.appendChild(text);
    const meta = document.createElement('span');
    meta.className = 'output-size';
    meta.textContent = fmtBytes(f.size) + ' · ' + Outputs.when(f.mtime);
    el.appendChild(meta);
    // PDFs/logs open in the browser; data files and figure folders are already
    // on disk, so they get "Show in Finder" instead of a pointless download.
    if (f.kind === 'report' || f.kind === 'log') {
      const open = document.createElement('a');
      open.className = 'icon-btn';
      open.href = API.outputUrl(f.path);
      open.target = '_blank';
      open.rel = 'noopener';
      open.title = 'Open';
      open.appendChild(Icon.el('external', 14));
      open.addEventListener('click', (e) => e.stopPropagation());
      el.appendChild(open);
    }
    const reveal = document.createElement('button');
    reveal.type = 'button';
    reveal.className = 'icon-btn';
    reveal.title = 'Show in Finder';
    reveal.appendChild(Icon.el('folder', 14));
    reveal.addEventListener('click', async (e) => {
      e.stopPropagation();
      try { await API.revealOutputs(f.dir); } catch (err) { alert(err.message); }
    });
    el.appendChild(reveal);
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'icon-btn danger';
    del.title = 'Delete';
    del.appendChild(Icon.el('trash2', 14));
    del.addEventListener('click', (e) => { e.stopPropagation(); Outputs.remove(f); });
    el.appendChild(del);
    if (el.classList.contains('output-pdf')) {
      el.title = 'Preview this report above';
      el.addEventListener('click', () => Run.showReport(f.path, f.name));
    }
    return el;
  },

  when(mtime) {
    const d = new Date(mtime * 1000);
    const today = new Date();
    const sameDay = d.toDateString() === today.toDateString();
    return sameDay
      ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : d.toLocaleDateString([], { day: 'numeric', month: 'short' });
  },

  async remove(f) {
    if (RunLock.running) return;
    if (!confirm(`Delete ${f.name}?`)) return;
    try { await API.deleteOutput(f.path); }
    catch (e) { alert('Could not delete: ' + e.message); }
    await Outputs.refresh();
  },

  async clean() {
    if (RunLock.running) return;
    const n = Outputs.files.length;
    if (!n || !confirm(`Delete all ${n} output file${n === 1 ? '' : 's'}? Input data files are kept.`)) return;
    const { dirs, inputs } = Outputs.fromConfig();
    try { await API.cleanOutputs(dirs, inputs); }
    catch (e) { alert('Could not delete: ' + e.message); }
    Run.clearReport();
    await Outputs.refresh();
  },

  init() {
    document.getElementById('btn-outputs-clean').addEventListener('click', () => Outputs.clean());
    document.getElementById('btn-outputs-reveal').addEventListener('click', async () => {
      const { dirs } = Outputs.fromConfig();
      try { await API.revealOutputs(dirs[0] || ''); }
      catch (e) { alert(e.message); }
    });
    document.getElementById('btn-demos-clean').addEventListener('click', () => Demos.clean());
  },
};
