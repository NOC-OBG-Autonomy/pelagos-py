// Demos tab: the demo dataset as a card. Config.load downloads first if needed.
const Demos = {
  render() {
    const root = document.getElementById('demos-list');
    if (!root) return;
    root.innerHTML = '';
    const grid = Forms.el('div', { class: 'demos-grid' });
    for (const name of Config.demo) grid.appendChild(Demos.card(name));
    root.appendChild(grid);
  },

  card(name) {
    const downloaded = Config.downloaded.includes(name);
    const active = name === Config.selected;
    const busy = Demos.downloading.has(name);
    const el = document.createElement('div');
    el.className = 'demo-card' + (active ? ' active' : '') + (downloaded ? '' : ' needs-download')
      + (busy ? ' downloading indeterminate' : '');
    el.dataset.name = name;
    el.setAttribute('role', 'button');
    el.tabIndex = 0;
    el.title = downloaded ? 'Load this demo pipeline' : 'Not downloaded yet — loading fetches its data file first';
    const load = async () => {
      if (RunLock.running) return;
      try { await Config.load(name); }
      catch (e) { Config.notice('Could not load ' + name + ': ' + e.message, { sticky: true, err: true }); }
    };
    el.appendChild(Forms.el('span', { class: 'demo-card-text', textContent: Config.demoLabel(name) }));
    el.appendChild(downloaded || busy
      ? Forms.el('span', { class: 'demo-card-hint', textContent: busy ? '' : active ? 'loaded' : fmtBytes(Config.sizes[name] || 0) })
      : Forms.button('Download', { cls: 'sm demo-card-hint', onclick: (e) => { e.stopPropagation(); Demos.download(name); } }));
    if (downloaded) {
      el.appendChild(Forms.button('', { icon: 'trash2', iconSize: 13, cls: 'icon-btn demo-card-del',
        title: 'Delete the downloaded file', onclick: (e) => { e.stopPropagation(); Demos.remove(name); } }));
    }
    el.addEventListener('click', load);
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); load(); } });
    return el;
  },

  downloading: new Set(),
  resumed: new Set(), // picked up after a page reload, so no request of ours ends them
  _polling: false,

  async resume() {
    let all = {};
    try { all = await API.demoProgress(); } catch (e) { return; }
    for (const name of Object.keys(all)) {
      Demos.resumed.add(name);
      Demos.trackDownload(name, true);
    }
  },

  // Picking a demo that's already downloading waits on the same request.
  requests: new Map(),
  download(name) {
    if (!Demos.downloading.has(name)) Demos.requests.set(name, Demos.fetchFile(name));
    return Demos.requests.get(name);
  },

  async fetchFile(name) {
    Demos.trackDownload(name, true);
    try { await API.downloadDemo(name); }
    catch (e) { Config.notice('Could not download ' + Config.demoLabel(name) + ': ' + e.message, { sticky: true, err: true }); }
    Demos.trackDownload(name, false);
    Demos.requests.delete(name);
    await Config.refreshList(Config.selected);
  },

  trackDownload(name, on) {
    if (on) Demos.downloading.add(name);
    else Demos.downloading.delete(name);
    Demos.render();
    if (!on || Demos._polling) return;
    Run.showTab('files');
    Demos._polling = true;
    const poll = async () => {
      if (!Demos.downloading.size) { Demos._polling = false; return; }
      let all = {};
      try { all = await API.demoProgress(); } catch (e) { /* retry */ }
      for (const n of Demos.downloading) {
        if (Demos.resumed.has(n) && !all[n]) {
          Demos.resumed.delete(n);
          Demos.trackDownload(n, false);
          Config.refreshList(Config.selected);
        } else {
          Demos.setProgress(n, all[n] || null);
        }
      }
      setTimeout(poll, 500);
    };
    poll();
  },

  setProgress(name, p) {
    const el = document.querySelector(`.demo-card[data-name="${CSS.escape(name)}"]`);
    if (!el) return;
    const frac = p && p.total ? p.done / p.total : 0;
    el.style.setProperty('--p', (frac * 100).toFixed(1) + '%');
    el.classList.toggle('indeterminate', !p || !p.total);
    const hint = el.querySelector('.demo-card-hint');
    hint.textContent = !p ? '' : p.total ? `${Math.round(frac * 100)}%` : fmtBytes(p.done);
  },

  async remove(name) {
    if (RunLock.running) return;
    if (!confirm(`Delete the downloaded ${Config.demoLabel(name)} file (${fmtBytes(Config.sizes[name] || 0)})?`)) return;
    try { await API.deleteDemo(name); }
    catch (e) { Config.notice('Could not delete: ' + e.message, { sticky: true, err: true }); }
    if (Build.active && Build.name === name) Build.close(); // its file is gone
    await Config.refreshList(Config.selected);
  },
};

// The user's own input files: paths remembered in the browser (the dashboard runs locally, nothing is uploaded).
const Files = {
  KEY: 'pelagos.files',
  info: {},
  list() { try { return JSON.parse(localStorage.getItem(Files.KEY)) || []; } catch (e) { return []; } },
  save(paths) { try { localStorage.setItem(Files.KEY, JSON.stringify(paths)); } catch (e) { /* no storage */ } },

  init() {
    const input = document.getElementById('files-path');
    const addBtn = document.getElementById('btn-files-add');
    let timer = null;
    input.addEventListener('input', () => {
      addBtn.disabled = true;
      clearTimeout(timer);
      timer = setTimeout(async () => {
        const paths = Files.parse(input.value);
        if (!paths.length || !paths.every((p) => p.endsWith('.nc'))) return;
        try { const info = await API.filesInfo(paths); addBtn.disabled = !paths.every((p) => info[p] && info[p].exists); }
        catch (e) { /* stays disabled */ }
      }, 250);
    });
    const add = () => { if (addBtn.disabled) return; Files.add(Files.parse(input.value)); input.value = ''; addBtn.disabled = true; };
    addBtn.addEventListener('click', add);
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') add(); });
    document.getElementById('btn-files-browse').addEventListener('click', async () => {
      try { Files.add(await API.pickFiles(Files.list()[0])); }
      catch (e) { Config.notice(e.message, { sticky: true, err: true }); }
    });
    Files.render();
    // Files get moved or deleted outside the dashboard.
    const panel = document.querySelector('.tab-panel[data-panel="files"]');
    setInterval(() => {
      if (!panel.classList.contains('hidden')) Files.render();
    }, 5000);
  },

  // Pasted text: one path per line, or several space-separated absolute paths.
  parse(text) { return (text || '').split(/\n|\s+(?=[/~])/).map((p) => p.trim()).filter(Boolean); },

  add(paths) {
    if (!paths.length) return;
    Files.save([...paths, ...Files.list().filter((p) => !paths.includes(p))]);
    Files.render();
  },

  forget(path) { Files.save(Files.list().filter((p) => p !== path)); Files.render(); },

  async render() {
    const root = document.getElementById('files-list');
    const paths = Files.list();
    root.innerHTML = '';
    if (!paths.length) {
      root.appendChild(Forms.el('div', { class: 'hint', textContent: 'No files yet — browse for one or paste its path above.' }));
      return;
    }
    try { Files.info = await API.filesInfo(paths); } catch (e) { Files.info = {}; }
    const gone = paths.filter((p) => Files.info[p] && !Files.info[p].exists);
    if (gone.length) { Files.save(paths.filter((p) => !gone.includes(p))); Files.render(); return; }
    const grid = Forms.el('div', { class: 'demos-grid' });
    for (const p of paths) grid.appendChild(Files.card(p));
    root.appendChild(grid);
  },

  card(path) {
    const info = Files.info[path] || {};
    const el = Forms.el('div', { class: 'demo-card file-card', role: 'button', tabIndex: 0, title: 'Set up a pipeline for this file' });
    el.appendChild(Forms.el('span', { class: 'demo-card-text' },
      Forms.el('span', { textContent: path.split('/').pop() }),
      Forms.el('span', { class: 'demo-card-path', textContent: path.replace(/[^/]*$/, ''), title: path })));
    el.appendChild(Forms.el('span', { class: 'demo-card-hint', textContent: fmtBytes(info.size || 0) }));
    el.appendChild(Forms.button('', { icon: 'folder', iconSize: 13, cls: 'icon-btn', title: 'Show in Finder',
      onclick: async (e) => { e.stopPropagation(); try { await API.revealFile(path); } catch (err) { alert(err.message); } } }));
    el.appendChild(Forms.button('', { icon: 'close', iconSize: 13, cls: 'icon-btn', title: 'Remove from this list (the file is not deleted)',
      onclick: (e) => { e.stopPropagation(); Files.forget(path); } }));
    const pick = () => { if (!RunLock.running) Build.start({ name: null, filePath: path }); };
    el.addEventListener('click', pick);
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    return el;
  },
};
