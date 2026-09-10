// Demos tab: the demo deployments from Config (missions/labels/downloaded), as
// cards grouped by mission. Clicking one calls Config.load, which downloads first if needed.
const Demos = {
  render() {
    const root = document.getElementById('demos-list');
    if (!root) return;
    root.innerHTML = '';
    for (const [mission, names] of Object.entries(Config.missions)) {
      const here = names.filter((n) => Config.known.includes(n));
      if (!here.length) continue;
      const sec = document.createElement('section');
      sec.className = 'demos-mission';
      const h = document.createElement('h3');
      h.className = 'demos-mission-title';
      h.textContent = mission;
      sec.appendChild(h);
      const grid = document.createElement('div');
      grid.className = 'demos-grid';
      for (const name of here) grid.appendChild(Demos.card(name));
      sec.appendChild(grid);
      root.appendChild(sec);
    }
    Demos.renderHead();
  },

  // "N downloaded · 1.2 GB" and the Delete-all button, hidden when nothing is on disk.
  renderHead() {
    const n = Config.downloaded.length;
    const bytes = Config.downloaded.reduce((t, name) => t + (Config.sizes[name] || 0), 0);
    document.getElementById('demos-sub').textContent = n
      ? `${n} file${n === 1 ? '' : 's'} downloaded · ${fmtBytes(bytes)} on disk`
      : 'Nothing downloaded yet';
    document.getElementById('btn-demos-clean').classList.toggle('hidden', !n);
  },

  card(name) {
    const downloaded = Config.downloaded.includes(name);
    const active = name === Config.selected;
    const busy = name === Demos.downloading;
    const el = document.createElement('div');
    el.className = 'demo-card' + (active ? ' active' : '') + (downloaded ? '' : ' needs-download')
      + (busy ? ' downloading' : '');
    el.dataset.name = name;
    el.setAttribute('role', 'button');
    el.tabIndex = 0;
    el.title = downloaded ? 'Load this demo pipeline' : 'Not downloaded yet — loading fetches its data file first';
    const ico = document.createElement('span');
    ico.className = 'demo-card-ico';
    ico.appendChild(Icon.el(busy ? 'rerun' : active ? 'check' : downloaded ? 'play' : 'download', 14));
    el.appendChild(ico);
    const text = document.createElement('span');
    text.className = 'demo-card-text';
    text.textContent = Config.demoLabel(name);
    el.appendChild(text);
    const hint = document.createElement('span');
    hint.className = 'demo-card-hint';
    hint.textContent = busy ? 'connecting…' : active ? 'loaded' : downloaded ? fmtBytes(Config.sizes[name] || 0) : 'download';
    el.appendChild(hint);
    if (downloaded) {
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'demo-card-del';
      del.title = 'Delete the downloaded file';
      del.appendChild(Icon.el('trash2', 13));
      del.addEventListener('click', (e) => { e.stopPropagation(); Demos.remove(name); });
      el.appendChild(del);
    }
    const load = async () => {
      if (Config.busy || RunLock.running) return;
      try { await Config.load(name); }
      catch (e) { Config.notice('Could not load ' + name + ': ' + e.message, { sticky: true, err: true }); }
    };
    el.addEventListener('click', load);
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); load(); } });
    return el;
  },

  // Poll the server while `name` downloads and fill its card with colour.
  // Called with null once the load settles (success or failure).
  downloading: null,
  trackDownload(name) {
    clearTimeout(Demos._pollTimer);
    Demos.downloading = name;
    if (!name) { Demos.render(); return; }
    Demos.render();
    Run.showTab('demos');
    const poll = async () => {
      if (Demos.downloading !== name) return;
      let p = null;
      try { p = (await API.demoProgress())[name] || null; } catch (e) { /* retry */ }
      Demos.setProgress(name, p);
      Demos._pollTimer = setTimeout(poll, 500);
    };
    poll();
  },

  setProgress(name, p) {
    const el = document.querySelector(`.demo-card[data-name="${CSS.escape(name)}"]`);
    if (!el) return;
    const frac = p && p.total ? p.done / p.total : 0;
    el.style.setProperty('--p', (frac * 100).toFixed(1) + '%');
    el.classList.toggle('indeterminate', !!p && !p.total);
    const hint = el.querySelector('.demo-card-hint');
    hint.textContent = !p ? 'connecting…'
      : p.total ? `${Math.round(frac * 100)}% of ${fmtBytes(p.total)}` : fmtBytes(p.done);
  },

  async remove(name) {
    if (RunLock.running) return;
    if (!confirm(`Delete the downloaded ${Config.demoLabel(name)} file (${fmtBytes(Config.sizes[name] || 0)})?`)) return;
    try { await API.deleteDemo(name); }
    catch (e) { Config.notice('Could not delete: ' + e.message, { sticky: true, err: true }); }
    await Config.refreshList(Config.selected);
  },

  async clean() {
    if (RunLock.running) return;
    const n = Config.downloaded.length;
    if (!n || !confirm(`Delete all ${n} downloaded demo file${n === 1 ? '' : 's'}? They can be downloaded again from here.`)) return;
    try { await API.cleanDemos(); }
    catch (e) { Config.notice('Could not delete: ' + e.message, { sticky: true, err: true }); }
    await Config.refreshList(Config.selected);
  },
};

function fmtBytes(b) {
  if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB';
  if (b >= 1e6) return Math.round(b / 1e6) + ' MB';
  if (b >= 1e3) return Math.round(b / 1e3) + ' kB';
  return b + ' B';
}
