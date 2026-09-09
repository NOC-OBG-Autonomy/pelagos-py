// Demos tab: the demo deployments from Config (missions/labels/downloaded),
// as cards grouped by mission. Clicking one is the same as picking it from
// the old Open menu — Config.load handles the download-first case.
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
  },

  card(name) {
    const downloaded = Config.downloaded.includes(name);
    const active = name === Config.selected;
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'demo-card' + (active ? ' active' : '') + (downloaded ? '' : ' needs-download');
    el.title = downloaded ? 'Load this demo pipeline' : 'Not downloaded yet — loading fetches its data file first';
    const ico = document.createElement('span');
    ico.className = 'demo-card-ico';
    ico.appendChild(Icon.el(active ? 'check' : downloaded ? 'play' : 'download', 14));
    el.appendChild(ico);
    const text = document.createElement('span');
    text.className = 'demo-card-text';
    text.textContent = Config.demoLabel(name);
    el.appendChild(text);
    const hint = document.createElement('span');
    hint.className = 'demo-card-hint';
    hint.textContent = active ? 'loaded' : downloaded ? 'ready' : 'download';
    el.appendChild(hint);
    el.addEventListener('click', async () => {
      if (Config.busy || (typeof RunLock !== 'undefined' && RunLock.running)) return;
      try { await Config.load(name); }
      catch (e) { Config.notice('Could not load ' + name + ': ' + e.message, { sticky: true, err: true }); }
    });
    return el;
  },
};
