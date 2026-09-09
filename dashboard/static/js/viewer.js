// Full-window viewer (lightbox) for diagnostic figures. Arrow keys / buttons
// walk a set of figures; Esc closes. Figures the runner could serialise (see
// fig_spec.py) open as an interactive WebGL chart of the full data -- box-zoom,
// double-click to reset, click a point for its exact values, legend toggling --
// with a progress bar while the data streams in. The toolbar switches back to
// the PNG; figures with no spec only ever show the PNG.

const Viewer = {
  el: null,
  imgEl: null,
  items: [],   // [{fname, caption, spec, url}]
  index: 0,
  interactive: true,  // sticky across figures while paging

  _ensure() {
    if (Viewer.el) return;
    const el = document.createElement('div');
    el.className = 'viewer hidden';
    el.innerHTML =
      `<div class="viewer-bar">` +
      `<span class="viewer-caption"></span>` +
      `<span class="viewer-note"></span>` +
      `<span class="viewer-count"></span>` +
      `<button class="viewer-btn viewer-reset hidden" title="Reset zoom (double-click the plot does the same)">Reset</button>` +
      `<button class="viewer-btn viewer-toggle hidden"></button>` +
      `<button class="viewer-btn viewer-icon viewer-png hidden" title="Download the current view as PNG">${Icon.svg('download', 14)}</button>` +
      `<a class="viewer-btn viewer-icon viewer-raw" target="_blank" rel="noopener" title="Open the PNG in a new tab">${Icon.svg('external', 14)}</a>` +
      `<button class="viewer-btn viewer-icon viewer-close" title="Close (Esc)">${Icon.svg('close', 16)}</button>` +
      `</div>` +
      `<div class="viewer-progress hidden"><div></div></div>` +
      `<div class="viewer-stage">` +
      `<button class="viewer-nav prev" title="Previous (←)">${Icon.svg('right', 22)}</button>` +
      `<img class="viewer-img" alt="" />` +
      `<div class="viewer-plot hidden"></div>` +
      `<button class="viewer-nav next" title="Next (→)">${Icon.svg('right', 22)}</button>` +
      `</div>` +
      `<div class="viewer-hint hidden">Drag a box to zoom · double-click to reset · click a point for its values · click a legend entry to hide it</div>`;
    document.body.appendChild(el);
    Viewer.el = el;
    Viewer.imgEl = el.querySelector('.viewer-img');
    Viewer.plotEl = el.querySelector('.viewer-plot');
    Viewer.progressEl = el.querySelector('.viewer-progress');

    el.querySelector('.viewer-close').onclick = Viewer.close;
    el.querySelector('.viewer-toggle').onclick = (e) => { e.stopPropagation(); Viewer.interactive = !Viewer.interactive; Viewer.show(); };
    el.querySelector('.viewer-reset').onclick = (e) => { e.stopPropagation(); if (Viewer.plotEl._chart) Viewer.plotEl._chart.reset(); };
    el.querySelector('.viewer-png').onclick = (e) => { e.stopPropagation(); Viewer.downloadPNG(); };
    Viewer.plotEl.onclick = (e) => e.stopPropagation();
    el.querySelector('.prev').onclick = (e) => { e.stopPropagation(); Viewer.step(-1); };
    el.querySelector('.next').onclick = (e) => { e.stopPropagation(); Viewer.step(1); };
    Viewer.imgEl.onclick = (e) => { e.stopPropagation(); Viewer.toggleZoom(); };
    el.querySelector('.viewer-stage').onclick = Viewer.close;
    document.addEventListener('keydown', Viewer._onKey);
  },

  _onKey(e) {
    if (!Viewer.el || Viewer.el.classList.contains('hidden')) return;
    if (e.key === 'Escape') { Viewer.close(); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { Viewer.step(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { Viewer.step(1); e.preventDefault(); }
  },

  url(fname) { return '/api/run/figure/' + encodeURIComponent(fname); },

  // Filenames restart at fig_001.png every run, so a bare URL could be served
  // from cache as a previous run's image. Minted once, when the figure is captured.
  _seq: 0,
  freshUrl(fname) { return Viewer.url(fname) + '?v=' + (++Viewer._seq); },
  src(fig) { return fig.url || Viewer.url(fig.fname); },

  open(items, index = 0) {
    if (!items || !items.length) return;
    Viewer._ensure();
    Viewer.items = items;
    Viewer.index = Math.max(0, Math.min(index, items.length - 1));
    Viewer.el.classList.remove('hidden');
    Viewer.show();
  },

  show() {
    const it = Viewer.items[Viewer.index];
    if (!it) return;
    const q = (sel) => Viewer.el.querySelector(sel);
    q('.viewer-caption').textContent = it.caption || it.fname;
    q('.viewer-count').textContent = Viewer.items.length > 1 ? `${Viewer.index + 1} / ${Viewer.items.length}` : '';
    q('.viewer-raw').href = Viewer.src(it);
    const multi = Viewer.items.length > 1;
    Viewer.el.querySelectorAll('.viewer-nav').forEach((b) => b.classList.toggle('hidden', !multi));

    const live = !!(it.spec && Viewer.interactive);
    const toggle = q('.viewer-toggle');
    toggle.classList.toggle('hidden', !it.spec);
    toggle.textContent = Viewer.interactive ? 'Image' : 'Interactive';
    toggle.title = Viewer.interactive ? 'Show the original matplotlib image' : 'Redraw this plot with the full data';
    q('.viewer-reset').classList.toggle('hidden', !live);
    q('.viewer-png').classList.toggle('hidden', !live);
    q('.viewer-hint').classList.toggle('hidden', !live);
    Viewer.note('');
    Viewer.progress(null);

    // Abandon any fetch still in flight for the previous figure and free its GPU buffers.
    if (Viewer._abort) Viewer._abort.abort();
    Plot.purge(Viewer.plotEl);
    if (live) Viewer.showPlot(it);
    else Viewer.showImage(it);
  },

  showImage(it) {
    Viewer.plotEl.classList.add('hidden');
    Viewer.imgEl.classList.remove('hidden', 'zoomed');
    Viewer.imgEl.src = Viewer.src(it);
    Viewer.imgEl.alt = it.caption || it.fname;
  },

  // Anything that goes wrong -- spec missing, no WebGL2 -- falls back to the image.
  showPlot(it) {
    const token = ++Viewer._token;
    const abort = new AbortController();
    Viewer._abort = abort;
    Viewer.imgEl.classList.add('hidden');
    Viewer.plotEl.classList.remove('hidden');
    Viewer.plotEl.innerHTML = '<div class="viewer-loading">Loading plot…</div>';
    Viewer.progress(0);
    Plot.fetchSpec(it.spec)
      .then((spec) => {
        if (token !== Viewer._token) return;
        Viewer.note(Viewer.fmtPoints(spec.points));
        return Plot.render(Viewer.plotEl, spec, {
          name: it.spec, signal: abort.signal,
          onProgress: (loaded, total) => { if (token === Viewer._token) Viewer.progress(total ? loaded / total : null, loaded); },
        });
      })
      .then(() => { if (token === Viewer._token) Viewer.progress(null); })
      .catch((err) => {
        if (token !== Viewer._token || err.name === 'AbortError') return;
        Viewer.progress(null);
        Viewer.showImage(it);
        Viewer.note('interactive view unavailable' + (err && err.message ? ' — ' + err.message : ''));
      });
  },

  _token: 0,

  fmtPoints(n) {
    if (!n) return '';
    return (n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(0) + 'k' : String(n)) + ' points';
  },

  // fraction 0..1 fills the bar; null hides it; undefined total shows bytes so far.
  progress(fraction, loaded) {
    const bar = Viewer.progressEl;
    if (!bar) return;
    bar.classList.toggle('hidden', fraction === null);
    if (fraction === null) return;
    const fill = bar.firstElementChild;
    fill.style.width = (fraction * 100).toFixed(1) + '%';
    if (loaded) Viewer.note((loaded / 1048576).toFixed(1) + ' MB' + (fraction ? ' · ' + Math.round(fraction * 100) + '%' : ''));
  },

  note(text) {
    const el = Viewer.el && Viewer.el.querySelector('.viewer-note');
    if (el) el.textContent = text;
  },

  downloadPNG() {
    const chart = Viewer.plotEl._chart;
    if (!chart) return;
    const it = Viewer.items[Viewer.index];
    const a = document.createElement('a');
    a.href = chart.toPNG();
    a.download = (it.fname || 'plot.png').replace(/\.png$/, '') + '_view.png';
    a.click();
  },

  step(dir) {
    if (Viewer.items.length < 2) return;
    Viewer.index = (Viewer.index + dir + Viewer.items.length) % Viewer.items.length;
    Viewer.show();
  },

  toggleZoom() { Viewer.imgEl.classList.toggle('zoomed'); },

  close() {
    if (!Viewer.el) return;
    Viewer.el.classList.add('hidden');
    Viewer._token += 1;
    if (Viewer._abort) Viewer._abort.abort();
    Plot.purge(Viewer.plotEl);
  },

  // A clickable thumbnail/card for one figure that opens the viewer on `items`.
  // Shared by the Plots tab gallery and the paused-step review panel.
  card(items, index, { caption = true, cls = '' } = {}) {
    const it = items[index];
    const fig = document.createElement('figure');
    fig.className = 'plot-card' + (cls ? ' ' + cls : '');
    if (it.isLog) {
      // A log-only step (e.g. Load Data, Export) draws no figure -- show the
      // diagnostics text it printed instead. An error card is the same shape.
      fig.classList.add('log-card');
      if (it.isError) fig.classList.add('error-card');
      const pre = document.createElement('pre');
      pre.className = 'log-card-text';
      pre.textContent = it.text;
      fig.appendChild(pre);
      fig.title = it.isError
        ? 'This step failed — edit its parameters and re-run, or Continue to skip it'
        : 'Diagnostics log — this step prints a summary instead of plotting';
      return fig;
    }
    const img = document.createElement('img');
    img.src = Viewer.src(it);
    img.alt = it.caption || it.fname;
    img.loading = 'lazy';
    fig.appendChild(img);
    if (caption && it.caption) {
      const cap = document.createElement('figcaption');
      cap.textContent = it.caption;
      fig.appendChild(cap);
    }
    fig.onclick = () => Viewer.open(items, index);
    if (it.spec) fig.classList.add('has-plot');
    fig.title = it.spec ? 'Click to open — this plot is interactive' : 'Click to view full size';
    return fig;
  },
};
