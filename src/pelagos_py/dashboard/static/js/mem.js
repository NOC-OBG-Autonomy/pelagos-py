// Live RAM meter for a run: a line of the server's RAM samples; hovering shows the nearest one.

const Mem = {
  samples: [], // {t, rss, step} every 0.5 s of processing time (__PELAGOS_SAMPLE__)
  peak: 0,     // running max RSS this run, also the plot's y-axis top

  // Shown as soon as Run is pressed, not when the first marker arrives.
  reset() {
    Mem.samples = [];
    Mem.peak = 0;
    const meter = document.getElementById('mem-meter');
    if (meter) meter.classList.remove('hidden');
    const spark = document.getElementById('mem-spark');
    if (spark) spark.innerHTML = '';
    ['mem-cur', 'mem-peak', 'mem-data'].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.textContent = '–';
    });
  },

  // "<data MB>" at a step's end.
  setData(payload) {
    Mem.setNum('mem-data', parseFloat(payload));
  },

  // "<active s>\t<rss MB>"
  sample(payload) {
    const parts = payload.split('\t');
    const t = parseFloat(parts[0]), rss = parseFloat(parts[1]);
    if (!isFinite(t) || !isFinite(rss)) return;
    const cur = Run.currentStep;
    const step = cur ? cur.name + (cur.test ? ' · ' + cur.test : '') : '';
    Mem.samples.push({ t, rss, step });
    if (rss > Mem.peak) Mem.peak = rss;
    Mem.setNum('mem-cur', rss);
    Mem.setNum('mem-peak', Mem.peak);
    Mem.render();
  },

  fmt(mb) { return fmtBytes(mb == null ? null : mb * 1048576); }, // readings arrive in MiB

  setNum(id, mb) {
    const el = document.getElementById(id);
    if (el) el.textContent = Mem.fmt(mb);
  },

  // The viewBox matches the rendered pixel size so preserveAspectRatio="none" can't squash the hover dot.
  render() {
    const svg = document.getElementById('mem-spark');
    if (!svg) return;
    const H = 34, pad = 3;
    const W = Math.max(svg.clientWidth || 1, 60);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = '';
    if (!Mem.samples.length) return;
    const top = Mem.peak || 1;
    const tmax = Mem.samples[Mem.samples.length - 1].t || 1;
    Mem.xy = Mem.samples.map((s) => ({
      x: pad + (s.t / tmax) * (W - 2 * pad),
      y: H - pad - (s.rss / top) * (H - 2 * pad),
    }));
    const linePts = Mem.xy.map((p) => `${p.x},${p.y}`).join(' ');
    const first = Mem.xy[0], last = Mem.xy[Mem.xy.length - 1];
    svg.innerHTML =
      `<line class="mem-spark-peak" x1="0" x2="${W}" y1="${pad}" y2="${pad}"/>` +
      `<polygon class="mem-spark-area" points="${first.x},${H - pad} ${linePts} ${last.x},${H - pad}"/>` +
      `<polyline class="mem-spark-line" points="${linePts}"/>` +
      '<circle class="mem-spark-dot hidden" r="2.6"/>';
  },

  showTooltip(clientX, clientY) {
    const svg = document.getElementById('mem-spark');
    const tip = document.getElementById('mem-tooltip');
    const rect = svg.getBoundingClientRect();
    if (!Mem.xy || !Mem.xy.length || !rect.width) return;
    const mx = (clientX - rect.left) * (svg.viewBox.baseVal.width / rect.width);
    let i = 0;
    Mem.xy.forEach((p, j) => { if (Math.abs(p.x - mx) < Math.abs(Mem.xy[i].x - mx)) i = j; });
    const s = Mem.samples[i];
    const dot = svg.querySelector('.mem-spark-dot');
    dot.setAttribute('cx', Mem.xy[i].x);
    dot.setAttribute('cy', Mem.xy[i].y);
    dot.classList.remove('hidden');
    tip.innerHTML = `<b>${Mem.fmt(s.rss)}</b> at ${RunClock.fmt(s.t)}` +
      (s.step ? `<br>${escapeHtml(s.step)}` : '');
    tip.style.left = `${clientX}px`;
    tip.style.top = `${clientY}px`;
    tip.classList.remove('hidden');
  },

  hideTooltip() {
    document.getElementById('mem-tooltip').classList.add('hidden');
    const dot = document.querySelector('#mem-spark .mem-spark-dot');
    if (dot) dot.classList.add('hidden');
  },
};

// Processing-time readout. The runner owns the clock (__PELAGOS_TIME__) and the browser ticks on from
// the marker's epoch, so a replayed backlog after a reconnect still lands on the right figure.
const RunClock = {
  active: 0, paused: true, epoch: null, timer: null,

  reset() {
    RunClock.stop();
    RunClock.active = 0; RunClock.paused = true; RunClock.epoch = null;
    RunClock.show();
  },

  // "<active s>\t<paused 0/1>\t<epoch s>"
  update(payload) {
    const parts = payload.split('\t');
    const active = parseFloat(parts[0]);
    if (!isFinite(active)) return;
    RunClock.active = active;
    RunClock.paused = parts[1].trim() === '1';
    RunClock.epoch = parseFloat(parts[2]);
    RunClock.show();
    if (!RunClock.paused && !RunClock.timer) RunClock.timer = setInterval(RunClock.show, 1000);
    if (RunClock.paused) RunClock.stop();
  },

  stop() {
    clearInterval(RunClock.timer);
    RunClock.timer = null;
  },

  seconds() {
    if (RunClock.paused || !isFinite(RunClock.epoch)) return RunClock.active;
    return RunClock.active + Math.max(0, Date.now() / 1000 - RunClock.epoch);
  },

  fmt(s) {
    if (s < 60) return s.toFixed(1) + ' s';
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    const h = Math.floor(m / 60);
    const mm = h ? String(m % 60).padStart(2, '0') : String(m);
    return (h ? h + ':' : '') + mm + ':' + String(sec).padStart(2, '0');
  },

  show() {
    const el = document.getElementById('run-time');
    if (!el) return;
    el.textContent = RunClock.epoch === null ? '–' : RunClock.fmt(RunClock.seconds());
    el.parentElement.classList.toggle('paused', RunClock.paused && RunClock.epoch !== null);
  },
};

// The viewBox tracks the spark's pixel width, so a resize needs a redraw.
window.addEventListener('resize', () => {
  if (Mem.samples.length) Mem.render();
});

(function () {
  const svg = document.getElementById('mem-spark');
  if (!svg) return;
  svg.addEventListener('mousemove', (e) => Mem.showTooltip(e.clientX, e.clientY));
  svg.addEventListener('mouseleave', Mem.hideTooltip);
})();
