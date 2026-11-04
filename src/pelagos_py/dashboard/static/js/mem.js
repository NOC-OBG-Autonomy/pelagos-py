// Live RAM meter for a run: one point per __PELAGOS_MEM__ marker, with per-step detail in each dot's tooltip.

const Mem = {
  points: [],  // {rss, stepPeak, added, data, label, t} per step, in run order
  samples: [], // {t, rss} every 0.5 s of processing time (__PELAGOS_SAMPLE__)
  peak: 0,     // running max RSS this run, also the plot's y-axis top

  // Shown as soon as Run is pressed, not when the first marker arrives.
  reset() {
    Mem.points = [];
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

  // "<rss>\t<runPeak>\t<data>\t<label>\t<stepPeak>\t<peakLabel>\t<stepStart>\t<t>", MB.
  add(payload) {
    const parts = payload.split('\t');
    const rss = parseFloat(parts[0]);
    if (!isFinite(rss)) return;
    const data = parseFloat(parts[2]); // NaN when the field is empty
    const label = (parts[3] || '').trim();
    const stepPeak = parseFloat(parts[4]);
    const stepStart = parseFloat(parts[6]);
    const t = parseFloat(parts[7]); // processing seconds at step end
    Mem.peak = parseFloat(parts[1]);
    Mem.points.push({ rss, stepPeak, added: Math.max(0, stepPeak - stepStart),
      data: isFinite(data) ? data : null, label, t });
    document.getElementById('mem-meter').classList.remove('hidden');
    Mem.setNum('mem-cur', rss);
    Mem.setNum('mem-peak', Mem.peak);
    Mem.setNum('mem-data', data);
    Mem.render();
  },

  // "<active s>\t<rss>": readings between step markers, to show what happens within a step.
  sample(payload) {
    const parts = payload.split('\t');
    const t = parseFloat(parts[0]), rss = parseFloat(parts[1]);
    if (!isFinite(t) || !isFinite(rss)) return;
    Mem.samples.push({ t, rss });
    if (rss > Mem.peak) Mem.peak = rss;
    Mem.setNum('mem-cur', rss);
    Mem.setNum('mem-peak', Mem.peak);
    Mem.render();
  },

  fmt(mb) {
    if (mb == null || !isFinite(mb)) return '–';
    return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB';
  },

  setNum(id, mb) {
    const el = document.getElementById(id);
    if (el) el.textContent = Mem.fmt(mb);
  },

  // The viewBox matches the rendered pixel size so preserveAspectRatio="none" can't squash circles.
  render() {
    const svg = document.getElementById('mem-spark');
    if (!svg) return;
    const H = 34, pad = 3;
    const W = Math.max(svg.clientWidth || 1, 60);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = '';
    const top = Mem.peak || 1;
    const y = (mb) => H - pad - (mb / top) * (H - 2 * pad);
    // Samples and step-end readings come from different clocks, so sort the merge.
    const readings = Mem.samples.concat(Mem.points.map((p) => ({ t: p.t, rss: p.rss })))
      .sort((a, b) => a.t - b.t);
    if (!readings.length) return;
    const tmax = readings[readings.length - 1].t || 1;
    const xt = (t) => pad + (t / tmax) * (W - 2 * pad);
    const series = readings.map((s) => ({ x: xt(s.t), y: y(s.rss) }));
    const dotXY = Mem.points.map((p) => ({ x: xt(p.t), y: y(p.rss) }));
    const svgns = 'http://www.w3.org/2000/svg';
    Mem._layout = { xs: dotXY.map((d) => d.x) };

    const guide = document.createElementNS(svgns, 'line');
    guide.setAttribute('x1', 0); guide.setAttribute('x2', W);
    guide.setAttribute('y1', y(top)); guide.setAttribute('y2', y(top));
    guide.setAttribute('class', 'mem-spark-peak');
    svg.appendChild(guide);

    if (series.length > 1) {
      const linePts = series.map((s) => `${s.x},${s.y}`).join(' ');
      const area = document.createElementNS(svgns, 'polygon');
      area.setAttribute('points',
        `${series[0].x},${H - pad} ${linePts} ${series[series.length - 1].x},${H - pad}`);
      area.setAttribute('class', 'mem-spark-area');
      svg.appendChild(area);

      const line = document.createElementNS(svgns, 'polyline');
      line.setAttribute('points', linePts);
      line.setAttribute('class', 'mem-spark-line');
      svg.appendChild(line);
    }

    // Step dots bunch up where quick steps run back to back, so they stay hidden until hovered.
    Mem.points.forEach((p, i) => {
      const d = dotXY[i];
      const isPeak = p.stepPeak >= Mem.peak;
      const dot = document.createElementNS(svgns, 'circle');
      dot.setAttribute('cx', d.x);
      dot.setAttribute('cy', d.y);
      dot.setAttribute('r', isPeak ? 2.6 : 1.8);
      dot.setAttribute('class', 'mem-spark-dot quiet' + (isPeak ? ' peak' : ''));
      dot.dataset.i = i;
      svg.appendChild(dot);
    });
  },

  nearestPoint(svg, clientX) {
    const xs = Mem._layout && Mem._layout.xs;
    if (!xs || !xs.length) return -1;
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return -1;
    const viewBox = svg.viewBox.baseVal;
    const mx = (clientX - rect.left) * (viewBox.width / rect.width);
    let best = 0, bestDist = Infinity;
    xs.forEach((xi, i) => {
      const d = Math.abs(xi - mx);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    return best;
  },

  showTooltip(i, clientX, clientY) {
    const svg = document.getElementById('mem-spark');
    const tip = document.getElementById('mem-tooltip');
    if (!svg || !tip) return;
    const p = Mem.points[i];
    if (!p) return;
    svg.querySelectorAll('.mem-spark-dot.active').forEach((d) => d.classList.remove('active'));
    const dot = svg.querySelector(`.mem-spark-dot[data-i="${i}"]`);
    if (dot) dot.classList.add('active');
    tip.innerHTML = `<b>${escapeHtml(p.label || 'step ' + (i + 1))}</b>` +
      `<br>peak ${Mem.fmt(p.stepPeak)} &nbsp;+${Mem.fmt(p.added)} this step` +
      `<br>settled ${Mem.fmt(p.rss)}` +
      (p.data != null ? `<br>data ${Mem.fmt(p.data)}` : '');
    tip.style.left = `${clientX}px`;
    tip.style.top = `${clientY}px`;
    tip.classList.remove('hidden');
  },

  hideTooltip() {
    const tip = document.getElementById('mem-tooltip');
    if (tip) tip.classList.add('hidden');
    const svg = document.getElementById('mem-spark');
    if (svg) svg.querySelectorAll('.mem-spark-dot.active').forEach((d) => d.classList.remove('active'));
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
  if (Mem.points.length) Mem.render();
});

// Pick the nearest step by x; a ~2px dot is too small to hit.
(function () {
  const svg = document.getElementById('mem-spark');
  if (!svg) return;
  svg.addEventListener('mousemove', (e) => {
    const i = Mem.nearestPoint(svg, e.clientX);
    if (i >= 0) Mem.showTooltip(i, e.clientX, e.clientY);
  });
  svg.addEventListener('mouseleave', Mem.hideTooltip);
})();
