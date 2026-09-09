// Live RAM meter for a run. Each __PELAGOS_MEM__ marker (one per executed step)
// adds a point; the sparkline plots RSS across steps so transient per-step
// spikes — and their release — are visible at a glance. Per-step detail lives
// in each dot's hover tooltip.

const Mem = {
  points: [],  // {rss, stepPeak, added, data, label, t} per step, in run order
  samples: [], // {t, rss} every 0.5 s of processing time (__PELAGOS_SAMPLE__)
  peak: 0,     // running max RSS this run — kept only to scale the plot's y-axis

  // Shown immediately when a run starts (not hidden) so the meter feels live
  // the instant Run is pressed, rather than popping in once the first step's
  // __PELAGOS_MEM__ marker arrives.
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

  // "<rss>\t<runPeak>\t<data>\t<label>\t<stepPeak>\t<peakLabel>\t<stepStart>\t<t>".
  // runPeak/peakLabel are consumed only for the plot scale / tooltip; trailing
  // fields are absent on older runners, so fall back gracefully. Values are MB.
  add(payload) {
    const parts = payload.split('\t');
    const rss = parseFloat(parts[0]);
    if (!isFinite(rss)) return;
    const runPeak = parseFloat(parts[1]);
    const data = parseFloat(parts[2]); // NaN when the field is empty
    const label = (parts[3] || '').trim();
    const stepPeak = isFinite(parseFloat(parts[4])) ? parseFloat(parts[4]) : rss;
    const stepStart = parseFloat(parts[6]); // NaN on older runners
    const t = parseFloat(parts[7]);         // processing seconds at step end
    // A step's own growth: how much RSS it added on top of what it inherited.
    // Without step-start, fall back to growth over the previous settle.
    const prev = Mem.points.length ? Mem.points[Mem.points.length - 1].rss : stepPeak;
    const added = Math.max(0, stepPeak - (isFinite(stepStart) ? stepStart : prev));
    Mem.peak = isFinite(runPeak) ? runPeak : Math.max(Mem.peak, rss);
    Mem.points.push({ rss, stepPeak, added, data: isFinite(data) ? data : null, label,
      t: isFinite(t) ? t : null });
    document.getElementById('mem-meter').classList.remove('hidden');
    Mem.setNum('mem-cur', rss);
    Mem.setNum('mem-peak', Mem.peak);
    Mem.setNum('mem-data', data);
    Mem.render();
  },

  // "<active s>\t<rss>": one RSS reading between step markers, so the trace
  // shows what happens *within* a step, not just where it ended up.
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

  // MB -> "1.9 GB" / "512 MB". null -> "–".
  fmt(mb) {
    if (mb == null || !isFinite(mb)) return '–';
    return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB';
  },

  setNum(id, mb) {
    const el = document.getElementById(id);
    if (el) el.textContent = Mem.fmt(mb);
  },

  // Redraw the sparkline. The line is each step's *in-step* peak RSS (the true
  // high-water while it ran, which the boundary reading alone would miss), so a
  // step that briefly spikes shows up. The y-axis runs 0..run-peak so a drop
  // after a spike is obvious. Hover detail is a custom tooltip (see below),
  // not native <title>s: the dots are only ~2px, far too small a target to
  // reliably hover, so layout() instead tracks each point's x position and a
  // mousemove listener picks the nearest one regardless of exact cursor y.
  //
  // The viewBox is set to the SVG's actual rendered pixel size (not a fixed
  // 100-unit box) so 1 user unit == 1px in both axes. Otherwise
  // preserveAspectRatio="none" stretches x and y by different factors and
  // circles render as squashed ellipses.
  render() {
    const svg = document.getElementById('mem-spark');
    if (!svg) return;
    const H = 34, pad = 3;
    const W = Math.max(svg.clientWidth || 1, 60);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const n = Mem.points.length;
    const top = Mem.peak || 1;
    const y = (mb) => H - pad - (mb / top) * (H - 2 * pad);
    // With samples the x-axis is processing time and the trace is one series:
    // the 0.5 s samples plus each step's own end reading, sorted by time (the
    // two come from different clocks, so an unsorted merge can double back).
    // The dots sit on that same series, so the line always passes through them.
    // Without samples (older runner, no psutil on the server) fall back to one
    // evenly spaced slot per step with the dot at the step's in-step peak.
    const timed = Mem.samples.length > 0;
    let series, dotXY;
    if (timed) {
      const pts = Mem.points.filter((p) => p.t != null);
      series = Mem.samples.concat(pts.map((p) => ({ t: p.t, rss: p.rss })))
        .sort((a, b) => a.t - b.t);
      const tmax = series[series.length - 1].t || 1;
      const xt = (t) => pad + (t / tmax) * (W - 2 * pad);
      series = series.map((s) => ({ x: xt(s.t), y: y(s.rss) }));
      dotXY = Mem.points.map((p) => (p.t == null ? null : { x: xt(p.t), y: y(p.rss) }));
    } else {
      const x = (i) => (n <= 1 ? W / 2 : pad + (i * (W - 2 * pad)) / (n - 1));
      series = Mem.points.map((p, i) => ({ x: x(i), y: y(p.stepPeak) }));
      dotXY = series;
    }
    const svgns = 'http://www.w3.org/2000/svg';
    svg.innerHTML = '';
    Mem._layout = { xs: dotXY.map((d) => (d ? d.x : -1)) };

    // Peak guide line, so the ceiling of the run is always marked.
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

    // On the time axis the step dots bunch up wherever quick steps run back to
    // back, so they stay invisible until hovered (see showTooltip).
    Mem.points.forEach((p, i) => {
      const d = dotXY[i];
      if (!d) return;
      const isPeak = p.stepPeak >= Mem.peak;
      const dot = document.createElementNS(svgns, 'circle');
      dot.setAttribute('cx', d.x);
      dot.setAttribute('cy', d.y);
      dot.setAttribute('r', isPeak ? 2.6 : 1.8);
      dot.setAttribute('class', (isPeak ? 'mem-spark-dot peak' : 'mem-spark-dot') +
        (timed ? ' quiet' : ''));
      dot.dataset.i = i;
      svg.appendChild(dot);
    });
  },

  // Nearest point to a mouse event's x position, in SVG user units.
  nearestPoint(svg, clientX) {
    const xs = Mem._layout && Mem._layout.xs;
    if (!xs || !xs.length) return -1;
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return -1;
    const viewBox = svg.viewBox.baseVal;
    const mx = (clientX - rect.left) * (viewBox.width / rect.width);
    let best = 0, bestDist = Infinity;
    xs.forEach((xi, i) => {
      if (xi < 0) return; // step with no time stamp: not drawn
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
    tip.innerHTML = `<b>${p.label || 'step ' + (i + 1)}</b>` +
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

// Processing-time readout beside the RAM meter. The runner owns the clock
// (__PELAGOS_TIME__: its accumulated active seconds, whether it is paused, and
// its wall clock at that moment); while running the browser ticks on from the
// marker's epoch, so a replayed backlog after a reconnect still lands on the
// right figure. Paused (review / manual QC) time is not counted.
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

  // The process has exited: the final marker's figure stands.
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

// The spark's viewBox tracks its own pixel width (see render()), so a window
// resize needs a redraw to stay unstretched.
window.addEventListener('resize', () => {
  if (Mem.points.length) Mem.render();
});

// Hover anywhere over the spark: pick the nearest step by x rather than
// requiring a precise hit on a ~2px dot.
(function () {
  const svg = document.getElementById('mem-spark');
  if (!svg) return;
  svg.addEventListener('mousemove', (e) => {
    const i = Mem.nearestPoint(svg, e.clientX);
    if (i >= 0) Mem.showTooltip(i, e.clientX, e.clientY);
  });
  svg.addEventListener('mouseleave', Mem.hideTooltip);
})();
