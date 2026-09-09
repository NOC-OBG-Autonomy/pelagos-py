// WebGL viewer for captured diagnostic figures.
//
// The runner writes a JSON spec (layout, labels, limits, trace styling) and a
// float32 binary of every trace's full x/y beside each diagnostic PNG (see
// dashboard/fig_spec.py). This draws all of it -- millions of points -- in one
// WebGL context, with axes, ticks, legend, box-zoom and a click tooltip drawn
// on plain 2D canvases above and below it. Panels the step drew with
// sharex/sharey keep their ranges linked. Dates arrive as seconds since the
// spec's t0 (epoch ms) and are formatted from that here.

const Plot = {
  _cache: {},
  specUrl(name) { return '/api/run/figspec/' + encodeURIComponent(name); },
  binUrl(name) { return '/api/run/figbin/' + encodeURIComponent(name.replace(/\.json$/, '.f32')); },
  pointUrl(name, panel, trace, index) {
    return '/api/run/figpoint/' + encodeURIComponent(name)
      + '?panel=' + panel + '&trace=' + trace + '&index=' + index;
  },

  // Specs are cached by filename: names are unique within a run and run.js
  // clears this when the next run starts.
  fetchSpec(name) {
    if (Plot._cache[name]) return Plot._cache[name];
    Plot._cache[name] = fetch(Plot.specUrl(name)).then((r) => {
      if (!r.ok) throw new Error('spec ' + r.status);
      return r.json();
    }).catch((err) => { delete Plot._cache[name]; throw err; });
    return Plot._cache[name];
  },

  // Stream the binary so the caller can show progress; the header is padded to
  // a 4-byte boundary so every array is a zero-copy view on the one buffer.
  async fetchBin(name, onProgress, signal) {
    const r = await fetch(Plot.binUrl(name), { signal });
    if (!r.ok) throw new Error('data ' + r.status);
    const total = Number(r.headers.get('Content-Length')) || 0;
    const reader = r.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (onProgress) onProgress(loaded, total);
    }
    const buf = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    const hl = new DataView(buf.buffer).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(buf.subarray(4, 4 + hl)));
    off = 4 + hl;
    const traces = {};
    for (const t of header.traces) {
      const x = new Float32Array(buf.buffer, off, t.n); off += t.n * 4;
      const y = new Float32Array(buf.buffer, off, t.n); off += t.n * 4;
      let rgba = null;
      if (t.rgba) { rgba = new Uint8Array(buf.buffer, off, t.n * 4); off += t.n * 4; }
      traces[t.panel + '_' + t.trace] = { x, y, rgba };
    }
    return traces;
  },

  async render(host, spec, { name, onProgress, signal } = {}) {
    const data = await Plot.fetchBin(name, onProgress, signal);
    if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
    Plot.purge(host);
    host._chart = new Chart(host, spec, data, name);
    return host._chart;
  },

  purge(host) {
    if (host && host._chart) { host._chart.destroy(); host._chart = null; }
    if (host) host.innerHTML = '';
  },
};

// ---------------------------------------------------------------- ticks ----
const Ticks = {
  niceStep(span, target) {
    const raw = Math.abs(span) / Math.max(1, target);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
    return 10 * p;
  },

  numeric(lo, hi, target, log) {
    const [a, b] = lo < hi ? [lo, hi] : [hi, lo];
    if (log) {
      const out = [];
      const step = Math.max(1, Math.ceil((b - a) / Math.max(1, target)));
      for (let k = Math.ceil(a); k <= b; k += step) out.push({ v: k, label: '1e' + k });
      return out;
    }
    const step = Ticks.niceStep(b - a, target);
    const dec = Math.max(0, -Math.floor(Math.log10(step)));
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) {
      const r = Math.abs(v) < step * 1e-9 ? 0 : v;
      out.push({ v: r, label: (Math.abs(r) >= 1e6 || (step < 1e-4)) ? r.toPrecision(3) : r.toFixed(dec) });
    }
    return out;
  },

  MONTHS: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
  STEPS: [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200,
    86400, 172800, 604800, 1209600],

  pad(n) { return (n < 10 ? '0' : '') + n; },
  hms(d) { return Ticks.pad(d.getUTCHours()) + ':' + Ticks.pad(d.getUTCMinutes()) + ':' + Ticks.pad(d.getUTCSeconds()); },
  hm(d) { return Ticks.pad(d.getUTCHours()) + ':' + Ticks.pad(d.getUTCMinutes()); },
  dmy(d) { return d.getUTCDate() + ' ' + Ticks.MONTHS[d.getUTCMonth()] + ' ' + d.getUTCFullYear(); },
  dm(d) { return d.getUTCDate() + ' ' + Ticks.MONTHS[d.getUTCMonth()]; },

  // Ticks on a date axis: values are seconds since t0 (epoch ms). Each has a
  // main label and, where the day/year changes, a second context line.
  date(lo, hi, target, t0) {
    const [a, b] = lo < hi ? [lo, hi] : [hi, lo];
    const span = b - a;
    const base = t0 / 1000;
    let step = Ticks.STEPS.find((s) => span / s <= target);
    const out = [];
    if (!step) {  // month-aligned steps
      const months = [1, 2, 3, 6, 12, 24, 60].find((m) => span / (m * 30 * 86400) <= target) || 120;
      const d = new Date((base + a) * 1000);
      d.setUTCDate(1); d.setUTCHours(0, 0, 0, 0);
      d.setUTCMonth(Math.floor(d.getUTCMonth() / months) * months);
      for (; ; d.setUTCMonth(d.getUTCMonth() + months)) {
        const v = d.getTime() / 1000 - base;
        if (v > b) break;
        if (v >= a) out.push({ v, label: Ticks.MONTHS[d.getUTCMonth()] + (months >= 12 || d.getUTCMonth() === 0 ? ' ' + d.getUTCFullYear() : ''), sub: '' });
      }
      return out;
    }
    let prevDay = null, prevYear = null;
    for (let t = Math.ceil((base + a) / step) * step; t <= base + b; t += step) {
      const d = new Date(t * 1000);
      const day = d.getUTCFullYear() * 400 + d.getUTCMonth() * 32 + d.getUTCDate();
      let label, sub = '';
      if (step < 60) { label = Ticks.hms(d); if (day !== prevDay) sub = Ticks.dm(d); }
      else if (step < 86400) { label = Ticks.hm(d); if (day !== prevDay) sub = Ticks.dm(d); }
      else { label = Ticks.dm(d); if (d.getUTCFullYear() !== prevYear) sub = String(d.getUTCFullYear()); }
      prevDay = day; prevYear = d.getUTCFullYear();
      out.push({ v: t - base, label, sub });
    }
    return out;
  },

  isoLike(ms) {
    const d = new Date(ms);
    return Ticks.dmy(d) + ' ' + Ticks.hms(d) + (ms % 1000 ? '.' + Ticks.pad(Math.round(ms % 1000 / 10)) : '');
  },
};

// ---------------------------------------------------------------- chart ----
const VS = `#version 300 es
in float ax; in float ay; in vec4 ac;
uniform float uOx, uSx, uOy, uSy, uSize, uOpacity;
uniform vec2 uShift; uniform vec4 uColor; uniform int uPer;
out vec4 vc;
void main() {
  gl_Position = vec4((ax - uOx) * uSx - 1.0 + uShift.x, (ay - uOy) * uSy - 1.0 + uShift.y, 0.0, 1.0);
  gl_PointSize = uSize;
  vc = (uPer == 1 ? ac : uColor) * vec4(1.0, 1.0, 1.0, uOpacity);
}`;
const FS = `#version 300 es
precision mediump float;
in vec4 vc; uniform int uRound; out vec4 o;
void main() {
  if (uRound == 1) { vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard; }
  o = vec4(vc.rgb * vc.a, vc.a);  // premultiplied, matching the canvas compositor
}`;

const FG = '#222', MUTED = '#666', GRID = '#e8eaee', FRAME = '#8a8f98', ACCENT = '#0b6bcb';
const FONT = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
const FONT_B = '600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

class Chart {
  constructor(host, spec, data, name) {
    this.host = host; this.spec = spec; this.name = name;
    this.dpr = window.devicePixelRatio || 1;
    this.stage = document.createElement('div');
    this.stage.className = 'plot-stage';
    this.bg = document.createElement('canvas');
    this.gl = document.createElement('canvas');
    this.fg = document.createElement('canvas');
    this.tip = document.createElement('div');
    this.tip.className = 'plot-tip hidden';
    for (const el of [this.bg, this.gl, this.fg, this.tip]) this.stage.appendChild(el);
    host.appendChild(this.stage);

    this.panels = spec.panels.map((p, i) => this._panel(p, i, data));
    this.box = null; this.pick = null;
    this._initGL();
    this._bind();
    this.resize();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(host);
  }

  _panel(p, i, data) {
    const xlog = p.xscale === 'log', ylog = p.yscale === 'log';
    const tf = (arr, log) => log ? arr.map((v) => Math.log10(v)) : arr;
    const traces = p.traces.map((t, j) => {
      const d = data[i + '_' + j] || { x: new Float32Array(0), y: new Float32Array(0), rgba: null };
      return { spec: t, x: tf(d.x, xlog), y: tf(d.y, ylog), rgba: d.rgba, n: d.x.length, hidden: false };
    });
    const lim = (l, log, k) => {
      if (l && l.every((v) => v !== null && isFinite(v))) return log ? l.map(Math.log10) : l.slice();
      let lo = Infinity, hi = -Infinity;
      for (const t of traces) for (const v of t[k]) if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      if (!isFinite(lo)) return [0, 1];
      const pad = (hi - lo || 1) * 0.05;
      return [lo - pad, hi + pad];
    };
    const [x0, x1] = lim(p.xlim, xlog, 'x');
    const [y0, y1] = lim(p.ylim, ylog, 'y');
    return { spec: p, traces, home: { x0, x1, y0, y1 }, view: { x0, x1, y0, y1 }, xlog, ylog, rect: null, legendHits: [] };
  }

  // ---- GL setup -----------------------------------------------------------
  _initGL() {
    const gl = this.gl.getContext('webgl2', { antialias: false, alpha: true, premultipliedAlpha: true });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.ctx = gl;
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    this.loc = {};
    for (const u of ['uOx', 'uSx', 'uOy', 'uSy', 'uSize', 'uOpacity', 'uShift', 'uColor', 'uPer', 'uRound']) this.loc[u] = gl.getUniformLocation(prog, u);
    this.attr = { ax: gl.getAttribLocation(prog, 'ax'), ay: gl.getAttribLocation(prog, 'ay'), ac: gl.getAttribLocation(prog, 'ac') };
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.SCISSOR_TEST);

    const buf = (arr) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW); return b; };
    for (const p of this.panels) for (const t of p.traces) {
      t.bx = buf(t.x); t.by = buf(t.y);
      t.bc = t.rgba ? buf(t.rgba) : null;
      if (t.spec.mode.includes('lines')) {
        // Segment index list skipping NaN gaps, as matplotlib breaks the line there.
        const idx = new Uint32Array((t.n - 1) * 2);
        let k = 0;
        for (let i = 0; i < t.n - 1; i++) {
          if (isFinite(t.x[i]) && isFinite(t.y[i]) && isFinite(t.x[i + 1]) && isFinite(t.y[i + 1])) { idx[k++] = i; idx[k++] = i + 1; }
        }
        t.bi = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, t.bi);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx.subarray(0, k), gl.STATIC_DRAW);
        t.ni = k;
      }
    }
  }

  _hexToRGBA(hex) {
    const h = (hex || '#1f77b4').replace('#', '');
    return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255, 1];
  }

  drawGL() {
    const gl = this.ctx, dpr = this.dpr, H = this.gl.height;
    gl.clearColor(0, 0, 0, 0);
    gl.disable(gl.SCISSOR_TEST); gl.clear(gl.COLOR_BUFFER_BIT); gl.enable(gl.SCISSOR_TEST);
    for (const p of this.panels) {
      const r = p.rect, v = p.view;
      const vx = Math.round(r.x * dpr), vy = Math.round(H - (r.y + r.h) * dpr);
      const vw = Math.round(r.w * dpr), vh = Math.round(r.h * dpr);
      gl.viewport(vx, vy, vw, vh); gl.scissor(vx, vy, vw, vh);
      gl.uniform1f(this.loc.uOx, v.x0); gl.uniform1f(this.loc.uSx, 2 / (v.x1 - v.x0));
      gl.uniform1f(this.loc.uOy, v.y0); gl.uniform1f(this.loc.uSy, 2 / (v.y1 - v.y0));
      for (const t of p.traces) {
        if (t.hidden || !t.n) continue;
        gl.bindBuffer(gl.ARRAY_BUFFER, t.bx); gl.enableVertexAttribArray(this.attr.ax); gl.vertexAttribPointer(this.attr.ax, 1, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, t.by); gl.enableVertexAttribArray(this.attr.ay); gl.vertexAttribPointer(this.attr.ay, 1, gl.FLOAT, false, 0, 0);
        if (t.bc) {
          gl.bindBuffer(gl.ARRAY_BUFFER, t.bc); gl.enableVertexAttribArray(this.attr.ac); gl.vertexAttribPointer(this.attr.ac, 4, gl.UNSIGNED_BYTE, true, 0, 0);
          gl.uniform1i(this.loc.uPer, 1);
        } else {
          gl.disableVertexAttribArray(this.attr.ac);
          gl.uniform1i(this.loc.uPer, 0);
          gl.uniform4fv(this.loc.uColor, this._hexToRGBA(t.spec.color));
        }
        gl.uniform1f(this.loc.uOpacity, t.spec.opacity == null ? 1 : t.spec.opacity);
        if (t.bi && t.ni) {
          // WebGL lines are 1 device px; thicker strokes are extra passes shifted a pixel.
          gl.uniform1i(this.loc.uRound, 0);
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, t.bi);
          const passes = Math.min(4, Math.max(1, Math.round((t.spec.width || 1) * dpr * 0.75)));
          const shifts = [[0, 0], [2 / vw, 0], [0, 2 / vh], [2 / vw, 2 / vh]];
          for (let k = 0; k < passes; k++) {
            gl.uniform2f(this.loc.uShift, shifts[k][0], shifts[k][1]);
            gl.drawElements(gl.LINES, t.ni, gl.UNSIGNED_INT, 0);
          }
          gl.uniform2f(this.loc.uShift, 0, 0);
        }
        if (t.spec.mode.includes('markers')) {
          gl.uniform1i(this.loc.uRound, 1);
          gl.uniform2f(this.loc.uShift, 0, 0);
          gl.uniform1f(this.loc.uSize, Math.min(15, Math.max(3, (t.spec.size || 4) * 1.5)) * dpr);
          gl.drawArrays(gl.POINTS, 0, t.n);
        }
      }
    }
  }

  // ---- layout ---------------------------------------------------------------
  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.w = w; this.h = h;
    for (const c of [this.bg, this.gl, this.fg]) {
      c.width = Math.round(w * this.dpr); c.height = Math.round(h * this.dpr);
      c.style.width = w + 'px'; c.style.height = h + 'px';
    }
    this.draw();
  }

  layout() {
    const top = this.spec.suptitle ? 26 : 4;
    const stage = { x: 0, y: top, w: this.w, h: this.h - top };
    const n = this.panels.length;
    const ctx = this.fg.getContext('2d');
    ctx.font = FONT;
    this.panels.forEach((p, i) => {
      const c = p.spec.cell;
      // A panel with a sharex sibling below it borrows that one's x axis (matplotlib
      // hides the upper tick labels too), so skip its labels and the room for them.
      p.xHidden = this.panels.some((q, j) => q !== p && q.spec.share_x === p.spec.share_x && (
        c && q.spec.cell ? q.spec.cell[1] > c[1] && q.spec.cell[0] < c[0] + c[2] && c[0] < q.spec.cell[0] + q.spec.cell[2] : j > i));
      const outer = c
        ? { x: stage.x + c[0] * stage.w, y: stage.y + c[1] * stage.h, w: c[2] * stage.w, h: c[3] * stage.h }
        : { x: stage.x, y: stage.y + (i / n) * stage.h, w: stage.w, h: stage.h / n };
      const yt = this._yticks(p, outer.h - 60);
      const tw = Math.max(20, ...yt.map((t) => ctx.measureText(t.label).width));
      const ml = 10 + tw + (p.spec.ylabel ? 20 : 4);
      const mt = (p.spec.title ? 18 : 0) + (p.spec.top_axis ? 32 : 10);
      const mb = p.xHidden ? 8 : (p.spec.xdate ? 34 : 22) + (p.spec.xlabel ? 16 : 0);
      p.outer = outer;
      p.rect = { x: outer.x + ml, y: outer.y + mt, w: Math.max(20, outer.w - ml - 16), h: Math.max(20, outer.h - mt - mb) };
    });
  }

  _yticks(p, hpx) {
    const v = p.view, target = Math.max(2, hpx / 45);
    return p.spec.ydate ? Ticks.date(v.y0, v.y1, target, this.spec.t0).map((t) => ({ v: t.v, label: t.sub ? t.sub + ' ' + t.label : t.label }))
      : Ticks.numeric(v.y0, v.y1, target, p.ylog);
  }

  // Data <-> CSS pixel maps for one panel.
  px(p, x) { return p.rect.x + (x - p.view.x0) / (p.view.x1 - p.view.x0) * p.rect.w; }
  py(p, y) { return p.rect.y + p.rect.h - (y - p.view.y0) / (p.view.y1 - p.view.y0) * p.rect.h; }
  dx(p, px) { return p.view.x0 + (px - p.rect.x) / p.rect.w * (p.view.x1 - p.view.x0); }
  dy(p, py) { return p.view.y0 + (p.rect.y + p.rect.h - py) / p.rect.h * (p.view.y1 - p.view.y0); }

  // ---- 2D drawing -------------------------------------------------------------
  draw() {
    if (!this.w) return;
    this.layout();
    this.drawGL();
    const bg = this.bg.getContext('2d');
    bg.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); bg.clearRect(0, 0, this.w, this.h);
    for (const p of this.panels) { p.ticks = this._ticks(p); this._drawGrid(p, bg, p.ticks); }
    this._drawFG();
  }

  _ticks(p) {
    const r = p.rect, v = p.view;
    return {
      xt: p.spec.xdate ? Ticks.date(v.x0, v.x1, Math.max(2, r.w / 90), this.spec.t0) : Ticks.numeric(v.x0, v.x1, Math.max(2, r.w / 80), p.xlog),
      yt: this._yticks(p, r.h),
    };
  }

  _drawGrid(p, bg, { xt, yt }) {
    const r = p.rect;
    bg.fillStyle = '#fff'; bg.fillRect(r.x, r.y, r.w, r.h);
    bg.strokeStyle = GRID; bg.lineWidth = 1;
    bg.beginPath();
    for (const t of xt) { const x = Math.round(this.px(p, t.v)) + 0.5; bg.moveTo(x, r.y); bg.lineTo(x, r.y + r.h); }
    for (const t of yt) { const y = Math.round(this.py(p, t.v)) + 0.5; bg.moveTo(r.x, y); bg.lineTo(r.x + r.w, y); }
    bg.stroke();
  }

  _drawAxes(p, fg, { xt, yt }) {
    const r = p.rect, s = p.spec;
    fg.font = FONT; fg.fillStyle = FG; fg.strokeStyle = FRAME; fg.lineWidth = 1;
    fg.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
    fg.textAlign = 'center'; fg.textBaseline = 'top';
    if (!p.xHidden) for (const t of xt) {
      const x = this.px(p, t.v);
      fg.fillStyle = FG; fg.fillText(t.label, x, r.y + r.h + 5);
      if (t.sub) { fg.fillStyle = MUTED; fg.fillText(t.sub, x, r.y + r.h + 18); }
    }
    fg.textAlign = 'right'; fg.textBaseline = 'middle'; fg.fillStyle = FG;
    for (const t of yt) fg.fillText(t.label, r.x - 6, this.py(p, t.v));
    if (s.xlabel && !p.xHidden) { fg.textAlign = 'center'; fg.textBaseline = 'bottom'; fg.fillText(s.xlabel, r.x + r.w / 2, p.outer.y + p.outer.h - 2); }
    if (s.ylabel) {
      fg.save(); fg.translate(p.outer.x + 12, r.y + r.h / 2); fg.rotate(-Math.PI / 2);
      fg.textAlign = 'center'; fg.textBaseline = 'middle'; fg.fillText(s.ylabel, 0, 0); fg.restore();
    }
    if (s.title) { fg.font = FONT_B; fg.textAlign = 'center'; fg.textBaseline = 'top'; fg.fillText(s.title, r.x + r.w / 2, p.outer.y + 2); fg.font = FONT; }
    if (s.top_axis) this._drawTopAxis(p, fg);

    fg.save(); fg.beginPath(); fg.rect(r.x, r.y, r.w, r.h); fg.clip();
    for (const rl of s.reflines || []) {
      fg.strokeStyle = rl.color; fg.globalAlpha = rl.opacity == null ? 1 : rl.opacity;
      fg.lineWidth = Math.max(1, rl.width || 1);
      fg.setLineDash(rl.dash === 'dash' ? [6, 4] : rl.dash === 'dot' ? [2, 3] : rl.dash === 'dashdot' ? [6, 3, 2, 3] : []);
      fg.beginPath();
      if (rl.axis === 'y') { const y = this.py(p, p.ylog ? Math.log10(rl.value) : rl.value); fg.moveTo(r.x, y); fg.lineTo(r.x + r.w, y); }
      else { const x = this.px(p, p.xlog ? Math.log10(rl.value) : rl.value); fg.moveTo(x, r.y); fg.lineTo(x, r.y + r.h); }
      fg.stroke();
    }
    fg.restore(); fg.setLineDash([]); fg.globalAlpha = 1;
    this._drawLegend(p, fg);
  }

  // N_MEASUREMENTS along the top: index ticks placed at their interpolated time.
  _drawTopAxis(p, fg) {
    const ta = p.spec.top_axis, r = p.rect, v = p.view;
    const interp = (x, xs, ys) => {
      if (x <= xs[0]) return ys[0];
      if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
      let lo = 0, hi = xs.length - 1;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
      return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
    };
    const i0 = interp(v.x0, ta.t, ta.i), i1 = interp(v.x1, ta.t, ta.i);
    fg.font = FONT; fg.fillStyle = FG; fg.textAlign = 'center'; fg.textBaseline = 'bottom';
    fg.strokeStyle = FRAME; fg.lineWidth = 1; fg.beginPath();
    for (const t of Ticks.numeric(i0, i1, Math.max(2, r.w / 80))) {
      const x = this.px(p, interp(t.v, ta.i, ta.t));
      if (x < r.x - 1 || x > r.x + r.w + 1) continue;
      fg.fillText(t.label, x, r.y - 5);
      fg.moveTo(Math.round(x) + 0.5, r.y); fg.lineTo(Math.round(x) + 0.5, r.y - 3);
    }
    fg.stroke();
    fg.fillStyle = MUTED; fg.fillText(ta.label, r.x + r.w / 2, r.y - 18);
  }

  _drawLegend(p, fg) {
    p.legendHits = [];
    if (!p.spec.legend) return;
    const entries = p.traces.filter((t) => t.spec.label && !t.spec.label.startsWith('_'));
    if (!entries.length) return;
    const r = p.rect, lh = 16, pad = 6;
    fg.font = FONT;
    const title = p.spec.legend_title || '';
    const w = Math.max(...entries.map((t) => fg.measureText(t.spec.label).width), title ? fg.measureText(title).width - 22 : 0) + 22 + pad * 2;
    const h = entries.length * lh + pad * 2 + (title ? lh : 0);
    const x = r.x + r.w - w - 8, y = r.y + 8;
    fg.fillStyle = 'rgba(255,255,255,.88)'; fg.strokeStyle = GRID;
    fg.fillRect(x, y, w, h); fg.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    fg.textAlign = 'left'; fg.textBaseline = 'middle';
    let cy = y + pad + lh / 2;
    if (title) { fg.fillStyle = MUTED; fg.fillText(title, x + pad, cy); cy += lh; }
    for (const t of entries) {
      fg.globalAlpha = t.hidden ? 0.35 : 1;
      const color = t.spec.color || (t.rgba ? this._rgbaCss(t.rgba, 0) : '#888');
      fg.fillStyle = color; fg.strokeStyle = color; fg.lineWidth = 2;
      if (t.spec.mode.includes('lines')) { fg.beginPath(); fg.moveTo(x + pad, cy); fg.lineTo(x + pad + 16, cy); fg.stroke(); }
      if (t.spec.mode.includes('markers')) { fg.beginPath(); fg.arc(x + pad + 8, cy, 3.5, 0, Math.PI * 2); fg.fill(); }
      fg.fillStyle = FG; fg.fillText(t.spec.label, x + pad + 22, cy);
      p.legendHits.push({ x, y: cy - lh / 2, w, h: lh, trace: t });
      cy += lh;
    }
    fg.globalAlpha = 1;
  }

  _rgbaCss(rgba, i) { return 'rgba(' + rgba[i * 4] + ',' + rgba[i * 4 + 1] + ',' + rgba[i * 4 + 2] + ',' + (rgba[i * 4 + 3] / 255) + ')'; }

  _drawOverlay(fg) {
    if (this.box) {
      const b = this.box, x = Math.min(b.x0, b.x1), y = Math.min(b.y0, b.y1);
      fg.fillStyle = 'rgba(11,107,203,.10)'; fg.strokeStyle = ACCENT; fg.lineWidth = 1; fg.setLineDash([4, 3]);
      fg.fillRect(x, y, Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
      fg.strokeRect(x + 0.5, y + 0.5, Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
      fg.setLineDash([]);
    }
    if (this.pick) {
      const pk = this.pick, p = pk.panel;
      const x = this.px(p, pk.x), y = this.py(p, pk.y);
      if (x >= p.rect.x && x <= p.rect.x + p.rect.w && y >= p.rect.y && y <= p.rect.y + p.rect.h) {
        fg.strokeStyle = ACCENT; fg.lineWidth = 2; fg.beginPath(); fg.arc(x, y, 6, 0, Math.PI * 2); fg.stroke();
        fg.strokeStyle = '#fff'; fg.lineWidth = 1; fg.beginPath(); fg.arc(x, y, 7.5, 0, Math.PI * 2); fg.stroke();
        this._placeTip(x, y);
      } else this.tip.classList.add('hidden');
    }
  }

  // ---- interaction ------------------------------------------------------------
  _panelAt(x, y) {
    return this.panels.find((p) => x >= p.rect.x && x <= p.rect.x + p.rect.w && y >= p.rect.y && y <= p.rect.y + p.rect.h) || null;
  }

  _pos(ev) { const b = this.fg.getBoundingClientRect(); return { x: ev.clientX - b.left, y: ev.clientY - b.top }; }

  _bind() {
    const el = this.fg;
    el.style.cursor = 'crosshair';
    const down = (ev) => {
      if (ev.button !== 0) return;
      const { x, y } = this._pos(ev);
      const p = this._panelAt(x, y);
      if (!p) return;
      const hit = p.legendHits.find((h) => x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h);
      if (hit) { hit.trace.hidden = !hit.trace.hidden; this.draw(); return; }
      this.box = { panel: p, x0: x, y0: y, x1: x, y1: y };
      el.setPointerCapture(ev.pointerId);
    };
    const move = (ev) => {
      if (!this.box) return;
      const { x, y } = this._pos(ev);
      const r = this.box.panel.rect;
      this.box.x1 = Math.max(r.x, Math.min(r.x + r.w, x));
      this.box.y1 = Math.max(r.y, Math.min(r.y + r.h, y));
      this._drawFG();
    };
    const up = (ev) => {
      if (!this.box) return;
      const b = this.box; this.box = null;
      if (Math.abs(b.x1 - b.x0) > 4 && Math.abs(b.y1 - b.y0) > 4) {
        this.zoomTo(b.panel, [this.dx(b.panel, Math.min(b.x0, b.x1)), this.dx(b.panel, Math.max(b.x0, b.x1))],
          [this.dy(b.panel, Math.max(b.y0, b.y1)), this.dy(b.panel, Math.min(b.y0, b.y1))]);
      } else this._click(b.panel, b.x0, b.y0);
    };
    el.addEventListener('pointerdown', down);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', () => { this.box = null; this._drawFG(); });
    el.addEventListener('dblclick', (ev) => { ev.preventDefault(); this.reset(); });
    this._onKey = (ev) => { if (ev.key === 'Escape' && this.pick) { this.pick = null; this.tip.classList.add('hidden'); this._drawFG(); ev.stopPropagation(); } };
    document.addEventListener('keydown', this._onKey, true);
  }

  // Redraw just the top 2D layer (axes, legend, box drag, pick marker).
  _drawFG() {
    const fg = this.fg.getContext('2d');
    fg.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); fg.clearRect(0, 0, this.w, this.h); fg.font = FONT;
    if (this.spec.suptitle) { fg.fillStyle = FG; fg.font = FONT_B; fg.textAlign = 'center'; fg.textBaseline = 'middle'; fg.fillText(this.spec.suptitle, this.w / 2, 13); }
    for (const p of this.panels) this._drawAxes(p, fg, p.ticks);
    this._drawOverlay(fg);
  }

  // Box-zoom: x follows the panel's sharex group, y its sharey group. Ranges
  // are given as [lo, hi] in data units and re-oriented to each panel's own
  // direction so an inverted depth axis stays inverted.
  zoomTo(panel, xr, yr) {
    xr = [Math.min(...xr), Math.max(...xr)]; yr = [Math.min(...yr), Math.max(...yr)];
    for (const p of this.panels) {
      if (p === panel || p.spec.share_x === panel.spec.share_x) {
        const inv = p.view.x0 > p.view.x1; p.view.x0 = inv ? xr[1] : xr[0]; p.view.x1 = inv ? xr[0] : xr[1];
      }
      if (p === panel || p.spec.share_y === panel.spec.share_y) {
        const inv = p.view.y0 > p.view.y1; p.view.y0 = inv ? yr[1] : yr[0]; p.view.y1 = inv ? yr[0] : yr[1];
      }
    }
    this.draw();
  }

  reset() {
    for (const p of this.panels) p.view = Object.assign({}, p.home);
    this.pick = null; this.tip.classList.add('hidden');
    this.draw();
  }

  // Nearest visible point (in screen space) within a small radius; brute force
  // over every point is ~tens of ms even at millions, and only runs on click.
  _click(p, cx, cy) {
    const R2 = 12 * 12;
    const kx = p.rect.w / (p.view.x1 - p.view.x0), ky = p.rect.h / (p.view.y1 - p.view.y0);
    const ox = p.rect.x - p.view.x0 * kx, oy = p.rect.y + p.rect.h + p.view.y0 * ky;
    let best = null, bestD = R2;
    p.traces.forEach((t, j) => {
      if (t.hidden) return;
      const xs = t.x, ys = t.y;
      for (let i = 0; i < t.n; i++) {
        const sx = xs[i] * kx + ox - cx, sy = oy - ys[i] * ky - cy;
        const d = sx * sx + sy * sy;
        if (d < bestD) { bestD = d; best = { trace: j, index: i }; }
      }
    });
    if (!best) { this.pick = null; this.tip.classList.add('hidden'); this._drawFG(); return; }
    const t = p.traces[best.trace];
    const pick = { panel: p, trace: t, index: best.index, x: t.x[best.index], y: t.y[best.index], exact: null };
    this.pick = pick;
    this._drawFG();
    fetch(Plot.pointUrl(this.name, this.panels.indexOf(p), best.trace, best.index))
      .then((r) => (r.ok ? r.json() : null))
      .then((ex) => { if (this.pick === pick && ex) { pick.exact = ex; this._drawFG(); } })
      .catch(() => {});
  }

  _fmt(p, v, isDate, log, exact) {
    if (exact !== undefined && exact !== null) return isDate ? exact.replace('T', ' ').replace(/\+00:00$/, '') : String(exact);
    if (isDate) return Ticks.isoLike(this.spec.t0 + v * 1000);
    const val = log ? Math.pow(10, v) : v;
    return Math.abs(val) >= 1e6 || (val !== 0 && Math.abs(val) < 1e-4) ? val.toExponential(4) : String(+val.toPrecision(6));
  }

  _placeTip(x, y) {
    const pk = this.pick, p = pk.panel, s = p.spec;
    const label = pk.trace.spec.label && !pk.trace.spec.label.startsWith('_') ? pk.trace.spec.label : '';
    const ex = pk.exact || {};
    const xs = this._fmt(p, pk.x, s.xdate, p.xlog, ex.x), ys = this._fmt(p, pk.y, s.ydate, p.ylog, ex.y);
    this.tip.innerHTML = (label ? '<b>' + this._esc(label) + '</b><br>' : '')
      + '<span>' + this._esc(s.xlabel || 'x') + '</span> ' + this._esc(xs) + '<br>'
      + '<span>' + this._esc(s.ylabel || 'y') + '</span> ' + this._esc(ys)
      + '<br><span>index</span> ' + pk.index
      + (pk.exact ? '' : '<br><i>float32 shown; exact values loading…</i>');
    this.tip.classList.remove('hidden');
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    let left = x + 14, top = y - th / 2;
    if (left + tw > this.w - 4) left = x - tw - 14;
    top = Math.max(4, Math.min(this.h - th - 4, top));
    this.tip.style.left = left + 'px'; this.tip.style.top = top + 'px';
  }

  _esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

  // PNG of the current view: the GL layer is redrawn so its buffer is fresh.
  toPNG() {
    const out = document.createElement('canvas');
    out.width = this.gl.width; out.height = this.gl.height;
    const c = out.getContext('2d');
    c.fillStyle = '#fff'; c.fillRect(0, 0, out.width, out.height);
    this.drawGL();
    c.drawImage(this.bg, 0, 0); c.drawImage(this.gl, 0, 0); c.drawImage(this.fg, 0, 0);
    return out.toDataURL('image/png');
  }

  destroy() {
    if (this._ro) this._ro.disconnect();
    document.removeEventListener('keydown', this._onKey, true);
    const gl = this.ctx;
    if (gl) {
      for (const p of this.panels) for (const t of p.traces) for (const b of [t.bx, t.by, t.bc, t.bi]) if (b) gl.deleteBuffer(b);
      const ext = gl.getExtension('WEBGL_lose_context');
      if (ext) ext.loseContext();
    }
    this.stage.remove();
  }
}
