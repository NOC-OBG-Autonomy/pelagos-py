// Fits both lines of the brand text to the icon height; measures real ink via canvas, not line boxes.
(function () {
  const INSET = 0.08, GAP = 0.144, R = 200;
  const ctx = document.createElement('canvas').getContext('2d');

  function measure(text, font) {
    ctx.font = font;
    const m = ctx.measureText(text);
    const asc = m.actualBoundingBoxAscent / R, desc = m.actualBoundingBoxDescent / R;
    const A = m.fontBoundingBoxAscent / R, D = m.fontBoundingBoxDescent / R;
    const h = asc + desc;
    return { w: m.width / R, h, shift: asc - ((h - (A + D)) / 2 + A) };
  }

  function apply(el, fs, m) {
    el.style.fontSize = fs + 'em';
    el.style.lineHeight = String(m.h);
    el.style.top = m.shift + 'em';
  }

  function fit() {
    const nameEl = document.querySelector('.brand-name');
    const orgEl = document.querySelector('.brand-org');
    if (!nameEl || !orgEl) return;
    nameEl.textContent = document.title;
    const n = measure(document.title.toUpperCase(), `400 ${R}px 'Bebas Neue'`);
    const o = measure(orgEl.textContent, `500 ${R}px 'Ubuntu'`);
    const vals = [n.w, n.h, n.shift, o.w, o.h, o.shift];
    if (!n.w || !o.w || !n.h || !o.h || vals.some(v => !Number.isFinite(v))) return;
    const ratio = n.w / o.w;
    const nameFs = (1 - INSET - GAP) / (n.h + o.h * ratio);
    apply(nameEl, nameFs, n);
    apply(orgEl, nameFs * ratio, o);
  }

  fit();
  Promise.all([document.fonts.load("1em 'Bebas Neue'"), document.fonts.load("500 1em 'Ubuntu'")]).then(fit);
})();
