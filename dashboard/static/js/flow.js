// Flowchart view of the pipeline (Flow in the builder bar): containers holding
// colour-coded step chips, snaking across columns, wired step to step. Reads the
// same STATE as the list view; a chip opens its editor in a dialog.
const Flow = {
  overlay: null,
  dialogFor: null, // id of the step open in the dialog

  kind(item) {
    if (isQcContainer(item.def)) return 'manual qc' in (item.values.qc_settings || {}) ? 'manual' : 'qc';
    return item.def.category === 'input_output' ? 'io' : item.def.category === 'quality_control' ? 'qc' : 'proc';
  },

  cols: 0,
  CELL: 236, GAP: 64, ROW_GAP: 64,

  render(host) {
    host.className = 'fc';
    const s = STATE.pipeline.settings || {};
    const head = document.createElement('div');
    head.className = 'fc-head';
    head.innerHTML = `<div class="fc-title">${Flow.esc(s.name || 'Pipeline')}</div>` +
      (s.description ? `<div class="fc-desc">${Flow.esc(s.description)}</div>` : '');
    host.appendChild(head);

    // The sequence of cells (stages, loose steps, ghosts), each
    // with the link that precedes it when it follows a real node.
    const seq = [];
    for (const g of defaultsView.sectionsFirst) seq.push({ el: Flow.ghostStage(g) });
    for (const g of defaultsView.first) seq.push({ el: Flow.ghostChip(g, null, null, true) });
    const nodes = STATE.pipeline.nodes;
    nodes.forEach((node, i) => {
      seq.push({ el: isSection(node) ? Flow.stage(node) : Flow.chipCell(node), link: i ? [nodes, i] : null });
      for (const g of defaultsView.sectionsAfter.get(node.id) || []) seq.push({ el: Flow.ghostStage(g) });
    });

    const cols = Flow.cols = Math.max(1, Math.floor((host.clientWidth + Flow.GAP) / (Flow.CELL + Flow.GAP)));
    const row = document.createElement('div');
    row.className = 'fc-row';
    row.style.width = (cols * Flow.CELL + (cols - 1) * Flow.GAP) + 'px';
    for (const c of seq) row.appendChild(c.el);
    host.appendChild(row);
    Flow.layout(row, seq, cols);
    Flow.legend();
    Flow.watch(host);
    if (Flow.dialogFor != null) Flow.refreshDialog();
  },

  // Each cell goes straight below the previous one or beside it in a
  // neighbouring column, whichever lands higher on the page, and never above
  // what its column already holds: short containers stack, tall ones sit alone.
  layout(row, seq, cols) {
    const bottom = new Array(cols).fill(-Flow.ROW_GAP);
    let prev = null, height = 0;
    for (const c of seq) {
      c.h = c.el.offsetHeight;
      let best = { col: 0, y: 0 };
      if (prev) {
        best = null;
        for (const col of [prev.col, prev.col + 1, prev.col - 1]) {
          if (col < 0 || col >= cols) continue;
          const y = col === prev.col ? bottom[col] + Flow.ROW_GAP : Math.max(prev.y, bottom[col] + Flow.ROW_GAP);
          if (!best || y < best.y) best = { col, y };
        }
      }
      Object.assign(c, best);
      c.x = c.col * (Flow.CELL + Flow.GAP);
      c.el.style.left = c.x + 'px';
      c.el.style.top = c.y + 'px';
      bottom[c.col] = c.y + c.h;
      height = Math.max(height, c.y + c.h);
      prev = c;
    }
    row.style.height = height + Flow.ROW_GAP + 'px';
    Flow.wire(row, seq);
  },

  // Wires, drawn once the cells have a size. The full run of each wire sits
  // under the cells (so it never crosses a container's heading) and only the
  // stubs inside a container's padding are drawn over it, chip edge to edge.
  // Hovering a wire between containers offers three insert points along it:
  // end of the previous container, between the two, start of the next. A wire
  // into a step already run is lit; the one into the executing step flows.
  wire(row, seq) {
    const NS = 'http://www.w3.org/2000/svg';
    const R = row.getBoundingClientRect();
    const box = (el) => {
      const b = el.getBoundingClientRect();
      return { l: b.left - R.left, t: b.top - R.top, r: b.right - R.left, b: b.bottom - R.top,
        cx: b.left + b.width / 2 - R.left, cy: b.top + b.height / 2 - R.top };
    };
    const layer = (cls, first) => {
      const svg = document.createElementNS(NS, 'svg');
      svg.setAttribute('class', 'fc-wires ' + cls);
      svg.setAttribute('width', row.offsetWidth);
      svg.setAttribute('height', row.offsetHeight);
      first ? row.prepend(svg) : row.appendChild(svg);
      return svg;
    };
    const under = layer('under', true), over = layer('over', false);
    const node = (svg, tag, attrs) => {
      const el = document.createElementNS(NS, tag);
      for (const k in attrs) el.setAttribute(k, attrs[k]);
      svg.appendChild(el);
      return el;
    };
    const chips = [...row.querySelectorAll('.fc-chip[data-step-id]')];
    const cur = RunLock.running ? (RunLock.runningIndex ?? RunLock.index) : null;
    const flowing = RunLock.runningIndex != null;
    const HEAD = { down: (x, y) => `${x},${y} ${x - 5},${y - 7} ${x + 5},${y - 7}`,
      right: (x, y) => `${x},${y} ${x - 7},${y - 5} ${x - 7},${y + 5}`,
      left: (x, y) => `${x},${y} ${x + 7},${y - 5} ${x + 7},${y + 5}` };
    const draw = ({ d, stubs = [], dir, tip, into, ports = [], cls = '' }) => {
      if (cur != null && into != null && into <= cur) cls += ' done';
      if (cur != null && into === cur && flowing) cls += ' flow';
      cls = cls.trim();
      if (d) node(under, 'path', { d, class: cls });
      for (const sd of stubs) node(over, 'path', { d: sd, class: cls });
      if (dir) node(over, 'polygon', { points: HEAD[dir](...tip), class: cls });
      if (!ports.length) return;
      const els = ports.map((p) => {
        const el = document.createElement('div');
        el.className = 'fc-port';
        el.style.left = p.x + 'px'; el.style.top = p.y + 'px';
        el.appendChild(Flow.plus(el, p.list, p.index, p.title));
        row.appendChild(el);
        return el;
      });
      const hot = (on) => () => els.forEach((el) => el.classList.toggle('hot', on));
      for (const el of [node(under, 'path', { d, class: 'fc-hit' }), ...els]) {
        el.addEventListener('mouseenter', hot(true));
        el.addEventListener('mouseleave', hot(false));
      }
    };
    const steps = (c) => (c.el._sec ? c.el._sec.steps : null);

    // Chip to chip inside a container: straight down across the spacer.
    for (const sp of row.querySelectorAll('.fc-arrow')) {
      const a = sp.previousElementSibling, b = sp.nextElementSibling;
      if (!a || !b) continue;
      const A = box(a), B = box(b), into = chips.indexOf(b);
      draw({ stubs: [`M${A.cx} ${A.b} V${B.t}`], dir: 'down', tip: [B.cx, B.t], into: into < 0 ? null : into });
    }
    // Cell to cell: an elbow through the gap between columns, or a drop.
    seq.forEach((c, j) => {
      if (!c.link || !j) return;
      const prev = seq[j - 1];
      const a = [...prev.el.querySelectorAll('.fc-chip[data-step-id]')].pop();
      const b = c.el.querySelector('.fc-chip[data-step-id]');
      if (!a || !b) return;
      const A = box(a), B = box(b), CA = box(prev.el), CB = box(c.el);
      const into = chips.indexOf(b);
      const sa = steps(prev), sb = steps(c);
      const ports = [], P = 8; // insert points sit this far outside a container's edge
      const at = (x, y, list, index, title) => ports.push({ x, y, list, index, title });
      if (c.col === prev.col) {
        // A drop ends at the container's top edge rather than piercing its heading.
        const yb = sb ? CB.t : B.t, my = (CA.b + yb) / 2;
        const d = B.cx === A.cx ? `M${A.cx} ${A.b} V${yb}` : `M${A.cx} ${A.b} V${my} H${B.cx} V${yb}`;
        if (sa) at(A.cx, CA.b + P, sa, sa.length, 'Add a step at the end of this container');
        at(A.cx, my, ...c.link, 'Insert a step between the containers');
        if (sb) at(B.cx, yb - P, sb, 0, 'Add a step at the start of this container');
        draw({ d, stubs: [`M${A.cx} ${A.b} V${CA.b}`], dir: 'down', tip: [B.cx, yb], into, ports });
      } else {
        // Rightward and leftward wires take different lanes so neighbours never merge.
        const right = c.col > prev.col;
        const mx = Math.min(prev.x, c.x) + Flow.CELL + Flow.GAP / 2 + (right ? -8 : 8);
        const x0 = right ? A.r : A.l, x1 = right ? B.l : B.r;
        const ea = right ? CA.r : CA.l, eb = right ? CB.l : CB.r, s = right ? 1 : -1;
        if (sa) at(ea + s * P, A.cy, sa, sa.length, 'Add a step at the end of this container');
        at(mx, (A.cy + B.cy) / 2, ...c.link, 'Insert a step between the containers');
        if (sb) at(eb - s * P, B.cy, sb, 0, 'Add a step at the start of this container');
        draw({ d: `M${x0} ${A.cy} H${mx} V${B.cy} H${x1}`, stubs: [`M${x0} ${A.cy} H${ea}`, `M${eb} ${B.cy} H${x1}`],
          dir: right ? 'right' : 'left', tip: [x1, B.cy], into, ports });
      }
    });
    // The last container trails off into a short stub carrying its append point.
    const last = seq[seq.length - 1];
    const a = last && [...last.el.querySelectorAll('.fc-chip[data-step-id]')].pop();
    if (a && steps(last)) {
      const A = box(a), CA = box(last.el), sa = steps(last);
      draw({ d: `M${A.cx} ${A.b} V${CA.b + 30}`, stubs: [`M${A.cx} ${A.b} V${CA.b}`], cls: 'tail',
        ports: [{ x: A.cx, y: CA.b + 16, list: sa, index: sa.length, title: 'Add a step at the end of this container' }] });
    }
  },

  // Re-lay the snake when the pane's width changes the column count.
  watch(host) {
    if (Flow.ro) return;
    Flow.ro = new ResizeObserver(() => {
      if (viewMode !== 'flow' || !host.isConnected) return;
      const cols = Math.max(1, Math.floor((host.clientWidth + Flow.GAP) / (Flow.CELL + Flow.GAP)));
      if (cols !== Flow.cols) Flow.render(host);
    });
    Flow.ro.observe(host);
  },

  // Where a dragged step or container would land: beside the slot.
  placeIndicator(ind, at, after) {
    Object.assign(ind.style, { left: (at.offsetLeft + (after ? Flow.CELL + 12 : -15)) + 'px', top: at.offsetTop + 'px', height: at.offsetHeight + 'px' });
  },

  stage(sec) {
    const el = document.createElement('div');
    el.className = 'fc-stage';
    el._sec = sec;
    const head = document.createElement('div');
    head.className = 'fc-stage-head';
    head.innerHTML = `<span class="drag reveal" title="Drag to move the container">${Icon.svg('grip', 14)}</span>`;
    const title = document.createElement('input');
    title.className = 'fc-stage-name';
    title.value = sec.title; title.placeholder = 'Container name';
    title.oninput = () => { sec.title = title.value; STATE.onChange(); };
    head.appendChild(title);
    head.appendChild(Forms.button('', { icon: 'close', iconSize: 13, cls: 'icon-btn reveal fc-x', title: 'remove container and its steps', onclick: () => removeSection(sec.id) }));
    Flow.dragHandle(el, head.querySelector('.drag'), { kind: 'section', id: sec.id });
    el.appendChild(head);

    const body = document.createElement('div');
    body.className = 'section-body fc-stage-body';
    body.dataset.secId = String(sec.id);
    for (const g of defaultsView.sectionFirst.get(sec.id) || []) body.appendChild(Flow.ghostChip(g, null, sec.id));
    sec.steps.forEach((item, i) => {
      if (i) body.appendChild(Flow.arrow(sec.steps, i));
      body.appendChild(Flow.chip(item));
      for (const g of defaultsView.after.get(item.id) || []) body.appendChild(Flow.ghostChip(g, item.id));
    });
    el.appendChild(body);
    return el;
  },

  // A loose step outside any container sits in its own cell.
  chipCell(item) {
    const el = document.createElement('div');
    el.className = 'fc-cell';
    el.appendChild(Flow.chip(item));
    for (const g of defaultsView.after.get(item.id) || []) el.appendChild(Flow.ghostChip(g, item.id));
    return el;
  },

  chip(item) {
    const index = STATE.pipeline.items.indexOf(item);
    const el = document.createElement('div');
    el.className = 'fc-chip fc-' + Flow.kind(item);
    el.dataset.stepId = item.id;
    const info = defaultsView.byId.get(item.id);
    const extra = !!(info && info.extra);
    const edited = !extra && !!(info && info.diff && info.diff.count);
    if (extra) el.classList.add('extra');
    const sub = stepSummary(item);
    el.innerHTML = `<div class="fc-name">${Flow.esc(item.name)}</div>` +
      (sub ? `<div class="fc-sub">${Flow.esc(sub)}</div>` : '') +
      (extra ? '<span class="tag ok fc-tag">added</span>' : edited ? '<span class="tag warn fc-tag">edited</span>' : '');
    if (RunLock.running) {
      el.classList.toggle('locked', !RunLock.editable(index));
      el.classList.toggle('running', index === RunLock.runningIndex);
      el.classList.toggle('unlocked', RunLock.editable(index));
    } else {
      el.appendChild(Forms.button('', { icon: 'close', iconSize: 13, cls: 'icon-btn reveal fc-x', title: 'remove', onclick: (e) => { e.stopPropagation(); removeStep(item.id); } }));
      Flow.dragHandle(el, el, { kind: 'move', id: item.id });
    }
    el.addEventListener('mousedown', () => highlightYamlForStep(item.id));
    el.addEventListener('click', (e) => {
      if (e.target.closest('button')) return;
      if (RunLock.editable(index)) Flow.openDialog(item);
    });
    return el;
  },

  // A default step this pipeline lacks, offering it back (see Defaults).
  ghostChip(base, anchorId, sectionId, loose = false) {
    const el = document.createElement('div');
    el.className = 'fc-chip fc-ghost' + (loose ? ' fc-cell' : '');
    el.innerHTML = `<div class="fc-name">${Flow.esc(base.name)}</div><div class="fc-sub">removed from default</div>`;
    el.appendChild(Defaults.addButton('Add back', () => Defaults.addStepBack(base, anchorId, sectionId)));
    return el;
  },

  ghostStage(ghost) {
    const el = document.createElement('div');
    el.className = 'fc-stage fc-ghost-stage';
    el.appendChild(Defaults.ghostSection(ghost));
    return el;
  },

  // Down arrow between two chips, carrying the insert point on hover.
  arrow(list, index) {
    const el = document.createElement('div');
    el.className = 'fc-arrow';
    el.appendChild(Flow.plus(el, list, index));
    return el;
  },

  plus(anchor, list, index, title = 'Insert a step here') {
    return Forms.button('', { icon: 'plus', iconSize: 12, cls: 'fc-plus', title,
      onclick: (e) => { e.stopPropagation(); anchor.classList.add('open'); openStepPicker(anchor, list, index); } });
  },

  legend() {
    document.getElementById('flow-legend').innerHTML = '<span class="fc-legend-title">Step type</span>' +
      [['io', 'Load / export'], ['proc', 'Processing / derivation'], ['qc', 'Quality control'], ['manual', 'Manual (human-in-the-loop) QC']]
        .map(([k, t]) => `<span class="fc-legend-item"><i class="fc-swatch fc-${k}"></i>${t}</span>`).join('') +
      '<span class="fc-legend-item"><i class="fc-swatch fc-extra"></i>Added to this pipeline</span>';
  },

  // `handle` starts a drag of `el`; the picker's palette items drive the same
  // dragState, so the list view's drop logic serves both.
  dragHandle(el, handle, state) {
    handle.addEventListener('mousedown', (e) => { if (!e.target.closest('input,button')) el.draggable = true; });
    handle.addEventListener('mouseup', () => { el.draggable = false; });
    el.addEventListener('dragstart', (e) => {
      dragState = state;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', String(state.id));
      el.classList.add('dragging');
      e.stopPropagation();
    });
    el.addEventListener('dragend', () => {
      el.draggable = false; el.classList.remove('dragging'); clearDropIndicator(); dragState = null;
    });
  },

  // ---- editor dialog: the list view's step card, floated over the builder pane.
  openDialog(item) {
    Flow.closeDialog();
    const ov = document.createElement('div');
    ov.className = 'fc-overlay';
    ov.addEventListener('mousedown', (e) => { if (e.target === ov) Flow.closeDialog(); });
    document.querySelector('.builder').appendChild(ov);
    Flow.overlay = ov;
    Flow.dialogFor = item.id;
    Flow.refreshDialog();
  },

  refreshDialog() {
    const item = STATE.pipeline.items.find((i) => i.id === Flow.dialogFor);
    if (!item || !Flow.overlay) { Flow.closeDialog(); return; }
    item.collapsed = false;
    Flow.overlay.innerHTML = '';
    const box = document.createElement('div');
    box.className = 'fc-dialog';
    const card = renderStepCard(item);
    box.appendChild(card);
    Flow.overlay.appendChild(box);
    const close = card.querySelector('.step-del');
    if (close) { close.title = 'close'; close.onclick = (e) => { e.stopPropagation(); Flow.closeDialog(); }; }
    if (RunLock.running) lockCard(card, STATE.pipeline.items.indexOf(item));
  },

  closeDialog() {
    if (!Flow.overlay) return;
    const item = STATE.pipeline.items.find((i) => i.id === Flow.dialogFor);
    if (item) item.collapsed = true;
    Flow.overlay.remove();
    Flow.overlay = null;
    Flow.dialogFor = null;
  },

  esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); },
};
