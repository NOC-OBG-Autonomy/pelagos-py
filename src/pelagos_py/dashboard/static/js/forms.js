// Schema-driven form fields from pelagos_py.utils.parameter_spec.describe, so any step renders without step-specific code.

const Forms = {
  types(spec) {
    const t = spec.type;
    if (t == null) return [];
    return Array.isArray(t) ? t : [t];
  },

  kind(spec) {
    const ts = Forms.types(spec);
    if (spec.options) return ts.includes('list') ? 'multiselect' : 'select';
    if (ts.length === 1 && ts[0] === 'bool') return 'bool';
    if (ts.some((t) => ['dict', 'list', 'tuple'].includes(t))) return 'yaml';
    if (ts.length === 1 && (ts[0] === 'int' || ts[0] === 'float')) return 'number';
    if (ts.length === 1 && ts[0] === 'str') return 'text';
    // unions / unknown -> YAML mini-editor
    return 'yaml';
  },

  defaultValue(spec) {
    if ('default' in spec) return Forms.clone(spec.default);
    switch (Forms.kind(spec)) {
      case 'bool': return false;
      case 'number': return null;
      case 'select': return spec.options[0];
      case 'multiselect': return [];
      case 'yaml': return null;
      default: return '';
    }
  },

  clone(v) { return v == null ? v : structuredClone(v); }, // JSON would turn ±Infinity (YAML .inf) into null

  _uid: 0,

  el(tag, props = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k in e) e[k] = v;
      else e.setAttribute(k, v);
    }
    e.append(...children.filter((c) => c != null));
    return e;
  },

  button(label, { icon, iconSize = 16, cls = '', ...props } = {}) {
    const b = Forms.el('button', { type: 'button', class: cls, html: icon ? Icon.svg(icon, iconSize) : '', ...props });
    b.append(label);
    return b;
  },

  // A placeholder is an empty-valued first option.
  select(options, value, onChange, { placeholder, cls = '' } = {}) {
    const sel = Forms.el('select', { class: cls });
    if (placeholder) sel.appendChild(Forms.el('option', { value: '', textContent: placeholder }));
    for (const o of options) {
      const [v, label] = Array.isArray(o) ? o : [o, o];
      sel.appendChild(Forms.el('option', { value: String(v), textContent: String(label), selected: v === value }));
    }
    if (onChange) sel.onchange = () => onChange(sel.value, sel);
    return sel;
  },

  seg(options, value, onChange, cls = '') {
    const seg = Forms.el('div', { class: 'seg ' + cls });
    const btns = options.map(([v, label]) => Forms.button(label, { onclick: () => { seg.set(v); onChange(v); } }));
    seg.append(...btns);
    seg.set = (v) => btns.forEach((b, i) => b.classList.toggle('on', options[i][0] === v));
    seg.set(value);
    return seg;
  },

  // Returns { el, input }: the <span class="switch"> and its checkbox.
  switchEl(checked, onChange) {
    const el = document.createElement('span');
    el.className = 'switch';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = !!checked;
    const slider = document.createElement('span');
    slider.className = 'slider';
    el.appendChild(input);
    el.appendChild(slider);
    if (onChange) input.onchange = () => onChange(input.checked);
    return { el, input };
  },

  typeLabel(spec) {
    const ts = Forms.types(spec);
    const base = ts.length ? ts.join(' | ') : 'any';
    return spec.unit ? `${base} · ${spec.unit}` : base;
  },

  // onChange() is called (no args) after any edit.
  render(spec, values, onChange) {
    const kind = Forms.kind(spec);
    const wrap = document.createElement('div');
    wrap.className = 'field' + (kind === 'bool' ? ' checkbox' : '');
    wrap.dataset.param = spec.name;

    const label = document.createElement('label');
    label.textContent = spec.name;
    if (spec.required) {
      const req = document.createElement('span');
      req.className = 'req'; req.textContent = '*';
      label.appendChild(req);
    }
    const typeTag = document.createElement('span');
    typeTag.className = 'type-tag';
    typeTag.textContent = Forms.typeLabel(spec);
    label.appendChild(typeTag);

    let input;
    let boolSwitch = null;
    const cur = spec.name in values ? values[spec.name] : Forms.defaultValue(spec);

    if (kind === 'select') {
      input = Forms.select(spec.options, cur, (v, sel) => { values[spec.name] = spec.options[sel.selectedIndex]; onChange(); });
    } else if (kind === 'multiselect') {
      input = document.createElement('div');
      input.className = 'multiselect';
      const chosen = new Set(Array.isArray(cur) ? cur : []);
      for (const opt of spec.options) {
        const item = document.createElement('label');
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = chosen.has(opt);
        box.onchange = () => {
          box.checked ? chosen.add(opt) : chosen.delete(opt);
          values[spec.name] = spec.options.filter((o) => chosen.has(o)); // keep schema order
          onChange();
        };
        item.appendChild(box);
        item.appendChild(document.createTextNode(String(opt)));
        input.appendChild(item);
      }
    } else if (kind === 'bool') {
      const sw = Forms.switchEl(!!cur, (v) => { values[spec.name] = v; onChange(); });
      input = sw.input;
      input.id = 'f-' + (++Forms._uid);
      label.htmlFor = input.id;
      label.classList.add('switch-label');
      boolSwitch = sw.el;
    } else if (kind === 'number') {
      input = document.createElement('input');
      input.type = 'number';
      if (spec.min != null) input.min = spec.min;
      if (spec.max != null) input.max = spec.max;
      if (spec.step != null) input.step = spec.step;
      if (cur != null) input.value = cur;
      input.onchange = () => {
        values[spec.name] = input.value === '' ? null : Number(input.value);
        onChange();
      };
    } else if (kind === 'yaml') {
      input = document.createElement('textarea');
      input.spellcheck = false;
      input.value = cur == null ? '' : Forms.dump(cur).trimEnd();
      input.placeholder = 'YAML / JSON value';
      input.onchange = () => {
        const txt = input.value.trim();
        if (txt === '') { values[spec.name] = null; onChange(); return; }
        try {
          values[spec.name] = jsyaml.load(txt);
          input.style.borderColor = '';
        } catch (e) {
          input.style.borderColor = 'var(--danger)';
        }
        onChange();
      };
    } else {
      input = document.createElement('input');
      input.type = 'text';
      if (cur != null) input.value = cur;
      input.onchange = () => { values[spec.name] = input.value; onChange(); };
    }

    if (kind === 'bool') {
      wrap.appendChild(boolSwitch);
      wrap.appendChild(label);
    } else if (spec.name === 'file_path' && input.type === 'text') {
      // The dashboard is local-only, so the path the server-side dialog returns is usable as-is.
      const row = document.createElement('div');
      row.className = 'file-row';
      input.placeholder = 'Path to your input NetCDF file';
      input.onchange = () => { values[spec.name] = input.value; syncOutputPath(input.value); onChange(); };
      const browse = Forms.button('Browse…', { icon: 'folder' });
      browse.onclick = async () => {
        browse.disabled = true;
        try {
          const path = await API.browseFile(input.value);
          if (path) {
            input.value = path; values[spec.name] = path; syncOutputPath(path); onChange();
            Build.start({ name: null, filePath: path }); // cancel keeps just the path
          }
        } catch (e) { alert(e.message); }
        finally { browse.disabled = false; }
      };
      row.appendChild(input);
      row.appendChild(browse);
      wrap.appendChild(label);
      wrap.appendChild(row);
    } else if (kind === 'yaml' && Array.isArray(cur) && cur.length > 3) {
      // Long lists (manual QC boxes) fold behind a count.
      const details = document.createElement('details');
      details.className = 'yaml-fold';
      const summary = document.createElement('summary');
      summary.textContent = `${cur.length} entries`;
      details.appendChild(summary);
      details.appendChild(input);
      wrap.appendChild(label);
      wrap.appendChild(details);
    } else {
      wrap.appendChild(label);
      wrap.appendChild(input);
    }

    if (spec.description) {
      const hint = document.createElement('div');
      hint.className = 'hint';
      hint.textContent = spec.description;
      wrap.appendChild(hint);
    }
    return wrap;
  },

  equal(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
  },

  // js-yaml would render `[20, 45]` as a block list; keep scalar lists inline like the hand-written configs.
  _isScalar(x) { return x === null || typeof x !== 'object'; },

  _scalarText(x) { return jsyaml.dump(x, { flowLevel: 0, lineWidth: -1 }).trimEnd(); },

  // One item per line, so a config with many manual QC boxes stays readable.
  _FLOW_KEYS: new Set(['boxes']),

  _flowMap(m) {
    const parts = Object.keys(m).map((k) => `${Forms._scalarText(k)}: ${Forms._emit(m[k], '')}`);
    return '{' + parts.join(', ') + '}';
  },

  _emit(v, indent) {
    if (Array.isArray(v)) {
      if (v.length === 0) return '[]';
      if (v.every(Forms._isScalar)) return '[' + v.map(Forms._scalarText).join(', ') + ']';
      const parts = v.map((item) => {
        const c = Forms._emit(item, indent + '  ');
        // Graft a block child's first line onto the `- ` marker so continuation lines stay aligned.
        if (c[0] === '\n') return indent + '- ' + c.slice(1 + indent.length + 2);
        return indent + '- ' + c;
      });
      return '\n' + parts.join('\n');
    }
    if (v && typeof v === 'object') {
      const keys = Object.keys(v);
      if (keys.length === 0) return '{}';
      const parts = keys.map((k) => {
        const flow = Forms._FLOW_KEYS.has(k) && Array.isArray(v[k]) && v[k].length &&
          v[k].every((m) => m && typeof m === 'object' && !Array.isArray(m) &&
            Object.values(m).every((x) => Forms._isScalar(x) || (Array.isArray(x) && x.every(Forms._isScalar))));
        const c = flow ? '\n' + v[k].map((m) => indent + '  - ' + Forms._flowMap(m)).join('\n')
          : Forms._emit(v[k], indent + '  ');
        // Integer keys (QC flags) stay unquoted so the pipeline reads them as ints.
        const key = /^-?\d+$/.test(k) ? k : Forms._scalarText(k);
        return c[0] === '\n' ? indent + key + ':' + c : indent + key + ': ' + c;
      });
      return '\n' + parts.join('\n');
    }
    return Forms._scalarText(v);
  },

  dump(v) {
    if (v === null || v === undefined) return '';
    const s = Forms._emit(v, '');
    return (s[0] === '\n' ? s.slice(1) : s) + '\n';
  },
};
