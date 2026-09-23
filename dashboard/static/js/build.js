// Build panel: shown in place of the step list when a data file is picked (a
// demo card or Browse…). Lists what the full template must change for that
// file — each with a default the user can override — and generates the whole
// config on Confirm. See pelagos_py.utils.config_builder.
const Build = {
  async start({ name, filePath, description, onCancel }) {
    const root = document.getElementById('build-panel');
    if (!root) return;
    root.innerHTML = '';
    root.hidden = false;
    document.querySelector('.builder').classList.add('building');
    // The old config is still what Run would execute: not what's on screen.
    Build.lockRun(true);
    const file = filePath.split('/').pop();

    const head = document.createElement('div');
    head.className = 'build-head';
    head.innerHTML = `<strong>Set up a pipeline for ${escapeHtml(file)}</strong>`
      + '<span>Inspecting the file…</span>';
    root.appendChild(head);

    let decisions;
    try { ({ decisions } = await API.buildDecisions(filePath)); }
    catch (e) {
      Build.close();
      Config.notice(`Could not inspect ${file}: ${e.message}`, { sticky: true, err: true });
      if (onCancel) onCancel();
      return;
    }
    head.querySelector('span').textContent = decisions.length
      ? 'The full pipeline template is adapted to this file as listed below. Change a choice if needed, then build.'
      : 'The file supports the full pipeline template; nothing needs changing.';

    const choices = {};
    const list = document.createElement('div');
    list.className = 'build-list';
    for (const d of decisions) {
      const row = document.createElement('div');
      row.className = 'build-row' + (d.options.length ? ' choice' : ' auto');
      const text = document.createElement('div');
      text.className = 'build-text';
      text.innerHTML = `<strong>${escapeHtml(d.title)}</strong><span>${escapeHtml(d.detail)}</span>`;
      row.appendChild(text);
      if (d.options.length) {
        const sel = document.createElement('select');
        for (const o of d.options) {
          const opt = document.createElement('option');
          opt.value = o.key;
          opt.textContent = o.label;
          sel.appendChild(opt);
        }
        sel.value = d.default;
        choices[d.id] = d.default;
        const note = document.createElement('div');
        note.className = 'build-changed hidden';
        const defaultLabel = d.options.find((o) => o.key === d.default).label;
        note.innerHTML = `Changed from the default (${escapeHtml(defaultLabel)}) — are you sure? `
          + '<a href="#">Restore default</a>';
        note.querySelector('a').onclick = (e) => { e.preventDefault(); sel.value = d.default; sel.onchange(); };
        sel.onchange = () => {
          choices[d.id] = sel.value;
          const changed = sel.value !== d.default;
          note.classList.toggle('hidden', !changed);
          row.classList.toggle('changed', changed);
        };
        row.appendChild(sel);
        text.appendChild(note);
      } else {
        const tag = document.createElement('span');
        tag.className = 'build-tag';
        tag.textContent = 'automatic';
        row.appendChild(tag);
      }
      list.appendChild(row);
    }
    root.appendChild(list);

    const actions = document.createElement('div');
    actions.className = 'build-actions';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'ghost';
    cancel.textContent = 'Cancel';
    cancel.onclick = () => { Build.close(); if (onCancel) onCancel(); };
    const confirm = document.createElement('button');
    confirm.type = 'button';
    confirm.className = 'primary';
    confirm.innerHTML = Icon.svg('check') + 'Build pipeline';
    confirm.onclick = async () => {
      confirm.disabled = cancel.disabled = true;
      try {
        const { yaml_content } = await API.build(filePath, choices, description);
        Build.close();
        Config.apply(yaml_content);
        if (name) Config.setCurrent(name);
        else Config.noteEdit(); // built over a locked config: fork it
      } catch (e) {
        confirm.disabled = cancel.disabled = false;
        Config.notice(`Could not build the config: ${e.message}`, { sticky: true, err: true });
      }
    };
    actions.appendChild(cancel);
    actions.appendChild(confirm);
    root.appendChild(actions);
    confirm.focus();
  },

  close() {
    const root = document.getElementById('build-panel');
    if (root) { root.hidden = true; root.innerHTML = ''; }
    document.querySelector('.builder').classList.remove('building');
    Build.lockRun(false);
  },

  active: false,
  lockRun(on) {
    Build.active = on;
    const btn = document.getElementById('btn-run');
    btn.disabled = on || RunLock.running;
    btn.title = on ? 'Build the pipeline first' : '';
  },
};
