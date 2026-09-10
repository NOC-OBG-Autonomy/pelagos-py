// Thin wrappers around the backend API.
async function _fail(r, msg) {
  throw new Error((await r.json().catch(() => ({}))).detail || msg);
}

const API = {
  async registry() {
    const r = await fetch('/api/registry');
    if (!r.ok) throw new Error('registry failed');
    return r.json();
  },
  async validate(yamlContent) {
    const r = await fetch('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml_content: yamlContent }),
    });
    return r.json();
  },
  // -> {configs, protected, demo, missions, labels, reference, downloaded: [name]}
  async listConfigs() {
    const r = await fetch('/api/configs');
    return r.json();
  },
  async loadConfig(name) {
    const r = await fetch('/api/configs/' + encodeURIComponent(name));
    if (!r.ok) await _fail(r, 'load failed');
    return r.json();
  },
  // -> {path, decisions: [{id, title, detail, options: [{key, label}], default}]}
  async buildDecisions(filePath) {
    const r = await fetch('/api/build/decisions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_path: filePath }),
    });
    if (!r.ok) await _fail(r, 'inspect failed');
    return r.json();
  },
  async build(filePath, choices, description) {
    const r = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_path: filePath, choices, description }),
    });
    if (!r.ok) await _fail(r, 'build failed');
    return r.json();
  },
  async saveConfig(name, yamlContent) {
    const r = await fetch('/api/configs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, yaml_content: yamlContent }),
    });
    if (!r.ok) await _fail(r, 'save failed');
    return r.json();
  },
  async deleteConfig(name) {
    const r = await fetch('/api/configs/' + encodeURIComponent(name), { method: 'DELETE' });
    if (!r.ok) await _fail(r, 'delete failed');
  },
  // Open the configs folder in the OS file browser (server-side, so this only
  // does anything when the dashboard is viewed on the machine running it).
  async revealConfigs() {
    const r = await fetch('/api/configs/reveal', { method: 'POST' });
    if (!r.ok) await _fail(r, 'could not open folder');
    return r.json();
  },
  // -> {name: {done, total}} for demo downloads in flight
  async demoProgress() {
    return (await fetch('/api/demos/progress')).json();
  },
  async deleteDemo(name) {
    const r = await fetch('/api/demos/' + encodeURIComponent(name), { method: 'DELETE' });
    if (!r.ok) await _fail(r, 'delete failed');
  },
  async cleanDemos() {
    const r = await fetch('/api/demos/clean', { method: 'POST' });
    if (!r.ok) await _fail(r, 'delete failed');
    return r.json();
  },
  // -> {dirs, files: [{path, name, dir, kind, size, mtime}]}
  async listOutputs(dirs, inputs) {
    const r = await fetch('/api/outputs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dirs, inputs }),
    });
    if (!r.ok) await _fail(r, 'listing failed');
    return r.json();
  },
  outputUrl(path) {
    return '/api/outputs/file?path=' + encodeURIComponent(path);
  },
  async deleteOutput(path) {
    const r = await fetch(API.outputUrl(path), { method: 'DELETE' });
    if (!r.ok) await _fail(r, 'delete failed');
  },
  async cleanOutputs(dirs, inputs) {
    const r = await fetch('/api/outputs/clean', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dirs, inputs }),
    });
    if (!r.ok) await _fail(r, 'delete failed');
    return r.json();
  },
  async revealOutputs(path) {
    const r = await fetch('/api/outputs/reveal', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: path || '' }),
    });
    if (!r.ok) await _fail(r, 'could not open folder');
    return r.json();
  },
  // Opens a native file dialog on the server; resolves to the path or null.
  async browseFile(start) {
    const r = await fetch('/api/browse', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start: start || '' }),
    });
    if (!r.ok) await _fail(r, 'Browse failed');
    return (await r.json()).path;
  },
  async run(yamlContent) {
    const r = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml_content: yamlContent }),
    });
    if (!r.ok) await _fail(r, 'run failed');
    return r.json();
  },
  async stopRun() {
    await fetch('/api/run/stop', { method: 'POST' });
  },
  async continueRun() {
    await fetch('/api/run/continue', { method: 'POST' });
  },
  async rerunStep(parameters) {
    await fetch('/api/run/rerun', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parameters }),
    });
  },
  async runStatus() {
    return (await fetch('/api/run/status')).json();
  },
};
