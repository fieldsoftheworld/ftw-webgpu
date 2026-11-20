import './style.css';

import { type Chip, type ChipIndex, loadChip, loadIndex, pickRandom, splitOf } from './chips.ts';
import { type Scores, score } from './metrics.ts';
import { drawClasses, drawRGB, stretchRange } from './render.ts';
import { planTiles, stackWindows } from './tiling.ts';
import type { Backend, WorkerRequest, WorkerResponse } from './worker.ts';

const BASE = import.meta.env.BASE_URL;
const HF = 'https://huggingface.co/fieldsoftheworld/ftw-prue-ccby/resolve/ffe4452768e1054b1d7c2ab39b84b4170f90c7cf';
const MODELS: Record<string, { name: string; url: string }> = {
  b3: { name: 'B3', url: `${HF}/prue_efnetb3_ccby-b35d59a6.onnx` },
  b5: { name: 'B5', url: `${HF}/prue_efnetb5_ccby-4fd92b24.onnx` },
  b7: { name: 'B7', url: `${HF}/prue_efnetb7_ccby-4bfbbbde.onnx` },
};

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const num = (id: string) => Number($<HTMLInputElement>(id).value);

const state = {
  country: 'any',
  split: 'test',
  model: 'b3',
  backend: 'auto' as Backend | 'auto',
  divisor: 3000,
  order: 'ba',
  inputSize: 256,
  patch: 256,
  overlap: 25,
  opacity: 70,
  showField: true,
  showBoundary: true,
  base: 'b' as 'a' | 'b',
  view: 'pred' as 'pred' | 'labels',
};

let index: ChipIndex | null = null;
let chip: Chip | null = null;
let ranges: Record<'a' | 'b', [number, number]> = { a: [0, 3000], b: [0, 3000] };
let modelReady = false;
let modelInfo = '';
let runId = 0;
let chipToken = 0;
let last: { classes: Uint8Array; size: number; scores: Scores; ms: number; tiles: number } | null = null;

// Exposed for end-to-end tests.
declare global {
  interface Window { __ftw: { ready: boolean; runs: number; last: typeof last; chip: () => Chip | null } }
}
window.__ftw = { ready: false, runs: 0, last: null, chip: () => chip };

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
const send = (msg: WorkerRequest, transfer: Transferable[] = []) => worker.postMessage(msg, transfer);

// ---------- status helpers ----------
function status(text: string, error = false) {
  const el = $('status');
  el.textContent = text;
  el.classList.toggle('error', error);
}
function stageMsg(text: string) {
  $('stage-msg').textContent = text;
}
function progress(frac: number | null) {
  const el = $('progress');
  el.hidden = frac === null;
  if (frac !== null) (el.firstElementChild as HTMLElement).style.width = `${Math.round(frac * 100)}%`;
}
const mb = (b: number) => (b / 1e6).toFixed(1);
const pct = (v: number | null) => (v === null ? '–' : `${(v * 100).toFixed(1)}%`);

// ---------- URL hash state ----------
function readHash(): { country?: string; id?: string } {
  const p = new URLSearchParams(location.hash.slice(1));
  const intIn = (k: string, lo: number, hi: number, fallback: number) => {
    const v = Number(p.get(k));
    return p.has(k) && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
  };
  state.model = MODELS[p.get('model') ?? ''] ? p.get('model')! : state.model;
  state.divisor = intIn('div', 500, 10000, state.divisor);
  state.order = p.get('order') === 'ab' ? 'ab' : 'ba';
  const backend = p.get('backend');
  if (backend === 'webgpu' || backend === 'wasm') state.backend = backend;
  state.inputSize = intIn('size', 128, 1024, state.inputSize);
  state.patch = intIn('patch', 64, 1024, state.patch);
  state.overlap = intIn('ov', 0, 50, state.overlap);
  return { country: p.get('c') ?? undefined, id: p.get('id') ?? undefined };
}
function writeHash() {
  const p = new URLSearchParams();
  if (chip) {
    p.set('c', chip.country);
    p.set('id', chip.id);
  }
  p.set('model', state.model);
  p.set('div', String(state.divisor));
  if (state.order !== 'ba') p.set('order', state.order);
  if (state.backend !== 'auto') p.set('backend', state.backend);
  p.set('size', String(state.inputSize));
  p.set('patch', String(state.patch));
  p.set('ov', String(state.overlap));
  history.replaceState(null, '', `#${p}`);
}

// ---------- rendering ----------
function renderBase() {
  if (!chip) return;
  drawRGB($<HTMLCanvasElement>('base'), state.base === 'a' ? chip.windowA : chip.windowB, chip.size, ranges[state.base]);
  document.querySelectorAll<HTMLElement>('.thumb').forEach((t) => t.classList.toggle('on', t.dataset.base === state.base));
}

function shownClasses() {
  const s = new Set<number>();
  if (state.showField) s.add(1);
  if (state.showBoundary) s.add(2);
  return s;
}

function renderOverlay() {
  const pred = $<HTMLCanvasElement>('pred');
  const labels = $<HTMLCanvasElement>('labels');
  const op = String(state.opacity / 100);
  pred.style.opacity = op;
  labels.style.opacity = op;
  pred.hidden = state.view !== 'pred';
  labels.hidden = state.view !== 'labels';
  if (state.view === 'pred' && last) drawClasses(pred, last.classes, last.size, shownClasses());
  if (state.view === 'labels' && chip?.mask) drawClasses(labels, chip.mask, chip.size, shownClasses());
  if (state.view === 'labels' && chip && !chip.mask) stageMsg('No label mask for this chip');
}

function renderStats() {
  const s = last?.scores;
  $('s-field').textContent = pct(s?.fieldIoU ?? null);
  $('s-boundary').textContent = pct(s?.boundaryIoU ?? null);
  $('s-acc').textContent = pct(s?.accuracy ?? null);
  $('s-ms').textContent = last ? `${last.ms < 1000 ? Math.round(last.ms) + ' ms' : (last.ms / 1000).toFixed(2) + ' s'}` : '–';
  $('s-tiles').textContent = last ? String(last.tiles) : '–';
  $('s-frac').textContent = s ? `${pct(s.fractions[1])} / ${pct(s.fractions[2])}` : '–';
}

function renderChipMeta() {
  if (!chip || !index) return;
  const c = index.countries[chip.country];
  const [w, s, e, n] = chip.bbox;
  const lat = (s + n) / 2;
  const lon = (w + e) / 2;
  const split = splitOf(index, chip.country, chip.id) ?? 'unknown';
  const osm = `https://www.openstreetmap.org/?mlat=${lat.toFixed(5)}&mlon=${lon.toFixed(5)}#map=14/${lat.toFixed(5)}/${lon.toFixed(5)}`;
  const meta = $('chip-meta');
  meta.replaceChildren();
  const h = document.createElement('h3');
  h.textContent = `${c.name} · ${chip.id}`;
  const tags = document.createElement('p');
  for (const [cls, text] of [['split', split], ['', `labels ${c.year}`], ['', c.license]]) {
    const t = document.createElement('span');
    t.className = `tag ${cls}`;
    t.textContent = text;
    tags.append(t);
  }
  const loc = document.createElement('p');
  const a = document.createElement('a');
  a.href = osm;
  a.target = '_blank';
  a.rel = 'noopener';
  a.textContent = `${lat.toFixed(4)}°, ${lon.toFixed(4)}°`;
  loc.append(a);
  meta.append(h, tags, loc);
  $('dates-b').textContent = `${c.window_b[0]} → ${c.window_b[1]}`;
  $('dates-a').textContent = `${c.window_a[0]} → ${c.window_a[1]}`;
}

function renderSliderLabels() {
  $('divisor-out').textContent = String(state.divisor);
  const scale = state.inputSize / (chip?.size ?? 256);
  $('size-out').textContent = `${state.inputSize} px · ${scale.toFixed(2)}×`;
  const plan = planTiles(state.inputSize, state.patch, state.overlap / 100);
  const tiles = plan.ys.length * plan.xs.length;
  const clamped = plan.patch < state.patch ? ` (using ${plan.patch})` : '';
  $('patch-out').textContent = `${state.patch} px · ${tiles} tile${tiles > 1 ? 's' : ''}${clamped}`;
  $('overlap-out').textContent = `${state.overlap}%`;
  $('opacity-out').textContent = `${state.opacity}%`;
}

// ---------- inference ----------
function infer() {
  renderSliderLabels();
  writeHash();
  if (!modelReady || !chip) return;
  const [first, second] = state.order === 'ba' ? [chip.windowB, chip.windowA] : [chip.windowA, chip.windowB];
  const image = stackWindows(first, second, state.divisor);
  runId++;
  $('stage').classList.add('busy');
  send({
    type: 'run', id: runId, image, bands: 8, size: chip.size,
    inputSize: state.inputSize, patch: state.patch, overlap: state.overlap / 100,
  }, [image.buffer]);
}

worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
  const msg = ev.data;
  switch (msg.type) {
    case 'progress':
      if (msg.cached) status(`Loading ${MODELS[state.model].name} from browser cache…`);
      else status(`Downloading ${MODELS[state.model].name}: ${mb(msg.loaded)}${msg.total ? ` / ${mb(msg.total)}` : ''} MB`);
      progress(msg.total ? msg.loaded / msg.total : null);
      break;
    case 'ready':
      modelReady = true;
      window.__ftw.ready = true;
      progress(null);
      modelInfo = `${MODELS[state.model].name} on ${msg.backend === 'webgpu' ? 'WebGPU' : `WebAssembly (${msg.threads} thread${msg.threads > 1 ? 's' : ''})`}`;
      status(`${modelInfo} · ready in ${(msg.loadMs / 1000).toFixed(1)} s`);
      infer();
      break;
    case 'tile':
      if (msg.id === runId) stageMsg(`Tile ${msg.done} / ${msg.total}`);
      break;
    case 'result': {
      if (msg.id !== runId) break;
      $('stage').classList.remove('busy');
      stageMsg('');
      const scores = score(msg.classes, msg.size, chip?.mask ?? null, chip?.size ?? 256);
      last = { classes: msg.classes, size: msg.size, scores, ms: msg.ms, tiles: msg.tiles };
      window.__ftw.last = last;
      window.__ftw.runs++;
      renderOverlay();
      renderStats();
      break;
    }
    case 'error':
      $('stage').classList.remove('busy');
      progress(null);
      status(msg.message, true);
      stageMsg('');
      break;
  }
};
worker.onerror = (e) => status(`Worker failed: ${e.message}`, true);

function loadModel() {
  modelReady = false;
  window.__ftw.ready = false;
  status(`Loading ${MODELS[state.model].name}…`);
  send({ type: 'load', url: MODELS[state.model].url, backend: state.backend, base: BASE });
}

async function showChip(country: string, id: string) {
  if (!index) return;
  const token = ++chipToken;
  stageMsg(`Fetching ${index.countries[country]?.name ?? country} ${id}…`);
  try {
    const next = await loadChip(index, country, id);
    if (token !== chipToken) return;
    chip = next;
    last = null;
    // Stretch each window separately so a cloudy window doesn't darken the other.
    const n = chip.size * chip.size;
    ranges = { a: stretchRange([chip.windowA], n), b: stretchRange([chip.windowB], n) };
    drawRGB($<HTMLCanvasElement>('thumb-a'), chip.windowA, chip.size, ranges.a);
    drawRGB($<HTMLCanvasElement>('thumb-b'), chip.windowB, chip.size, ranges.b);
    renderBase();
    renderChipMeta();
    $<HTMLCanvasElement>('pred').getContext('2d')?.clearRect(0, 0, 1e4, 1e4);
    renderOverlay();
    renderStats();
    stageMsg(modelReady ? '' : 'Waiting for the model…');
    infer();
  } catch (e) {
    if (token !== chipToken) return;
    stageMsg(`Could not load ${country} ${id}: ${(e as Error).message}`);
  }
}

async function randomChip(tries = 3) {
  if (!index) return;
  for (let i = 0; i < tries; i++) {
    const pick = pickRandom(index, state.country, state.split);
    await showChip(pick.country, pick.id);
    if (chip?.id === pick.id) return;
  }
}

// ---------- controls ----------
function bindControls() {
  const sel = (id: string, key: 'country' | 'split' | 'model' | 'backend' | 'order') => {
    const el = $<HTMLSelectElement>(id);
    el.value = String(state[key]);
    return el;
  };
  sel('country', 'country').addEventListener('change', (e) => {
    state.country = (e.target as HTMLSelectElement).value;
    randomChip();
  });
  sel('split', 'split').addEventListener('change', (e) => {
    state.split = (e.target as HTMLSelectElement).value;
    randomChip();
  });
  sel('model', 'model').addEventListener('change', (e) => {
    state.model = (e.target as HTMLSelectElement).value;
    loadModel();
  });
  sel('backend', 'backend').addEventListener('change', (e) => {
    state.backend = (e.target as HTMLSelectElement).value as Backend | 'auto';
    loadModel();
  });
  sel('order', 'order').addEventListener('change', (e) => {
    state.order = (e.target as HTMLSelectElement).value;
    infer();
  });

  const slider = (id: string, key: 'divisor' | 'inputSize' | 'patch' | 'overlap' | 'opacity', rerun: boolean) => {
    const el = $<HTMLInputElement>(id);
    el.value = String(state[key]);
    el.addEventListener('input', () => {
      state[key] = num(id);
      if (rerun) infer();
      else {
        renderSliderLabels();
        renderOverlay();
      }
    });
  };
  slider('divisor', 'divisor', true);
  slider('size', 'inputSize', true);
  slider('patch', 'patch', true);
  slider('overlap', 'overlap', true);
  slider('opacity', 'opacity', false);

  for (const [id, key] of [['show-field', 'showField'], ['show-boundary', 'showBoundary']] as const) {
    $<HTMLInputElement>(id).addEventListener('change', (e) => {
      state[key] = (e.target as HTMLInputElement).checked;
      renderOverlay();
    });
  }
  document.querySelectorAll<HTMLElement>('.thumb').forEach((t) => t.addEventListener('click', () => {
    state.base = t.dataset.base as 'a' | 'b';
    renderBase();
  }));
  document.querySelectorAll<HTMLButtonElement>('.view-toggle button').forEach((b) => b.addEventListener('click', () => {
    state.view = b.dataset.view as 'pred' | 'labels';
    document.querySelectorAll<HTMLButtonElement>('.view-toggle button').forEach((o) => {
      o.classList.toggle('on', o === b);
      o.setAttribute('aria-checked', String(o === b));
    });
    stageMsg('');
    renderOverlay();
  }));
  $('random').addEventListener('click', () => randomChip());
  window.addEventListener('keydown', (e) => {
    const tag = (e.target as HTMLElement).tagName;
    if ((e.key === 'r' || e.key === 'R') && !e.metaKey && !e.ctrlKey && tag !== 'INPUT' && tag !== 'SELECT') randomChip();
  });
}

async function main() {
  const fromHash = readHash();
  bindControls();
  renderSliderLabels();
  loadModel();
  try {
    index = await loadIndex(`${BASE}chips.json`);
  } catch (e) {
    stageMsg(`Could not load the chip index: ${(e as Error).message}`);
    return;
  }
  const countrySel = $<HTMLSelectElement>('country');
  for (const [key, c] of Object.entries(index.countries).sort((a, b) => a[1].name.localeCompare(b[1].name))) {
    const o = document.createElement('option');
    o.value = key;
    o.textContent = `${c.name} · ${c.license}`;
    countrySel.append(o);
  }
  if (fromHash.country && fromHash.id && index.countries[fromHash.country]) await showChip(fromHash.country, fromHash.id);
  else await randomChip();
}

main();
