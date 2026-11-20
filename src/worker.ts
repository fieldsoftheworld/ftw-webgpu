// Inference worker: downloads and caches the ONNX model, then runs sliding-window inference.
import * as ort from 'onnxruntime-web/webgpu';

import {
  NUM_CLASSES, accumulate, argmax, blendWeights, cropTile, planTiles, resizeBilinear,
} from './tiling.ts';

export type Backend = 'webgpu' | 'wasm';

export type WorkerRequest =
  | { type: 'load'; url: string; backend: Backend | 'auto'; base: string }
  | {
    type: 'run'; id: number; image: Float32Array; bands: number; size: number;
    inputSize: number; patch: number; overlap: number;
  };

export type WorkerResponse =
  | { type: 'progress'; loaded: number; total: number; cached: boolean }
  | { type: 'ready'; backend: Backend; threads: number; loadMs: number }
  | { type: 'tile'; id: number; done: number; total: number }
  | {
    type: 'result'; id: number; classes: Uint8Array; size: number; ms: number;
    tiles: number; patch: number; overlap: number;
  }
  | { type: 'error'; id?: number; message: string };

const CACHE = 'ftw-models-v1';
let session: ort.InferenceSession | null = null;
let latestRun = 0;

const post = (msg: WorkerResponse, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(msg, transfer);

async function fetchModel(url: string): Promise<Uint8Array> {
  const cache = 'caches' in self ? await caches.open(CACHE).catch(() => null) : null;
  const hit = await cache?.match(url);
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    post({ type: 'progress', loaded: buf.length, total: buf.length, cached: true });
    return buf;
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`model download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ type: 'progress', loaded, total, cached: false });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  await cache?.put(url, new Response(buf.slice(), { headers: { 'content-type': 'application/octet-stream' } }))
    .catch(() => undefined);
  return buf;
}

async function hasWebGPU(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return (await gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

async function load(url: string, want: Backend | 'auto', base: string): Promise<void> {
  const t0 = performance.now();
  ort.env.wasm.wasmPaths = {
    mjs: new URL(`${base}ort/ort-wasm-simd-threaded.asyncify.mjs`, self.location.origin).href,
    wasm: new URL(`${base}ort/ort-wasm-simd-threaded.asyncify.wasm`, self.location.origin).href,
  };
  const threads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;
  ort.env.wasm.numThreads = threads;
  const backend: Backend = want === 'auto' ? ((await hasWebGPU()) ? 'webgpu' : 'wasm') : want;
  const bytes = await fetchModel(url);
  await session?.release();
  session = null;
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: backend === 'webgpu' ? ['webgpu'] : ['wasm'],
    graphOptimizationLevel: 'all',
  });
  post({ type: 'ready', backend, threads, loadMs: performance.now() - t0 });
}

async function run(req: Extract<WorkerRequest, { type: 'run' }>): Promise<void> {
  if (!session) throw new Error('model not loaded');
  const t0 = performance.now();
  const { id, bands, size, inputSize } = req;
  const image = resizeBilinear(req.image, bands, size, size, inputSize, inputSize);
  const plan = planTiles(inputSize, req.patch, req.overlap);
  const p = plan.patch;
  const w1 = blendWeights(p, plan.overlap);
  const acc = new Float32Array(NUM_CLASSES * inputSize * inputSize);
  const total = plan.ys.length * plan.xs.length;
  let done = 0;
  for (const y of plan.ys) {
    for (const x of plan.xs) {
      if (id !== latestRun) return; // superseded by a newer request
      const tile = cropTile(image, bands, inputSize, inputSize, y, x, p, p);
      const out = await session.run({ image: new ort.Tensor('float32', tile, [1, bands, p, p]) });
      const logits = out[session.outputNames[0]];
      accumulate(acc, inputSize, inputSize, logits.data as Float32Array, NUM_CLASSES, y, x, p, p, w1, w1);
      logits.dispose();
      done++;
      if (total > 1) post({ type: 'tile', id, done, total });
    }
  }
  const classes = argmax(acc, NUM_CLASSES, inputSize * inputSize);
  post({
    type: 'result', id, classes, size: inputSize, ms: performance.now() - t0,
    tiles: total, patch: p, overlap: plan.overlap,
  }, [classes.buffer]);
}

// Requests run one at a time (a session can't run concurrently). A newer run request makes any
// queued or in-progress older run stop at its next tile boundary.
let chain: Promise<void> = Promise.resolve();

self.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  if (req.type === 'run') latestRun = req.id;
  chain = chain.then(async () => {
    try {
      if (req.type === 'load') await load(req.url, req.backend, req.base);
      else if (req.id === latestRun) await run(req);
    } catch (e) {
      post({ type: 'error', id: req.type === 'run' ? req.id : undefined, message: String((e as Error)?.message ?? e) });
    }
  });
};
