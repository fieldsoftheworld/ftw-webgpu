// Copy runtime files that must be served as-is from public/: the onnxruntime-web WASM build that
// matches the `onnxruntime-web/webgpu` bundle, and the cross-origin isolation service worker.
import { copyFileSync, mkdirSync } from 'node:fs';

const ort = 'node_modules/onnxruntime-web/dist';
mkdirSync('public/ort', { recursive: true });
for (const f of ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) {
  copyFileSync(`${ort}/${f}`, `public/ort/${f}`);
}
copyFileSync('node_modules/coi-serviceworker/coi-serviceworker.min.js', 'public/coi-serviceworker.min.js');
