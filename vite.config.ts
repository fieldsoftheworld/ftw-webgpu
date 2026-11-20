import { defineConfig, type Plugin } from 'vite';

// The worker points onnxruntime-web at public/ort/ (env.wasm.wasmPaths) so WASM threads start from
// that separate module rather than re-running the worker. Vite still emits the bundle's own copy
// of the .wasm, which nothing loads, so drop it.
const dropBundledOrtWasm = (): Plugin => ({
  name: 'drop-bundled-ort-wasm',
  generateBundle(_, bundle) {
    for (const key of Object.keys(bundle)) if (/ort-wasm.*\.wasm$/.test(key)) delete bundle[key];
  },
});

// Served from https://fieldsofthe.world/ftw-webgpu/ (GitHub Pages for fieldsoftheworld/ftw-webgpu)
export default defineConfig({
  base: '/ftw-webgpu/',
  worker: { format: 'es', plugins: () => [dropBundledOrtWasm()] },
  plugins: [dropBundledOrtWasm()],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022' },
});
