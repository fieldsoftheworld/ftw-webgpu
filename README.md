# FTW field boundaries in the browser

https://fieldsofthe.world/ftw-webgpu/

This app runs the [Fields of The World](https://fieldsofthe.world) CC-BY PRUE field boundary model on chips from the FTW benchmark, entirely in the browser with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). It picks a random chip from the benchmark on Source Cooperative, stacks its two Sentinel-2 windows, runs the model, and draws the predicted field and boundary classes over the imagery. Each prediction is scored against the chip's 3-class label. Changing a slider reruns inference right away, which shows how the predictions depend on the normalization divisor, the input scale, and the sliding-window patch size.

## What it runs

- **Model.** The [FTW PRUE CC-BY](https://huggingface.co/fieldsoftheworld/ftw-prue-ccby) U-Nets from the FTW v3.1 release (EfficientNet-B3, B5 or B7), exported to ONNX and pinned to commit `ffe4452`. They were trained only on the CC-BY and CC0 countries of the dataset and are licensed CC-BY-4.0. The model downloads once and is cached in the browser.
- **Input.** Sentinel-2 bands B04, B03, B02 and B08, with window B first and window A second. This is the order `ftw_tools` stacks them in training (`temporal_options="stacked"`). Reflectance is divided by the normalization divisor (3000 in training).
- **Output.** Per-pixel logits for background, field and boundary. The app shows their argmax.
- **Chips.** The [FTW benchmark](https://source.coop/kerner-lab/fields-of-the-world): 256 × 256 GeoTIFFs for `window_a`, `window_b` and `label_masks/semantic_3class`, fetched directly from `data.source.coop`. The default split is test, which the model never saw. Label licenses vary by country and are shown with each chip.

## Controls

- **Normalization divisor.** What reflectance is divided by before inference.
- **Window order.** Feeds the windows in training order or swapped.
- **Input size.** The chip is bilinearly resampled to this size (a multiple of 32, from 128 to 1024 px) before inference. Upsampling approximates a larger tile, or imagery at a finer resolution than the model was trained on.
- **Patch size and overlap.** An input larger than the patch runs as a sliding window. Overlapping logits are blended with a linear ramp before the argmax.

The URL hash keeps the chip and settings, so a link reopens the same view.

The WebGPU backend is used when the browser has it. Otherwise inference runs on WebAssembly, using threads when the page is cross-origin isolated. GitHub Pages can't set the COOP/COEP headers that isolation needs, so [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) adds them, and the page reloads once on its first visit.

Raster-to-polygon conversion of the predictions isn't implemented yet.

## Development

```bash
npm ci
npm run dev        # http://localhost:5173/ftw-webgpu/
npm test           # unit tests (Node's built-in runner)
npm run build      # typecheck + production build into dist/
```

`public/chips.json` is the per-country chip index (ids by split, window dates, label license). Rebuild it from Source Cooperative with `python scripts/build_chip_index.py`, which needs `duckdb`.

Pushing to `main` deploys to GitHub Pages through `.github/workflows/deploy.yml`.

## License

The code is MIT. The model weights are CC-BY-4.0 from Fields of The World. Chip imagery is Copernicus Sentinel-2 data, and each country's labels keep their source license.
