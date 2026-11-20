// Canvas rendering: percentile-stretched RGB composites and class overlays.

export const CLASS_COLORS: Record<number, [number, number, number]> = {
  1: [192, 216, 91], // field, FTW Yellow Green
  2: [255, 127, 65], // boundary, FTW Burning Orange
};
// Fields cover most of a chip, so they are drawn lighter than boundaries to keep the imagery visible.
const CLASS_ALPHA: Record<number, number> = { 1: 120, 2: 255 };

/** 2nd and 98th percentile of the RGB bands (the first three) of the given images, pooled. */
export function stretchRange(images: Uint16Array[], n: number): [number, number] {
  const hist = new Uint32Array(65536);
  let count = 0;
  for (const img of images) {
    for (let i = 0; i < 3 * n; i += 7) { // subsample; plenty for a percentile
      hist[img[i]]++;
      count++;
    }
  }
  const lo = count * 0.02;
  const hi = count * 0.98;
  let acc = 0;
  let pLo = 0;
  let pHi = 65535;
  for (let v = 0; v < 65536; v++) {
    acc += hist[v];
    if (acc < lo) pLo = v + 1;
    if (acc >= hi) {
      pHi = v;
      break;
    }
  }
  return [pLo, Math.max(pHi, pLo + 1)];
}

/** Draw bands 0-2 (B04, B03, B02) of a band-sequential image as RGB. */
export function drawRGB(canvas: HTMLCanvasElement, img: Uint16Array, size: number, range: [number, number]): void {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const out = ctx.createImageData(size, size);
  const n = size * size;
  const [lo, hi] = range;
  const k = 255 / (hi - lo);
  for (let p = 0; p < n; p++) {
    out.data[p * 4] = (img[p] - lo) * k;
    out.data[p * 4 + 1] = (img[n + p] - lo) * k;
    out.data[p * 4 + 2] = (img[2 * n + p] - lo) * k;
    out.data[p * 4 + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

/** Draw a class map; `show` lists the classes to paint, everything else is transparent. */
export function drawClasses(canvas: HTMLCanvasElement, classes: Uint8Array, size: number, show: Set<number>): void {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const out = ctx.createImageData(size, size);
  for (let p = 0; p < classes.length; p++) {
    const c = classes[p];
    const rgb = show.has(c) ? CLASS_COLORS[c] : undefined;
    if (!rgb) continue;
    out.data[p * 4] = rgb[0];
    out.data[p * 4 + 1] = rgb[1];
    out.data[p * 4 + 2] = rgb[2];
    out.data[p * 4 + 3] = CLASS_ALPHA[c];
  }
  ctx.putImageData(out, 0, 0);
}
