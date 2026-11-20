// Pure array math for preprocessing and sliding-window inference.
// Images are band-sequential Float32Arrays shaped (bands, height, width).

export const NUM_CLASSES = 3; // 0 background, 1 field, 2 boundary

/** Round to the nearest multiple of 32 (the model's stride), at least `min`. */
export function snap32(n: number, min = 32): number {
  return Math.max(min, Math.round(n / 32) * 32);
}

/** Concatenate two 4-band uint16 windows into one 8-band float image, divided by `divisor`. */
export function stackWindows(first: Uint16Array, second: Uint16Array, divisor: number): Float32Array {
  const out = new Float32Array(first.length + second.length);
  const inv = 1 / divisor;
  for (let i = 0; i < first.length; i++) out[i] = first[i] * inv;
  const off = first.length;
  for (let i = 0; i < second.length; i++) out[off + i] = second[i] * inv;
  return out;
}

/** Bilinear resize matching torch's F.interpolate(mode="bilinear", align_corners=False). */
export function resizeBilinear(
  src: Float32Array, bands: number, h: number, w: number, outH: number, outW: number,
): Float32Array {
  if (h === outH && w === outW) return src;
  const axis = (inLen: number, outLen: number) => {
    const i0 = new Int32Array(outLen);
    const i1 = new Int32Array(outLen);
    const f = new Float32Array(outLen);
    const scale = inLen / outLen;
    for (let o = 0; o < outLen; o++) {
      const s = Math.max(0, (o + 0.5) * scale - 0.5);
      const lo = Math.min(Math.floor(s), inLen - 1);
      i0[o] = lo;
      i1[o] = Math.min(lo + 1, inLen - 1);
      f[o] = s - lo;
    }
    return { i0, i1, f };
  };
  const ys = axis(h, outH);
  const xs = axis(w, outW);
  const out = new Float32Array(bands * outH * outW);
  for (let b = 0; b < bands; b++) {
    const base = b * h * w;
    const obase = b * outH * outW;
    for (let y = 0; y < outH; y++) {
      const r0 = base + ys.i0[y] * w;
      const r1 = base + ys.i1[y] * w;
      const fy = ys.f[y];
      const orow = obase + y * outW;
      for (let x = 0; x < outW; x++) {
        const x0 = xs.i0[x];
        const x1 = xs.i1[x];
        const fx = xs.f[x];
        const top = src[r0 + x0] + (src[r0 + x1] - src[r0 + x0]) * fx;
        const bot = src[r1 + x0] + (src[r1 + x1] - src[r1 + x0]) * fx;
        out[orow + x] = top + (bot - top) * fy;
      }
    }
  }
  return out;
}

/** Tile start offsets covering [0, len) with tiles of `patch` and step `stride`; the last tile ends at `len`. */
export function tileStarts(len: number, patch: number, stride: number): number[] {
  if (patch >= len) return [0];
  const step = Math.max(1, stride);
  const starts: number[] = [];
  for (let s = 0; s + patch < len; s += step) starts.push(s);
  starts.push(len - patch);
  return starts;
}

export interface TilePlan {
  patch: number;
  overlap: number;
  ys: number[];
  xs: number[];
}

/**
 * Plan a sliding window over a size x size image. `patch` is clamped to the image and snapped to 32;
 * `overlapFrac` (0-0.5) of each tile is shared with its neighbour.
 */
export function planTiles(size: number, patch: number, overlapFrac: number): TilePlan {
  const p = Math.min(snap32(patch), size);
  const overlap = p >= size ? 0 : Math.round(p * Math.min(Math.max(overlapFrac, 0), 0.5));
  const starts = tileStarts(size, p, p - overlap);
  return { patch: p, overlap, ys: starts, xs: starts };
}

/** 1-D blending weights: 1 in the middle, ramping down across the `overlap` pixels at each edge. */
export function blendWeights(patch: number, overlap: number): Float32Array {
  const w = new Float32Array(patch).fill(1);
  if (overlap <= 0) return w;
  for (let i = 0; i < patch; i++) {
    w[i] = Math.min(1, (i + 1) / (overlap + 1), (patch - i) / (overlap + 1));
  }
  return w;
}

/** Copy a (bands, ph, pw) window starting at (y, x) out of a (bands, h, w) image. */
export function cropTile(
  src: Float32Array, bands: number, h: number, w: number, y: number, x: number, ph: number, pw: number,
): Float32Array {
  if (y === 0 && x === 0 && ph === h && pw === w) return src;
  const out = new Float32Array(bands * ph * pw);
  for (let b = 0; b < bands; b++) {
    for (let r = 0; r < ph; r++) {
      const s = b * h * w + (y + r) * w + x;
      out.set(src.subarray(s, s + pw), b * ph * pw + r * pw);
    }
  }
  return out;
}

/** Add weighted tile logits (classes, ph, pw) into the (classes, h, w) accumulator at (y, x). */
export function accumulate(
  acc: Float32Array, h: number, w: number, logits: Float32Array, classes: number,
  y: number, x: number, ph: number, pw: number, wy: Float32Array, wx: Float32Array,
): void {
  for (let c = 0; c < classes; c++) {
    const abase = c * h * w;
    const lbase = c * ph * pw;
    for (let r = 0; r < ph; r++) {
      const arow = abase + (y + r) * w + x;
      const lrow = lbase + r * pw;
      const wr = wy[r];
      for (let k = 0; k < pw; k++) acc[arow + k] += wr * wx[k] * logits[lrow + k];
    }
  }
}

/** Per-pixel argmax over the class axis of a (classes, h, w) array. */
export function argmax(acc: Float32Array, classes: number, hw: number): Uint8Array {
  const out = new Uint8Array(hw);
  for (let p = 0; p < hw; p++) {
    let best = 0;
    let bestVal = acc[p];
    for (let c = 1; c < classes; c++) {
      const v = acc[c * hw + p];
      if (v > bestVal) {
        bestVal = v;
        best = c;
      }
    }
    out[p] = best;
  }
  return out;
}
