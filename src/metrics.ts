// Compare a predicted class map against the FTW 3-class label mask.

export const NODATA = 3;

export interface Scores {
  fieldIoU: number | null;
  boundaryIoU: number | null;
  accuracy: number | null;
  /** Fraction of predicted pixels per class (background, field, boundary). */
  fractions: [number, number, number];
}

/**
 * `pred` is predSize x predSize, `mask` is maskSize x maskSize (label value 3 = no data).
 * The prediction is sampled at each label pixel's centre (nearest neighbour).
 */
export function score(pred: Uint8Array, predSize: number, mask: Uint8Array | null, maskSize: number): Scores {
  const counts = [0, 0, 0];
  for (let i = 0; i < pred.length; i++) counts[pred[i]]++;
  const n = pred.length || 1;
  const fractions: [number, number, number] = [counts[0] / n, counts[1] / n, counts[2] / n];
  if (!mask) return { fieldIoU: null, boundaryIoU: null, accuracy: null, fractions };

  const inter = [0, 0, 0];
  const union = [0, 0, 0];
  let correct = 0;
  let valid = 0;
  const idx = new Int32Array(maskSize);
  for (let i = 0; i < maskSize; i++) idx[i] = Math.min(predSize - 1, Math.floor(((i + 0.5) * predSize) / maskSize));
  for (let y = 0; y < maskSize; y++) {
    const prow = idx[y] * predSize;
    for (let x = 0; x < maskSize; x++) {
      const g = mask[y * maskSize + x];
      if (g === NODATA) continue;
      const p = pred[prow + idx[x]];
      valid++;
      if (p === g) {
        correct++;
        inter[g]++;
        union[g]++;
      } else {
        union[g]++;
        union[p]++;
      }
    }
  }
  const iou = (c: number) => (union[c] ? inter[c] / union[c] : null);
  return { fieldIoU: iou(1), boundaryIoU: iou(2), accuracy: valid ? correct / valid : null, fractions };
}
