import assert from 'node:assert/strict';
import { test } from 'node:test';

import { score } from '../src/metrics.ts';
import {
  accumulate, argmax, blendWeights, cropTile, planTiles, resizeBilinear, snap32, stackWindows, tileStarts,
} from '../src/tiling.ts';

test('snap32 rounds to the model stride', () => {
  assert.equal(snap32(250), 256);
  assert.equal(snap32(10), 32);
  assert.equal(snap32(0, 64), 64);
});

test('stackWindows puts the first window first and divides', () => {
  const out = stackWindows(Uint16Array.of(3000, 6000), Uint16Array.of(1500), 3000);
  assert.deepEqual(Array.from(out), [1, 2, 0.5]);
});

test('tileStarts covers the image and ends flush', () => {
  assert.deepEqual(tileStarts(256, 256, 192), [0]);
  assert.deepEqual(tileStarts(512, 256, 256), [0, 256]);
  assert.deepEqual(tileStarts(512, 256, 192), [0, 192, 256]);
  assert.deepEqual(tileStarts(100, 300, 10), [0]);
});

test('planTiles clamps the patch and drops overlap for one tile', () => {
  assert.deepEqual(planTiles(256, 512, 0.25), { patch: 256, overlap: 0, ys: [0], xs: [0] });
  const p = planTiles(1024, 256, 0.25);
  assert.equal(p.overlap, 64);
  assert.deepEqual(p.ys, [0, 192, 384, 576, 768]);
});

test('blendWeights ramps at both edges and stays positive', () => {
  const w = blendWeights(8, 3);
  assert.deepEqual(Array.from(w), [0.25, 0.5, 0.75, 1, 1, 0.75, 0.5, 0.25]);
  assert.ok(Array.from(blendWeights(64, 32)).every((v) => v > 0));
});

test('resizeBilinear matches torch align_corners=False on a 2x2 upsample', () => {
  // torch.nn.functional.interpolate(torch.tensor([[[[0., 1.], [2., 3.]]]]), size=4, mode="bilinear")
  const out = resizeBilinear(Float32Array.of(0, 1, 2, 3), 1, 2, 2, 4, 4);
  const expected = [0, 0.25, 0.75, 1, 0.5, 0.75, 1.25, 1.5, 1.5, 1.75, 2.25, 2.5, 2, 2.25, 2.75, 3];
  Array.from(out).forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-6, `${i}: ${v}`));
});

test('cropTile and accumulate reassemble an image from overlapping tiles', () => {
  const size = 8;
  const img = Float32Array.from({ length: 2 * size * size }, (_, i) => i);
  const plan = { patch: 4, ys: [0, 2, 4], xs: [0, 2, 4] };
  const acc = new Float32Array(img.length);
  const wsum = new Float32Array(img.length);
  const ones = new Float32Array(4).fill(1);
  for (const y of plan.ys) {
    for (const x of plan.xs) {
      accumulate(acc, size, size, cropTile(img, 2, size, size, y, x, 4, 4), 2, y, x, 4, 4, ones, ones);
      accumulate(wsum, size, size, new Float32Array(2 * 16).fill(1), 2, y, x, 4, 4, ones, ones);
    }
  }
  acc.forEach((v, i) => assert.ok(Math.abs(v / wsum[i] - img[i]) < 1e-4));
});

test('argmax picks the largest class per pixel', () => {
  // 3 classes x 2 pixels
  assert.deepEqual(Array.from(argmax(Float32Array.of(1, 0, 2, 5, 0, 9), 3, 2)), [1, 2]);
});

test('score computes IoU, ignores nodata, and samples a larger prediction', () => {
  const mask = Uint8Array.of(0, 1, 2, 3);
  const pred = Uint8Array.of(0, 1, 2, 2); // 2x2, last label pixel is nodata
  const s = score(pred, 2, mask, 2);
  assert.equal(s.fieldIoU, 1);
  assert.equal(s.boundaryIoU, 1);
  assert.equal(s.accuracy, 1);
  // 4x4 prediction of all field vs the 2x2 mask
  const big = score(new Uint8Array(16).fill(1), 4, mask, 2);
  assert.equal(big.fieldIoU, 1 / 3);
  assert.deepEqual(big.fractions, [0, 1, 0]);
});
