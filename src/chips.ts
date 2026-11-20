// FTW benchmark chips on Source Cooperative: index, random sampling, and GeoTIFF decoding.
import { fromArrayBuffer } from 'geotiff';

export interface Country {
  name: string;
  year: string;
  license: string;
  window_a: [string, string];
  window_b: [string, string];
  splits: Record<string, string[]>;
}

export interface ChipIndex {
  base: string;
  countries: Record<string, Country>;
}

export interface Chip {
  country: string;
  id: string;
  size: number;
  /** Band-sequential uint16 B04, B03, B02, B08. */
  windowA: Uint16Array;
  windowB: Uint16Array;
  /** 0 background, 1 field, 2 boundary, 3 no data; null when the label is missing. */
  mask: Uint8Array | null;
  /** [west, south, east, north] in degrees. */
  bbox: number[];
}

export async function loadIndex(url: string): Promise<ChipIndex> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`chip index: HTTP ${res.status}`);
  return res.json();
}

/** Pick a random chip. With country "any", pick a country uniformly, then a chip in it. */
export function pickRandom(index: ChipIndex, country: string, split: string): { country: string; id: string } {
  const pool = country === 'any'
    ? Object.keys(index.countries).filter((c) => index.countries[c].splits[split]?.length)
    : [country];
  const c = pool[Math.floor(Math.random() * pool.length)];
  const ids = index.countries[c]?.splits[split] ?? [];
  if (!ids.length) throw new Error(`no ${split} chips in ${c}`);
  return { country: c, id: ids[Math.floor(Math.random() * ids.length)] };
}

export function splitOf(index: ChipIndex, country: string, id: string): string | null {
  const splits = index.countries[country]?.splits ?? {};
  for (const [s, ids] of Object.entries(splits)) if (ids.includes(id)) return s;
  return null;
}

async function readTiff(url: string, signal?: AbortSignal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${url.split('/').slice(-3).join('/')}: HTTP ${res.status}`);
  const tiff = await fromArrayBuffer(await res.arrayBuffer());
  const image = await tiff.getImage();
  const rasters = (await image.readRasters({ interleave: false })) as unknown as ArrayLike<number>[];
  return { image, rasters, width: image.getWidth(), height: image.getHeight() };
}

function bandSequential(rasters: ArrayLike<number>[], n: number): Uint16Array {
  const out = new Uint16Array(rasters.length * n);
  rasters.forEach((r, b) => out.set(r as ArrayLike<number>, b * n));
  return out;
}

export async function loadChip(index: ChipIndex, country: string, id: string, signal?: AbortSignal): Promise<Chip> {
  const root = `${index.base}/${country}`;
  const [a, b, m] = await Promise.all([
    readTiff(`${root}/s2_images/window_a/${id}.tif`, signal),
    readTiff(`${root}/s2_images/window_b/${id}.tif`, signal),
    readTiff(`${root}/label_masks/semantic_3class/${id}.tif`, signal).catch(() => null),
  ]);
  if (a.width !== b.width || a.height !== b.height || a.width !== a.height) {
    throw new Error(`unexpected chip shape ${a.width}x${a.height} / ${b.width}x${b.height}`);
  }
  const n = a.width * a.height;
  return {
    country,
    id,
    size: a.width,
    windowA: bandSequential(a.rasters, n),
    windowB: bandSequential(b.rasters, n),
    mask: m && m.width === a.width ? Uint8Array.from(m.rasters[0] as ArrayLike<number>) : null,
    bbox: a.image.getBoundingBox(),
  };
}
