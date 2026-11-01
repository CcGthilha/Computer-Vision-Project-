/** สถิติและภาพสำหรับแสดงผล (ไม่มีผลต่อการทำนายของโมเดล) */
import { H_BINS, IMAGE_SIZE, ProcessedImage } from './hsv-processing';

/** พิกเซลผลไม้ที่ V ต่ำกว่าค่านี้นับเป็น "บริเวณสีคล้ำ" (มักเป็นรอยช้ำ/เน่า) */
export const DARK_V = 90;

export interface FruitStats {
  /** สัดส่วนพื้นที่ผลไม้ในภาพ (%) */
  fruitArea: number;
  otsuThreshold: number;
  meanH: number;
  meanS: number;
  meanV: number;
  /** สัดส่วนพิกเซลสีคล้ำในผลไม้ (%) */
  darkArea: number;
  /** Hue histogram 32 bins เฉพาะพิกเซลผลไม้ (รวมเป็น 1) */
  hueHistogram: number[];
}

export function computeStats(p: ProcessedImage): FruitStats {
  const n = p.mask.length;
  const hue = new Array<number>(H_BINS).fill(0);
  let count = 0;
  let sumH = 0;
  let sumS = 0;
  let sumV = 0;
  let dark = 0;
  for (let i = 0; i < n; i++) {
    if (p.mask[i] === 0) continue;
    const h = p.hsv[i * 3];
    const v = p.hsv[i * 3 + 2];
    count++;
    sumH += h;
    sumS += p.hsv[i * 3 + 1];
    sumV += v;
    if (v < DARK_V) dark++;
    hue[Math.min(H_BINS - 1, Math.floor((h * H_BINS) / 180))]++;
  }
  const c = Math.max(count, 1);
  return {
    fruitArea: (count / n) * 100,
    otsuThreshold: p.otsuThreshold,
    meanH: sumH / c,
    meanS: sumS / c,
    meanV: sumV / c,
    darkArea: (dark / c) * 100,
    hueHistogram: hue.map((v) => v / c),
  };
}

function toImageData(rgba: Uint8ClampedArray): ImageData {
  return new ImageData(new Uint8ClampedArray(rgba), IMAGE_SIZE, IMAGE_SIZE);
}

/** ภาพ 128x128 หลัง resize */
export function resizedImage(p: ProcessedImage): ImageData {
  return toImageData(p.resized);
}

/** Mask ขาว-ดำ */
export function maskImage(p: ProcessedImage): ImageData {
  const out = new Uint8ClampedArray(p.mask.length * 4);
  p.mask.forEach((m, i) => {
    out.fill(m, i * 4, i * 4 + 3);
    out[i * 4 + 3] = 255;
  });
  return toImageData(out);
}

/** เฉพาะผลไม้ (พื้นหลังดำ) และไฮไลต์บริเวณสีคล้ำเป็นสีม่วง */
export function maskedImage(p: ProcessedImage, highlightDark: boolean): ImageData {
  const out = new Uint8ClampedArray(p.resized);
  p.mask.forEach((m, i) => {
    if (m === 0) {
      out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = 0;
    } else if (highlightDark && p.hsv[i * 3 + 2] < DARK_V) {
      out[i * 4] = 200;
      out[i * 4 + 1] = 40;
      out[i * 4 + 2] = 220;
    }
  });
  return toImageData(out);
}

/** อ่านไฟล์ภาพเป็น RGBA โดยไม่แปลง color space (ให้ใกล้เคียงกับ cv2.imdecode) */
export async function readImageFile(file: Blob): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const bitmap = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const data = drawToRgba(bitmap, bitmap.width, bitmap.height);
  bitmap.close();
  return data;
}

export function drawToRgba(source: CanvasImageSource, width: number, height: number) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(source, 0, 0, width, height);
  return { data: ctx.getImageData(0, 0, width, height).data, width, height };
}
