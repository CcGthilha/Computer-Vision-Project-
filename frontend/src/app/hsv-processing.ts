/**
 * Image Processing + Feature Extraction ฝั่ง client (TypeScript ล้วน ไม่ใช้ OpenCV.js)
 *
 * ต้องให้ผลตรงกับ src/preprocessing.py และ src/feature_extraction.py ทุกขั้นตอน:
 *   resize 128x128 (cv2.INTER_AREA) -> cv2.COLOR_BGR2HSV -> Otsu threshold บนช่อง S
 *   -> morphology close + open (ellipse 5x5) -> cv2.calcHist 32x32x8 -> cv2.normalize (L2)
 *
 * แต่ละฟังก์ชันเขียนตามอัลกอริทึมใน OpenCV เพื่อให้ได้ค่าเท่ากัน (ตรวจด้วย scripts/verify-features.mts)
 */

export const IMAGE_SIZE = 128;
export const H_BINS = 32;
export const S_BINS = 32;
export const V_BINS = 8;
export const N_FEATURES = H_BINS * S_BINS * V_BINS;

export interface ProcessedImage {
  /** ภาพหลัง resize (RGBA, IMAGE_SIZE x IMAGE_SIZE) */
  resized: Uint8ClampedArray;
  /** HSV แบบ OpenCV: H 0-179, S 0-255, V 0-255 (3 ช่องต่อพิกเซล) */
  hsv: Uint8Array;
  /** 255 = ผลไม้, 0 = พื้นหลัง */
  mask: Uint8Array;
  /** threshold ที่ Otsu เลือก (บนช่อง S) */
  otsuThreshold: number;
  /** HSV Histogram 8192 ค่า (L2 normalized) = input ของโมเดล */
  features: Float32Array;
}

export function processImage(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): ProcessedImage {
  const resized = resizeRgba(rgba, width, height, IMAGE_SIZE, IMAGE_SIZE);
  const hsv = rgbaToHsv(resized);
  const sat = channel(hsv, 1);
  const otsuThreshold = otsu(sat);
  let mask = threshold(sat, otsuThreshold);
  mask = close(mask, IMAGE_SIZE, IMAGE_SIZE);
  mask = open(mask, IMAGE_SIZE, IMAGE_SIZE);
  const features = hsvHistogram(hsv, mask);
  return { resized, hsv, mask, otsuThreshold, features };
}

// ---------------------------------------------------------------------------
// Resize: cv2.resize(..., interpolation=cv2.INTER_AREA)
// ---------------------------------------------------------------------------

interface AreaTab {
  di: number;
  si: number;
  alpha: number;
}

/** computeResizeAreaTab() ใน opencv/modules/imgproc/src/resize.cpp */
function areaTab(ssize: number, dsize: number, scale: number): AreaTab[] {
  const tab: AreaTab[] = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1);
    let sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) {
      tab.push({ di: dx, si: sx1 - 1, alpha: Math.fround((sx1 - fsx1) / cellWidth) });
    }
    for (let sx = sx1; sx < sx2; sx++) {
      tab.push({ di: dx, si: sx, alpha: Math.fround(1 / cellWidth) });
    }
    if (fsx2 - sx2 > 1e-3) {
      tab.push({ di: dx, si: sx2, alpha: Math.fround(Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth) });
    }
  }
  return tab;
}

/** saturate_cast<uchar>(float) = ปัดเศษแบบ round-half-to-even แล้วจำกัด 0-255 */
function toUchar(v: number): number {
  let r = Math.round(v);
  if (r - v === 0.5 && r % 2 !== 0) r -= 1; // half -> even
  return r < 0 ? 0 : r > 255 ? 255 : r;
}

function resizeRgba(src: Uint8ClampedArray | Uint8Array, sw: number, sh: number, dw: number, dh: number): Uint8ClampedArray {
  // cv::resize คำนวณ scale = 1 / (dst / src) (ผลลัพธ์ double ต่างจาก src / dst เล็กน้อย)
  const scaleX = 1 / (dw / sw);
  const scaleY = 1 / (dh / sh);
  if (scaleX < 1 || scaleY < 1) {
    // ภาพเล็กกว่า 128: OpenCV INTER_AREA จะใช้ bilinear แทน
    return resizeBilinear(src, sw, sh, dw, dh);
  }
  if (Math.abs(scaleX - Math.round(scaleX)) < Number.EPSILON && Math.abs(scaleY - Math.round(scaleY)) < Number.EPSILON) {
    return resizeAreaFast(src, sw, dw, dh, Math.round(scaleX), Math.round(scaleY));
  }

  const xtab = areaTab(sw, dw, scaleX);
  const ytab = areaTab(sh, dh, scaleY);
  const dst = new Uint8ClampedArray(dw * dh * 4);
  const buf = new Float32Array(dw * 3);
  const sum = new Float32Array(dw * 3);
  let prevDy = ytab[0].di;

  const flush = (dy: number) => {
    for (let dx = 0; dx < dw; dx++) {
      const o = (dy * dw + dx) * 4;
      dst[o] = toUchar(sum[dx * 3]);
      dst[o + 1] = toUchar(sum[dx * 3 + 1]);
      dst[o + 2] = toUchar(sum[dx * 3 + 2]);
      dst[o + 3] = 255;
    }
  };

  for (const { di: dy, si: sy, alpha: beta } of ytab) {
    buf.fill(0);
    const row = sy * sw * 4;
    for (const { di: dxn, si: sxn, alpha } of xtab) {
      const s = row + sxn * 4;
      for (let c = 0; c < 3; c++) {
        buf[dxn * 3 + c] = Math.fround(buf[dxn * 3 + c] + Math.fround(src[s + c] * alpha));
      }
    }
    if (dy !== prevDy) {
      flush(prevDy);
      for (let i = 0; i < buf.length; i++) sum[i] = Math.fround(beta * buf[i]);
      prevDy = dy;
    } else {
      for (let i = 0; i < buf.length; i++) sum[i] = Math.fround(sum[i] + Math.fround(beta * buf[i]));
    }
  }
  flush(prevDy);
  return dst;
}

/** resizeAreaFast_ (ย่อด้วยสเกลจำนวนเต็ม เช่น 256 -> 128) */
function resizeAreaFast(src: Uint8ClampedArray | Uint8Array, sw: number, dw: number, dh: number, sx: number, sy: number): Uint8ClampedArray {
  const dst = new Uint8ClampedArray(dw * dh * 4);
  const area = sx * sy;
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const o = (y * dw + x) * 4;
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let j = 0; j < sy; j++) {
          for (let i = 0; i < sx; i++) s += src[((y * sy + j) * sw + x * sx + i) * 4 + c];
        }
        dst[o + c] = area === 4 ? (s + 2) >> 2 : toUchar(Math.fround(s * Math.fround(1 / area)));
      }
      dst[o + 3] = 255;
    }
  }
  return dst;
}

function resizeBilinear(src: Uint8ClampedArray | Uint8Array, sw: number, sh: number, dw: number, dh: number): Uint8ClampedArray {
  const dst = new Uint8ClampedArray(dw * dh * 4);
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(Math.max((y + 0.5) * (sh / dh) - 0.5, 0), sh - 1);
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, sh - 1);
    const wy = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(Math.max((x + 0.5) * (sw / dw) - 0.5, 0), sw - 1);
      const x0 = Math.floor(fx);
      const x1 = Math.min(x0 + 1, sw - 1);
      const wx = fx - x0;
      const o = (y * dw + x) * 4;
      for (let c = 0; c < 3; c++) {
        const top = src[(y0 * sw + x0) * 4 + c] * (1 - wx) + src[(y0 * sw + x1) * 4 + c] * wx;
        const bottom = src[(y1 * sw + x0) * 4 + c] * (1 - wx) + src[(y1 * sw + x1) * 4 + c] * wx;
        dst[o + c] = Math.round(top * (1 - wy) + bottom * wy);
      }
      dst[o + 3] = 255;
    }
  }
  return dst;
}

// ---------------------------------------------------------------------------
// HSV: cv2.cvtColor(img, cv2.COLOR_BGR2HSV) สำหรับภาพ 8 บิต (H 0-179)
// ---------------------------------------------------------------------------

const HSV_SHIFT = 12;
const SDIV = new Int32Array(256);
const HDIV = new Int32Array(256);
for (let i = 1; i < 256; i++) {
  SDIV[i] = Math.round((255 << HSV_SHIFT) / i);
  HDIV[i] = Math.round((180 << HSV_SHIFT) / (6 * i));
}

export function rgbaToHsv(rgba: Uint8ClampedArray): Uint8Array {
  const n = rgba.length / 4;
  const hsv = new Uint8Array(n * 3);
  const half = 1 << (HSV_SHIFT - 1);
  for (let p = 0; p < n; p++) {
    const r = rgba[p * 4];
    const g = rgba[p * 4 + 1];
    const b = rgba[p * 4 + 2];
    const v = Math.max(r, g, b);
    const diff = v - Math.min(r, g, b);
    const s = (diff * SDIV[v] + half) >> HSV_SHIFT;
    let h: number;
    if (v === r) h = g - b;
    else if (v === g) h = b - r + 2 * diff;
    else h = r - g + 4 * diff;
    h = (h * HDIV[diff] + half) >> HSV_SHIFT;
    if (h < 0) h += 180;
    hsv[p * 3] = h;
    hsv[p * 3 + 1] = s;
    hsv[p * 3 + 2] = v;
  }
  return hsv;
}

function channel(img: Uint8Array, c: number): Uint8Array {
  const out = new Uint8Array(img.length / 3);
  for (let i = 0; i < out.length; i++) out[i] = img[i * 3 + c];
  return out;
}

// ---------------------------------------------------------------------------
// Thresholding: cv2.threshold(S, 0, 255, THRESH_BINARY + THRESH_OTSU)
// ---------------------------------------------------------------------------

const FLT_EPSILON = 1.1920928955078125e-7;

/** getThreshVal_Otsu_8u() */
export function otsu(gray: Uint8Array): number {
  const hist = new Float64Array(256);
  for (const v of gray) hist[v]++;
  const scale = 1 / gray.length;
  let mu = 0;
  for (let i = 0; i < 256; i++) mu += i * hist[i];
  mu *= scale;

  let mu1 = 0;
  let q1 = 0;
  let maxSigma = 0;
  let maxVal = 0;
  for (let i = 0; i < 256; i++) {
    const pi = hist[i] * scale;
    mu1 *= q1;
    q1 += pi;
    const q2 = 1 - q1;
    if (Math.min(q1, q2) < FLT_EPSILON || Math.max(q1, q2) > 1 - FLT_EPSILON) continue;
    mu1 = (mu1 + i * pi) / q1;
    const mu2 = (mu - q1 * mu1) / q2;
    const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > maxSigma) {
      maxSigma = sigma;
      maxVal = i;
    }
  }
  return maxVal;
}

function threshold(gray: Uint8Array, t: number): Uint8Array {
  const out = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) out[i] = gray[i] > t ? 255 : 0;
  return out;
}

// ---------------------------------------------------------------------------
// Morphology: cv2.getStructuringElement(MORPH_ELLIPSE, (5, 5))
// ขอบภาพ: OpenCV ไม่นับพิกเซลนอกภาพ (border value = +inf สำหรับ erode, -inf สำหรับ dilate)
// ---------------------------------------------------------------------------

const KERNEL = [
  [0, 0, 1, 0, 0],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [0, 0, 1, 0, 0],
];
const KERNEL_OFFSETS: [number, number][] = [];
KERNEL.forEach((row, y) => row.forEach((k, x) => k && KERNEL_OFFSETS.push([x - 2, y - 2])));

function morph(src: Uint8Array, w: number, h: number, dilate: boolean): Uint8Array {
  const out = new Uint8Array(src.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = dilate ? 0 : 255;
      for (const [dx, dy] of KERNEL_OFFSETS) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const s = src[yy * w + xx];
        v = dilate ? Math.max(v, s) : Math.min(v, s);
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

function close(mask: Uint8Array, w: number, h: number): Uint8Array {
  return morph(morph(mask, w, h, true), w, h, false);
}

function open(mask: Uint8Array, w: number, h: number): Uint8Array {
  return morph(morph(mask, w, h, false), w, h, true);
}

// ---------------------------------------------------------------------------
// Feature: cv2.calcHist([hsv], [0,1,2], mask, [32,32,8], [0,180,0,256,0,256]) + cv2.normalize (L2)
// ---------------------------------------------------------------------------

/** calcHistLookupTables_8u: bin = floor(value * bins / range) */
function binTable(bins: number, high: number): Int32Array {
  const a = bins / high;
  const tab = new Int32Array(256);
  for (let j = 0; j < 256; j++) {
    const idx = Math.floor(j * a);
    tab[j] = idx < bins ? idx : -1;
  }
  return tab;
}
const H_TAB = binTable(H_BINS, 180);
const S_TAB = binTable(S_BINS, 256);
const V_TAB = binTable(V_BINS, 256);

export function hsvHistogram(hsv: Uint8Array, mask: Uint8Array): Float32Array {
  const n = mask.length;
  const useMask = mask.some((m) => m !== 0); // ไม่พบผลไม้ -> ใช้ทั้งภาพ (เหมือนฝั่ง Python)
  const hist = new Float64Array(N_FEATURES);
  for (let p = 0; p < n; p++) {
    if (useMask && mask[p] === 0) continue;
    const h = H_TAB[hsv[p * 3]];
    const s = S_TAB[hsv[p * 3 + 1]];
    const v = V_TAB[hsv[p * 3 + 2]];
    if (h < 0 || s < 0 || v < 0) continue;
    hist[(h * S_BINS + s) * V_BINS + v]++;
  }
  let norm = 0;
  for (const c of hist) norm += c * c;
  norm = Math.sqrt(norm);
  const scale = Math.fround(norm > 0 ? 1 / norm : 0); // convertTo 32f ใช้ scale แบบ float
  const out = new Float32Array(N_FEATURES);
  for (let i = 0; i < N_FEATURES; i++) out[i] = hist[i] * scale;
  return out;
}
