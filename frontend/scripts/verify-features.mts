/**
 * ตรวจว่า src/app/hsv-processing.ts ให้ผลเท่ากับ Python (OpenCV) ทุกขั้นตอน
 *
 *   python src/export_reference.py            (ที่ root ของ repo)
 *   node scripts/verify-features.mts ../reference   (ในโฟลเดอร์ frontend)
 *
 * ใช้พิกเซลที่ Python decode แล้ว จึงตรวจเฉพาะอัลกอริทึม ไม่รวมความต่างของ JPEG decoder ในเบราว์เซอร์
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { IMAGE_SIZE, N_FEATURES, processImage } from '../src/app/hsv-processing.ts';

interface RefEntry {
  file: string;
  width: number;
  height: number;
}

const dir = process.argv[2] ?? '../reference';
const manifest: RefEntry[] = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8'));
const px = IMAGE_SIZE * IMAGE_SIZE;

const stats = { resized: 0, hsv: 0, mask: 0, features: 0, maxFeatureDiff: 0 };
for (let i = 0; i < manifest.length; i++) {
  const { width, height, file } = manifest[i];
  const buf = readFileSync(join(dir, `${i}.bin`));
  let off = 0;
  const take = (n: number) => {
    const s = buf.subarray(off, off + n);
    off += n;
    return s;
  };
  const rgb = take(width * height * 3);
  const refResized = take(px * 3);
  const refHsv = take(px * 3);
  const refMask = take(px);
  const refFeat = new Float32Array(buf.buffer.slice(buf.byteOffset + off, buf.byteOffset + off + N_FEATURES * 4));

  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let p = 0; p < width * height; p++) {
    rgba.set(rgb.subarray(p * 3, p * 3 + 3), p * 4);
    rgba[p * 4 + 3] = 255;
  }
  const out = processImage(rgba, width, height);

  let resizedOk = true;
  for (let p = 0; p < px && resizedOk; p++) {
    for (let c = 0; c < 3; c++) if (out.resized[p * 4 + c] !== refResized[p * 3 + c]) resizedOk = false;
  }
  const hsvOk = out.hsv.every((v, k) => v === refHsv[k]);
  const maskOk = out.mask.every((v, k) => v === refMask[k]);
  let diff = 0;
  for (let k = 0; k < N_FEATURES; k++) diff = Math.max(diff, Math.abs(out.features[k] - refFeat[k]));

  stats.resized += +resizedOk;
  stats.hsv += +hsvOk;
  stats.mask += +maskOk;
  stats.features += +(diff === 0);
  stats.maxFeatureDiff = Math.max(stats.maxFeatureDiff, diff);
  if (!(resizedOk && hsvOk && maskOk && diff === 0)) {
    console.log(`mismatch ${file}: resized=${resizedOk} hsv=${hsvOk} mask=${maskOk} featDiff=${diff}`);
  }
}

const n = manifest.length;
console.log(`images: ${n}`);
console.log(`resize  exact: ${stats.resized}/${n}`);
console.log(`hsv     exact: ${stats.hsv}/${n}`);
console.log(`mask    exact: ${stats.mask}/${n}`);
console.log(`feature exact: ${stats.features}/${n} (max abs diff ${stats.maxFeatureDiff})`);
process.exit(stats.features === n ? 0 : 1);
