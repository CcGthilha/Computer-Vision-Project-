import { Component, ElementRef, OnDestroy, computed, effect, inject, signal, viewChild } from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';

import { FruitStats, computeStats, drawToRgba, maskImage, maskedImage, readImageFile, resizedImage } from './fruit-analysis';
import { H_BINS, IMAGE_SIZE, ProcessedImage, processImage } from './hsv-processing';
import { ModelInference, Prediction } from './model-inference';

interface Analysis {
  id: number;
  name: string;
  source: 'upload' | 'camera';
  thumbnail: string;
  processed: ProcessedImage;
  stats: FruitStats;
  prediction: Prediction;
  /** เวลาประมวลผลทั้งหมด (ms) */
  elapsed: number;
}

interface Sample {
  file: string;
  label: string;
}

const LIVE_INTERVAL_MS = 500;
const MAX_HISTORY = 100;

/** ภาพตัวอย่างจาก test set (frontend/public/samples) */
const SAMPLES: Sample[] = [
  { file: 'fresh_peach.jpg', label: 'ลูกพีช (สด)' },
  { file: 'rotten_peach.jpg', label: 'ลูกพีช (เน่า)' },
  { file: 'fresh_pomegranate.jpg', label: 'ทับทิม (สด)' },
  { file: 'rotten_pomegranate.jpg', label: 'ทับทิม (เน่า)' },
  { file: 'fresh_strawberry.jpg', label: 'สตรอว์เบอร์รี (สด)' },
  { file: 'rotten_strawberry.jpg', label: 'สตรอว์เบอร์รี (เน่า)' },
];

@Component({
  selector: 'app-root',
  imports: [DecimalPipe, PercentPipe],
  templateUrl: './app.html',
})
export class App implements OnDestroy {
  protected readonly model = inject(ModelInference);

  protected readonly mode = signal<'upload' | 'camera'>('upload');
  protected readonly busy = signal(false);
  protected readonly dragging = signal(false);
  protected readonly highlightDark = signal(true);
  protected readonly history = signal<Analysis[]>([]);
  protected readonly current = signal<Analysis | null>(null);
  protected readonly message = signal<string | null>(null);
  protected readonly progress = signal('');
  protected readonly samples = SAMPLES;

  protected readonly cameraOn = signal(false);
  protected readonly live = signal(false);

  /** เตือนเมื่อแยกผลไม้ออกจากพื้นหลังได้ไม่ดี (ผลอาจไม่น่าเชื่อถือ) */
  protected readonly warning = computed(() => {
    const a = this.current();
    if (!a) return null;
    if (a.stats.fruitArea < 3) return 'ตรวจไม่พบผลไม้ชัดเจนในภาพ ผลการทำนายอาจไม่ถูกต้อง';
    if (a.stats.fruitArea > 90) return 'แยกผลไม้ออกจากพื้นหลังไม่ได้ (ควรถ่ายบนพื้นหลังสีอ่อนและเห็นผลไม้ทั้งผล)';
    return null;
  });

  protected readonly summary = computed(() => {
    const items = this.history();
    const fresh = items.filter((a) => a.prediction.label === 'fresh').length;
    const avg = (f: (a: Analysis) => number) => (items.length ? items.reduce((s, a) => s + f(a), 0) / items.length : 0);
    return {
      total: items.length,
      fresh,
      rotten: items.length - fresh,
      freshRatio: items.length ? fresh / items.length : 0,
      avgConfidence: avg((a) => a.prediction.confidence),
      avgTime: avg((a) => a.elapsed),
    };
  });

  protected readonly imageSize = IMAGE_SIZE;

  private readonly originalCanvas = viewChild<ElementRef<HTMLCanvasElement>>('originalCanvas');
  private readonly maskCanvas = viewChild<ElementRef<HTMLCanvasElement>>('maskCanvas');
  private readonly maskedCanvas = viewChild<ElementRef<HTMLCanvasElement>>('maskedCanvas');
  private readonly hueCanvas = viewChild<ElementRef<HTMLCanvasElement>>('hueCanvas');
  private readonly video = viewChild<ElementRef<HTMLVideoElement>>('video');

  private stream: MediaStream | null = null;
  private liveTimer: ReturnType<typeof setInterval> | null = null;
  private liveRunning = false;
  private nextId = 1;

  constructor() {
    // ต่อกล้องเข้ากับ <video> หลังจาก element ถูกสร้างแล้ว
    effect(() => {
      const v = this.video()?.nativeElement;
      if (v && this.cameraOn() && this.stream && v.srcObject !== this.stream) {
        v.srcObject = this.stream;
        v.play().catch(() => {});
      }
    });

    effect(() => {
      const a = this.current();
      const canvases = [this.originalCanvas(), this.maskCanvas(), this.maskedCanvas(), this.hueCanvas()];
      if (!a || canvases.some((c) => !c)) return;
      const [orig, mask, masked, hue] = canvases.map((c) => c!.nativeElement);
      orig.getContext('2d')!.putImageData(resizedImage(a.processed), 0, 0);
      mask.getContext('2d')!.putImageData(maskImage(a.processed), 0, 0);
      masked.getContext('2d')!.putImageData(maskedImage(a.processed, this.highlightDark()), 0, 0);
      drawHueHistogram(hue, a.stats.hueHistogram);
    });
  }

  // ---------------- Upload ----------------

  protected onFileInput(event: Event) {
    const input = event.target as HTMLInputElement;
    this.analyzeFiles(Array.from(input.files ?? []));
    input.value = '';
  }

  protected onDrop(event: DragEvent) {
    event.preventDefault();
    this.dragging.set(false);
    this.analyzeFiles(Array.from(event.dataTransfer?.files ?? []));
  }

  protected onDragOver(event: DragEvent) {
    event.preventDefault();
    this.dragging.set(true);
  }

  private async analyzeFiles(files: File[]) {
    if (this.busy()) return;
    const images = files.filter((f) => f.type.startsWith('image/'));
    const skipped = files.length - images.length;
    if (!images.length) {
      this.message.set('ไฟล์ที่เลือกไม่ใช่ภาพ กรุณาเลือกไฟล์ jpg, png หรือ webp');
      return;
    }
    this.message.set(null);
    this.busy.set(true);
    const failed: string[] = [];
    try {
      for (const [i, file] of images.entries()) {
        this.progress.set(images.length > 1 ? `${i + 1}/${images.length}` : '');
        try {
          const t0 = performance.now();
          const { data, width, height } = await readImageFile(file);
          await this.analyze(data, width, height, file.name, 'upload', t0);
        } catch (e) {
          console.error(file.name, e);
          failed.push(file.name);
        }
      }
    } finally {
      this.busy.set(false);
      this.progress.set('');
    }
    const notes = [];
    if (failed.length) notes.push(`อ่านไฟล์ไม่ได้ ${failed.length} ไฟล์: ${failed.join(', ')}`);
    if (skipped) notes.push(`ข้ามไฟล์ที่ไม่ใช่ภาพ ${skipped} ไฟล์`);
    this.message.set(notes.length ? notes.join(' / ') : null);
  }

  protected async loadSample(sample: Sample) {
    if (this.busy()) return;
    this.busy.set(true);
    this.message.set(null);
    try {
      const t0 = performance.now();
      const blob = await (await fetch(`samples/${sample.file}`)).blob();
      const { data, width, height } = await readImageFile(blob);
      await this.analyze(data, width, height, `ตัวอย่าง: ${sample.label}`, 'upload', t0);
    } catch (e) {
      this.message.set(`โหลดภาพตัวอย่างไม่สำเร็จ: ${e instanceof Error ? e.message : e}`);
    } finally {
      this.busy.set(false);
    }
  }

  // ---------------- Camera ----------------

  protected async startCamera() {
    this.message.set(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      this.message.set('เบราว์เซอร์นี้ใช้กล้องไม่ได้ (ต้องเปิดผ่าน https หรือ localhost)');
      return;
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      this.cameraOn.set(true);
    } catch (e) {
      const name = e instanceof DOMException ? e.name : '';
      this.message.set(
        name === 'NotAllowedError'
          ? 'ไม่ได้รับอนุญาตให้ใช้กล้อง กรุณากดอนุญาตในเบราว์เซอร์'
          : name === 'NotFoundError'
            ? 'ไม่พบกล้องในเครื่องนี้'
            : `เปิดกล้องไม่ได้: ${e instanceof Error ? e.message : e}`,
      );
    }
  }

  protected stopCamera() {
    this.setLive(false);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.cameraOn.set(false);
  }

  /** ถ่ายภาพจากกล้องแล้วตรวจ และบันทึกลงประวัติ */
  protected async capture() {
    if (this.busy()) return;
    this.busy.set(true);
    try {
      await this.analyzeFrame(true);
    } finally {
      this.busy.set(false);
    }
  }

  private async analyzeFrame(record: boolean) {
    const v = this.video()?.nativeElement;
    if (!v || !v.videoWidth) return;
    const t0 = performance.now();
    // ครอปตรงกลางเป็นสี่เหลี่ยมจัตุรัส เหมือนภาพใน dataset
    const side = Math.min(v.videoWidth, v.videoHeight);
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = side;
    canvas.getContext('2d')!.drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, side, side);
    const { data } = drawToRgba(canvas, side, side);
    const name = `กล้อง ${new Date().toLocaleTimeString('th-TH')}`;
    await this.analyze(data, side, side, record ? name : 'กล้อง (ตรวจต่อเนื่อง)', 'camera', t0, record);
  }

  /** โหมดตรวจต่อเนื่อง: อัปเดตผลทุก 0.5 วินาที แต่ไม่บันทึกลงประวัติ (กด "บันทึกผล" เพื่อบันทึก) */
  protected setLive(on: boolean) {
    this.live.set(on);
    if (this.liveTimer) clearInterval(this.liveTimer);
    this.liveTimer = null;
    if (!on) return;
    this.liveTimer = setInterval(async () => {
      if (this.liveRunning || this.busy()) return;
      this.liveRunning = true;
      try {
        await this.analyzeFrame(false);
      } finally {
        this.liveRunning = false;
      }
    }, LIVE_INTERVAL_MS);
  }

  protected switchMode(mode: 'upload' | 'camera') {
    if (mode === 'upload') this.stopCamera();
    this.mode.set(mode);
  }

  // ---------------- Pipeline ----------------

  private async analyze(
    data: Uint8ClampedArray,
    width: number,
    height: number,
    name: string,
    source: Analysis['source'],
    t0: number,
    record = true,
  ) {
    const processed = processImage(data, width, height);
    const prediction = await this.model.predict(processed.features);
    const analysis: Analysis = {
      id: this.nextId++,
      name,
      source,
      thumbnail: record ? thumbnail(processed) : '',
      processed,
      stats: computeStats(processed),
      prediction,
      elapsed: performance.now() - t0,
    };
    this.current.set(analysis);
    if (record) this.history.update((h) => [analysis, ...h].slice(0, MAX_HISTORY));
  }

  protected select(a: Analysis) {
    this.current.set(a);
  }

  protected clearHistory() {
    this.history.set([]);
    this.current.set(null);
  }

  protected exportCsv() {
    const rows = [['name', 'source', 'prediction', 'p_fresh', 'p_rotten', 'fruit_area_pct', 'dark_area_pct', 'mean_h', 'mean_s', 'mean_v', 'time_ms']];
    for (const a of [...this.history()].reverse()) {
      rows.push([
        a.name,
        a.source,
        a.prediction.label,
        a.prediction.probabilities.fresh.toFixed(4),
        a.prediction.probabilities.rotten.toFixed(4),
        a.stats.fruitArea.toFixed(2),
        a.stats.darkArea.toFixed(2),
        a.stats.meanH.toFixed(1),
        a.stats.meanS.toFixed(1),
        a.stats.meanV.toFixed(1),
        a.elapsed.toFixed(1),
      ]);
    }
    const csv = rows.map((r) => r.map((c) => `"${c.replaceAll('"', '""')}"`).join(',')).join('\n');
    const url = URL.createObjectURL(new Blob(['﻿' + csv], { type: 'text/csv' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'fruit-freshness-results.csv';
    link.click();
    URL.revokeObjectURL(url);
  }

  ngOnDestroy() {
    this.stopCamera();
  }
}

function thumbnail(p: ProcessedImage): string {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = IMAGE_SIZE;
  canvas.getContext('2d')!.putImageData(resizedImage(p), 0, 0);
  return canvas.toDataURL('image/jpeg', 0.8);
}

/** กราฟแท่ง Hue histogram ระบายสีตามค่า Hue จริง */
function drawHueHistogram(canvas: HTMLCanvasElement, hist: number[]) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 320;
  const h = canvas.clientHeight || 120;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const max = Math.max(...hist, 1e-6);
  const bw = w / H_BINS;
  hist.forEach((v, i) => {
    const bh = (v / max) * (h - 4);
    ctx.fillStyle = `hsl(${((i + 0.5) * 360) / H_BINS}, 70%, 50%)`;
    ctx.fillRect(i * bw + 1, h - bh, bw - 2, bh);
  });
}
