import { Injectable, signal } from '@angular/core';
import type { InferenceSession } from 'onnxruntime-web';

import { N_FEATURES } from './hsv-processing';

export const CLASS_NAMES = ['fresh', 'rotten'] as const;
export type FruitClass = (typeof CLASS_NAMES)[number];

export interface Prediction {
  label: FruitClass;
  /** ความน่าจะเป็นของคลาสที่ทาย (0-1) */
  confidence: number;
  probabilities: Record<FruitClass, number>;
}

/** โหลด model.onnx (StandardScaler + RandomForest) และทำนายในเบราว์เซอร์ด้วย onnxruntime-web */
@Injectable({ providedIn: 'root' })
export class ModelInference {
  readonly status = signal<'loading' | 'ready' | 'error'>('loading');
  readonly error = signal<string | null>(null);

  private ort?: typeof import('onnxruntime-web');
  private session?: InferenceSession;
  private readonly ready = this.load();

  private async load(): Promise<void> {
    try {
      // lazy import เพื่อไม่ให้ไฟล์ JS หลักใหญ่เกินไป
      this.ort = await import('onnxruntime-web/wasm');
      this.ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href; // คัดลอกจาก node_modules ผ่าน angular.json
      this.ort.env.wasm.numThreads = 1;
      this.session = await this.ort.InferenceSession.create(new URL('model.onnx', document.baseURI).href, {
        executionProviders: ['wasm'],
      });
      this.status.set('ready');
    } catch (e) {
      console.error(e);
      this.error.set(e instanceof Error ? e.message : String(e));
      this.status.set('error');
      throw e;
    }
  }

  async predict(features: Float32Array): Promise<Prediction> {
    await this.ready;
    const input = new this.ort!.Tensor('float32', features, [1, N_FEATURES]);
    const out = await this.session!.run({ features: input });
    const probs = out['probabilities'].data as Float32Array;
    const probabilities = { fresh: probs[0], rotten: probs[1] };
    const label: FruitClass = probabilities.rotten > probabilities.fresh ? 'rotten' : 'fresh';
    return { label, confidence: probabilities[label], probabilities };
  }
}
