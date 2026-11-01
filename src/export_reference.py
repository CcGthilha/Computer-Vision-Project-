"""สร้างข้อมูลอ้างอิงจากฝั่ง Python ให้ frontend/scripts/verify-features.ts ตรวจว่า TypeScript ได้ผลเท่ากัน

    python src/export_reference.py --data dataset --out reference

ต่อภาพ: <i>.bin = RGB ต้นฉบับ | resized RGB 128x128 | HSV | mask | features float32
"""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from dataset import list_split
from feature_extraction import hsv_histogram
from preprocessing import load_image, preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--model", default="models/model.pkl")
    ap.add_argument("--out", default="reference")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    bundle = joblib.load(args.model)
    manifest = []
    for split in args.splits:
        paths, labels = list_split(args.data, split)
        for path, label in zip(paths, labels):
            img = load_image(path)
            small, hsv, mask = preprocess(img)
            feat = hsv_histogram(hsv, mask).astype(np.float32)
            prob = bundle["model"].predict_proba(bundle["scaler"].transform(feat[None].astype(np.float64)))[0]
            i = len(manifest)
            with open(out / f"{i}.bin", "wb") as f:
                f.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
                f.write(cv2.cvtColor(small, cv2.COLOR_BGR2RGB).tobytes())
                f.write(hsv.tobytes())
                f.write(mask.tobytes())
                f.write(feat.tobytes())
            manifest.append({"file": str(path), "width": img.shape[1], "height": img.shape[0],
                             "label": int(label), "prob_rotten": float(prob[1])})
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"wrote {len(manifest)} images -> {out}")


if __name__ == "__main__":
    main()
