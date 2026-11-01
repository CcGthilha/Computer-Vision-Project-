"""แปลง models/model.pkl (StandardScaler + RandomForest) -> ONNX สำหรับ onnxruntime-web

    python src/export_onnx.py --model models/model.pkl --out frontend/public/model.onnx

Input : "features" float32 [N, 8192]  (HSV Histogram ที่ L2 normalize แล้ว)
Output: "label" int64 [N], "probabilities" float32 [N, 2]  (index 0 = fresh, 1 = rotten)
"""
import argparse
from pathlib import Path

import joblib
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

from feature_extraction import N_FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model.pkl")
    ap.add_argument("--out", default="frontend/public/model.onnx")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    pipeline = Pipeline([("scaler", bundle["scaler"]), ("model", bundle["model"])])

    onx = to_onnx(
        pipeline,
        initial_types=[("features", FloatTensorType([None, N_FEATURES]))],
        options={id(bundle["model"]): {"zipmap": False}},
        target_opset={"": 17, "ai.onnx.ml": 3},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onx.SerializeToString())
    print(f"saved -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
