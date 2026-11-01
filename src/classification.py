"""Train Random Forest จาก HSV Histogram feature

    python src/classification.py --data dataset --out models/model_new.pkl

ค่าเริ่มต้นบันทึกเป็นไฟล์ใหม่ เพื่อไม่ให้ทับ models/model.pkl ที่ใช้ใน Application
"""
import argparse

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from dataset import load_split
from feature_extraction import H_BINS, S_BINS, V_BINS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="models/model_new.pkl")
    args = ap.parse_args()

    X_train, y_train, _ = load_split(args.data, "train")
    X_val, y_val, _ = load_split(args.data, "val")
    print(f"train: {X_train.shape}, val: {X_val.shape}")

    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    val_acc = accuracy_score(y_val, model.predict(scaler.transform(X_val)))
    print(f"validation accuracy: {val_acc:.4f}")

    joblib.dump({
        "scaler": scaler,
        "model": model,
        "feature_params": {"h_bins": H_BINS, "s_bins": S_BINS, "v_bins": V_BINS, "use_mask": True},
    }, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
