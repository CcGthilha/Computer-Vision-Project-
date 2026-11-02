"""เปรียบเทียบโมเดลบน Validation / Test set (ผลในตารางที่ 4 และ 5 ของรายงาน)

    python src/compare_models.py --data dataset

- SVM (RBF, C = 1 และ 10) ใช้ feature ชุดเดียวกับ models/model.pkl
- Random Forest แบบไม่ใช้ mask (คิด Histogram ทั้งภาพ) เพื่อดูผลของการตัดพื้นหลัง
- Random Forest แบบใช้ mask = models/model.pkl
"""
import argparse

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import list_split, load_split
from feature_extraction import hsv_histogram
from preprocessing import load_image, preprocess


def no_mask_features(root, split):
    paths, y = list_split(root, split)
    return np.array([hsv_histogram(preprocess(load_image(p))[1], None) for p in paths]), y


def report(name, y_val, p_val, y_test, p_test):
    print(f"{name:<28} val acc {accuracy_score(y_val, p_val):.4f} | test acc {accuracy_score(y_test, p_test):.4f}"
          f"  P {precision_score(y_test, p_test):.4f}  R {recall_score(y_test, p_test):.4f}"
          f"  F1 {f1_score(y_test, p_test):.4f}  CM {confusion_matrix(y_test, p_test).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--model", default="models/model.pkl")
    args = ap.parse_args()

    X_tr, y_tr, _ = load_split(args.data, "train")
    X_va, y_va, _ = load_split(args.data, "val")
    X_te, y_te, _ = load_split(args.data, "test")
    scaler = StandardScaler().fit(X_tr)
    for C in (1, 10):
        svm = SVC(kernel="rbf", C=C).fit(scaler.transform(X_tr), y_tr)
        report(f"SVM (C = {C})", y_va, svm.predict(scaler.transform(X_va)), y_te, svm.predict(scaler.transform(X_te)))

    N_tr, _ = no_mask_features(args.data, "train")
    N_va, _ = no_mask_features(args.data, "val")
    N_te, _ = no_mask_features(args.data, "test")
    s2 = StandardScaler().fit(N_tr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42).fit(s2.transform(N_tr), y_tr)
    report("Random Forest (no mask)", y_va, rf.predict(s2.transform(N_va)), y_te, rf.predict(s2.transform(N_te)))

    b = joblib.load(args.model)
    report("Random Forest (mask)", y_va, b["model"].predict(b["scaler"].transform(X_va)),
           y_te, b["model"].predict(b["scaler"].transform(X_te)))
    print("(Precision / Recall / F1 คิดให้ rotten เป็นคลาสบวก)")


if __name__ == "__main__":
    main()
