"""ประเมินผล models/model.pkl บน Validation / Test set

    python src/evaluate.py --data dataset --model models/model.pkl

ผลลัพธ์:
    results/metrics.json
    results/confusion_matrix.png
    results/classification_report.png
    images/preprocessing_steps.png
"""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from dataset import CLASSES, list_split, load_split
from preprocessing import load_image, preprocess


def evaluate_split(bundle, data, split):
    X, y, _ = load_split(data, split)
    pred = bundle["model"].predict(bundle["scaler"].transform(X))
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro")
    return {
        "n_images": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "per_class": classification_report(y, pred, target_names=CLASSES, output_dict=True),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }, y, pred


def save_confusion_matrix(y, pred, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=CLASSES, cmap="Greens", ax=ax)
    ax.set_title("Confusion Matrix (Test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_report_table(metrics, path):
    rows = []
    for name in CLASSES:
        c = metrics["per_class"][name]
        rows.append([name, f"{c['precision']:.3f}", f"{c['recall']:.3f}", f"{c['f1-score']:.3f}", int(c["support"])])
    rows.append(["macro avg", f"{metrics['precision_macro']:.3f}", f"{metrics['recall_macro']:.3f}",
                 f"{metrics['f1_macro']:.3f}", metrics["n_images"]])
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["class", "precision", "recall", "f1-score", "support"],
                     loc="center", cellLoc="center")
    table.scale(1, 1.5)
    ax.set_title(f"Classification Report (Test) - accuracy {metrics['accuracy']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_preprocessing_steps(data, path):
    """ตัวอย่างขั้นตอน Image Processing: Original -> HSV -> Mask -> Masked fruit"""
    paths, _ = list_split(data, "test")
    samples = [paths[0], paths[-1]]  # fresh 1 ภาพ, rotten 1 ภาพ
    fig, axes = plt.subplots(len(samples), 4, figsize=(10, 2.8 * len(samples)))
    for row, p in zip(axes, samples):
        small, hsv, mask = preprocess(load_image(p))
        masked = cv2.bitwise_and(small, small, mask=mask)
        panels = [(small, "Original (128x128)"), (hsv, "HSV"), (mask, "Otsu (S) + Morphology"),
                  (masked, "Masked fruit")]
        for ax, (im, title) in zip(row, panels):
            if im.ndim == 2:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if title != "HSV" else im)
            ax.set_title(f"{title}\n{p.parent.name}" if title.startswith("Original") else title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--model", default="models/model.pkl")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("images").mkdir(exist_ok=True)
    bundle = joblib.load(args.model)

    metrics = {}
    for split in ("val", "test"):
        metrics[split], y, pred = evaluate_split(bundle, args.data, split)
        m = metrics[split]
        print(f"[{split}] n={m['n_images']} acc={m['accuracy']:.4f} "
              f"precision={m['precision_macro']:.4f} recall={m['recall_macro']:.4f} f1={m['f1_macro']:.4f}")

    save_confusion_matrix(y, pred, "results/confusion_matrix.png")
    save_report_table(metrics["test"], "results/classification_report.png")
    save_preprocessing_steps(args.data, "images/preprocessing_steps.png")
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(classification_report(y, pred, target_names=CLASSES, digits=4))


if __name__ == "__main__":
    main()
