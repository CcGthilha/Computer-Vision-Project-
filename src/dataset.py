"""โหลด Dataset แบบโฟลเดอร์: dataset/{train,val,test}/{fresh,rotten}/*.jpg"""
from pathlib import Path

import numpy as np

from feature_extraction import extract_features
from preprocessing import load_image

CLASSES = ["fresh", "rotten"]  # index = label ของโมเดล (0 = fresh, 1 = rotten)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_split(root, split):
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        files = sorted(p for p in (Path(root) / split / name).iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
        paths += files
        labels += [label] * len(files)
    return paths, np.array(labels)


def load_split(root, split):
    """คืนค่า (X features, y labels, paths)"""
    paths, y = list_split(root, split)
    X = np.array([extract_features(load_image(p)) for p in paths])
    return X, y, paths
