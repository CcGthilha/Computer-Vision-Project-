"""Feature Extraction: 3D HSV Histogram (32 x 32 x 8 = 8192 bins) + L2 normalization"""
import cv2
import numpy as np

from preprocessing import preprocess

H_BINS, S_BINS, V_BINS = 32, 32, 8
N_FEATURES = H_BINS * S_BINS * V_BINS


def hsv_histogram(hsv, mask=None):
    if mask is not None and cv2.countNonZero(mask) == 0:
        mask = None  # ไม่พบผลไม้ -> ใช้ทั้งภาพ
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [H_BINS, S_BINS, V_BINS],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)  # L2 norm
    return hist.flatten()


def extract_features(img_bgr):
    _, hsv, mask = preprocess(img_bgr)
    return hsv_histogram(hsv, mask)
