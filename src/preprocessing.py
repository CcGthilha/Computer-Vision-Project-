"""Image Processing: Resize -> HSV Color Space -> Thresholding (Otsu) -> Morphology.

ทุกขั้นตอนต้องตรงกับ frontend/src/app/hsv-processing.ts
"""
import cv2
import numpy as np

IMAGE_SIZE = 128
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def load_image(path):
    """อ่านภาพเป็น BGR (รองรับ path ภาษาไทยบน Windows)"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"อ่านภาพไม่ได้: {path}")
    return img


def resize(img_bgr):
    return cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def to_hsv(img_bgr):
    """OpenCV HSV: H = 0-179, S = 0-255, V = 0-255"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


def fruit_mask(hsv):
    """แยกผลไม้ออกจากพื้นหลัง: Otsu threshold บนช่อง Saturation
    (พื้นหลังสีขาว/เทามีค่า S ต่ำ) แล้วใช้ Morphology Close + Open ลบรูและจุดรบกวน"""
    _, mask = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    return mask


def preprocess(img_bgr):
    """คืนค่า (ภาพที่ resize แล้ว, ภาพ HSV, mask)"""
    small = resize(img_bgr)
    hsv = to_hsv(small)
    return small, hsv, fruit_mask(hsv)
