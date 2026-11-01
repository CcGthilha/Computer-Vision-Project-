# Fruit Freshness Detection

ระบบ Computer Vision สำหรับตรวจสอบความสดของผลไม้ (Fresh / Rotten) โดยใช้ Image Processing ร่วมกับ Machine Learning แบบ Classical (ไม่ใช้ YOLO หรือ Deep Learning Object Detection) พัฒนาเป็น Web Application ด้วย Angular (Client-side Inference)

วิชา Image Processing and Computer Vision | ระยะเวลาโครงงาน 3 สัปดาห์

## Problem & Objective

**Problem:** การคัดแยกผลไม้สดและเน่าด้วยสายตามนุษย์ใช้เวลาและมีความคลาดเคลื่อน

**Objective:** พัฒนาระบบที่รับภาพผลไม้ ประมวลผลภาพ และจำแนกว่าเป็นผลไม้สด (Fresh) หรือเน่า (Rotten) ได้โดยอัตโนมัติ พร้อมแสดงผลผ่าน Web Application

## Dataset

| รายการ | รายละเอียด |
|---|---|
| ชื่อ Dataset | Fresh and Rotten Fruits |
| จำนวน Class | 2 (Fresh / Rotten) |
| แหล่งข้อมูล | Kaggle / Roboflow (ระบุแหล่งที่ใช้จริง) |
| การแบ่งข้อมูล | Train / Validation / Test (ระบุสัดส่วนที่ใช้จริง) |

> TODO: เติมจำนวนภาพจริงต่อคลาส และตัวอย่างภาพ dataset

## Methodology

Workflow ของโครงงาน:

```
Dataset → Data Preparation → Image Processing → Feature Extraction
        → Model Training (SVM / Random Forest) → Evaluation
        → Export to ONNX → Angular Web Application
```

### Image Processing
- **HSV Color Space** — แปลงภาพจาก RGB เป็น HSV เพื่อแยกลักษณะสีที่บ่งบอกความสดของผลไม้
- **Thresholding** — แยกส่วนผลไม้ (foreground) ออกจากพื้นหลัง

### Feature Extraction
- **HSV Histogram** — สกัด Feature จากการกระจายตัวของค่าสี (H, S, V) เพื่อใช้เป็น Input ของโมเดล

## Algorithm / Model

- **SVM (Support Vector Machine)** และ/หรือ **Random Forest** สำหรับจำแนกภาพเป็น Fresh / Rotten จาก HSV Histogram Feature ที่สกัดได้ (Classical Machine Learning ไม่ใช้ Deep Learning / YOLO)

## Evaluation

| Metric | ค่าที่ได้ |
|---|---|
| Accuracy | TODO |
| Precision | TODO |
| Recall | TODO |
| F1-score | TODO |
| Confusion Matrix | `results/confusion_matrix.png` |

## Application

Web Application พัฒนาด้วย **Angular** แบบ **Client-side Inference** (ไม่มี Backend API) — โมเดลถูกแปลงเป็น ONNX และฝังรันในเบราว์เซอร์ทั้งหมด รองรับ:
- Upload ภาพผลไม้
- ประมวลผลภาพ (HSV conversion + Thresholding) ด้วย Canvas API ในฝั่ง client
- สกัด HSV Histogram feature ด้วย JavaScript/TypeScript
- โหลดโมเดล `model.onnx` ด้วย `onnxruntime-web` แล้วทำนายผลในเบราว์เซอร์
- แสดงผลลัพธ์ (Fresh / Rotten) พร้อมค่าความมั่นใจ (confidence)
- แสดงสถิติ/ข้อมูลที่เกี่ยวข้องกับการตรวจจับ

> หมายเหตุ: HSV Histogram ที่คำนวณตอน Train (Python) และตอน Predict (JS ใน Angular) ต้องให้ผลตรงกัน (bin size, normalization เหมือนกัน) ต้องทดสอบเทียบค่าก่อน deploy จริง

### Application Screenshot

> TODO: แนบภาพหน้าจอ Application ที่ `images/result.jpg`

## Project Structure

```
project-name/
├── frontend/                       # Angular project (client-side inference ทั้งหมด)
│   ├── src/
│   │   ├── app/
│   │   │   ├── hsv-processing.ts   # คำนวณ HSV + Histogram ด้วย Canvas API
│   │   │   └── model-inference.ts  # โหลด model.onnx + predict
│   │   └── assets/
│   │       └── model.onnx          # โมเดลที่แปลงจาก sklearn → ONNX
│   └── package.json
├── src/                             # โค้ด Python ฝั่ง Training เท่านั้น
│   ├── preprocessing.py            # HSV conversion, Thresholding
│   ├── feature_extraction.py       # HSV Histogram
│   ├── classification.py           # Train SVM / Random Forest
│   └── export_onnx.py              # แปลงโมเดล sklearn → ONNX
├── requirements.txt
├── README.md
├── images/
│   ├── input.jpg
│   └── result.jpg
├── results/
│   ├── confusion_matrix.png
│   └── classification_report.png
└── demo/
    └── demo.mp4
```

## How to Run

### 1. Train Model และแปลงเป็น ONNX (Python)

```bash
git clone <repository-url>
cd project-name
pip install -r requirements.txt

python src/classification.py      # Train SVM / Random Forest
python src/export_onnx.py         # แปลงโมเดล → model.onnx แล้วย้ายไป frontend/src/assets/
```

### 2. รัน Angular Web Application

```bash
cd frontend
npm install
ng serve
```

เปิดเบราว์เซอร์ที่ `http://localhost:4200` แล้วอัปโหลดภาพผลไม้เพื่อทดสอบ (ประมวลผลและทำนายทั้งหมดในเบราว์เซอร์ ไม่ต้องมี Backend Server)

## Technology Stack

| เครื่องมือ | หน้าที่ |
|---|---|
| Python | พัฒนาและ Train Model |
| OpenCV | Image Processing (HSV, Thresholding) ฝั่ง Training |
| scikit-learn | SVM / Random Forest, Evaluation |
| skl2onnx | แปลงโมเดล sklearn → ONNX |
| NumPy | จัดการข้อมูลภาพ/ฟีเจอร์ |
| Pillow | อ่านและจัดการภาพ |
| Angular | สร้าง Web Application (Frontend) |
| onnxruntime-web | รันโมเดล ONNX แบบ Client-side ในเบราว์เซอร์ |
| TypeScript / Canvas API | ประมวลผลภาพและสกัด HSV Histogram ฝั่ง Client |
| Git / GitHub | จัดเก็บ Source Code |

## Results

> TODO: สรุปผลลัพธ์ Model พร้อมตัวอย่างภาพ Before/After และผลการจำแนก

## Demo

> TODO: แนบลิงก์หรือไฟล์ `demo/demo.mp4`

## Team & Task Division

| คนที่ | หน้าที่ | งานที่ต้องทำ | Deliverable |
|---|---|---|---|
| 1 | Dataset & Data Preparation | หา/ดาวน์โหลด dataset, ทำความสะอาดข้อมูล, แบ่ง Train/Val/Test, สรุปข้อมูลลง README | Dataset พร้อมใช้ + เอกสารสรุป |
| 2 | Image Processing & Feature Extraction | เขียน HSV conversion, Thresholding, สกัด HSV Histogram feature | `src/preprocessing.py`, `src/feature_extraction.py` |
| 3 | Model Training & Evaluation | Train SVM/Random Forest, ประเมินผล (accuracy, precision, recall, F1, confusion matrix) | `src/classification.py`, `model.pkl`, `results/confusion_matrix.png` |
| 4 | Angular Web App (Client-side Inference) | แปลงโมเดลเป็น ONNX, เขียน Angular app ประมวลผล HSV histogram + predict ในเบราว์เซอร์, จัดการ GitHub/README, อัดวิดีโอ demo | `frontend/`, `model.onnx`, repo ที่ใช้งานได้จริง, `demo/demo.mp4` |

> TODO: ระบุรายชื่อสมาชิกกลุ่มจริงแทนตำแหน่งที่ 1-4
