# MediScan AI — Chest X-ray Classification

An educational full-stack ML project demonstrating binary chest X-ray image classification (NORMAL vs PNEUMONIA) using MobileNetV2 transfer learning, a Flask API, SQLite history and a responsive dashboard.

> **Medical safety:** This is a student research prototype, not a medical device. It is not clinically validated and must not be used to diagnose, screen, triage or make treatment decisions. Model confidence is not the probability that a person has a disease. Never upload identifiable patient information.

## Features
- Responsive healthcare analytics dashboard
- PNG/JPG upload with type and 10 MB size validation
- TensorFlow/Keras MobileNetV2 transfer-learning pipeline
- NORMAL / PNEUMONIA model output and confidence display
- SQLite prediction history and dashboard statistics
- API health check and clear model-not-trained response
- Test evaluation metrics saved locally

## Stack
Python · Flask · TensorFlow/Keras · MobileNetV2 · SQLite · HTML · CSS · JavaScript

## Structure
- app.py — Flask routes and SQLite persistence
- model.py — model loading and inference
- train.py — training and held-out test evaluation
- templates/index.html — dashboard UI
- static/style.css and static/app.js — responsive frontend
- models/, data/, uploads/ — local-only generated/runtime assets

## Run locally
Python 3.10–3.12 is recommended. TensorFlow installation depends on your OS and Python version.

1. Clone: git clone https://github.com/gudla0098-arch/ml.git
2. Enter folder: cd ml
3. Create environment: python -m venv .venv
4. Activate on Windows PowerShell: .venv\Scripts\Activate.ps1
   On macOS/Linux: source .venv/bin/activate
5. Install: pip install -r requirements.txt
6. Download the Kaggle Chest X-Ray Images (Pneumonia) dataset:
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
7. Extract to data/chest_xray with this layout:
   data/chest_xray/train/NORMAL and PNEUMONIA
   data/chest_xray/val/NORMAL and PNEUMONIA
   data/chest_xray/test/NORMAL and PNEUMONIA
8. Train: python train.py
9. Run: python app.py
10. Open http://127.0.0.1:5000

The first training run downloads ImageNet weights. Training requires sufficient RAM and compute. The dataset and trained weights are deliberately not committed to GitHub.

## API
- GET /api/health — API and model readiness
- GET /api/stats — dashboard counts
- GET /api/history — latest 100 records
- POST /api/analyze — multipart upload using field name image

Example:
curl -X POST -F "image=@sample-xray.jpg" http://127.0.0.1:5000/api/analyze

## Responsible ML and limitations
Review accuracy, precision, recall, class balance and confusion matrix before interpreting results. A strong score on one dataset does not establish clinical performance. The dataset may contain label noise, demographic/acquisition bias and leakage risks. Serious research requires patient-level splits, external validation, calibration, subgroup analysis and expert review. Never represent this model as clinically accurate.

## Future work
Grad-CAM explainability, patient-level split, external validation, model versioning, authentication, access controls, retention policy, encrypted storage and production deployment hardening.

## License
Educational starter project. Check dataset terms and attribution requirements before redistributing images or model weights.
