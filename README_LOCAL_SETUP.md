# SmartECG-ML — Local Application Setup

## What this version does

SmartECG-ML has two input paths:

1. **WFDB ZIP:** available now for research records containing matching `.hea`, `.dat`, and `.atr` files.
2. **Paper ECG image/PDF:** uses the built-in waveform digitizer by default. It can use ECGLight when its dedicated environment and all four official YOLOv11 digitization weights are installed.

ECG-Image-Kit is used only for synthetic ECG image generation and QA. The vendored ECG-Digitiser source is isolated and is not called by SmartECG runtime code.

ECG beat classes are summarized as normal when every analyzed beat is `N`, abnormal when the model predicts `S`, `V`, or `F`, and inconclusive when `Q` (Unknown / Other) is present without an abnormal-class prediction. This is a research model output, not a medical diagnosis.

## Requirements

- Windows
- Python 3.10/3.11 recommended
- Node.js + npm
- Existing model files under `models/`

## 1. Create Python environment

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

If an old `.venv` points to a previous project location, delete it and recreate it.

## 2. Start backend

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Backend:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs

## 3. Start frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The Vite proxy sends `/api` requests to `http://127.0.0.1:8000`, so either localhost:5173 or localhost:5174 is fine.

## 4. Using the site as a normal user

### Paper ECG (when ECGLight is ready)

Take a clear, straight-on photo of the entire ECG paper, or scan it. Upload:

- `.jpg`
- `.jpeg`
- `.png`
- `.pdf`

The built-in digitizer is used by default and extracts a waveform from the image. To select ECGLight instead, set `SMART_ECG_PREFER_CLASSICAL_FALLBACK=0`, then configure its separate Python environment and four official model weights. ECGLight returns calibrated 500 Hz leads; SmartECG selects Lead II and passes it through the existing 360 Hz preprocessing, R-peak detection, beat extraction, 23-feature HGB pipeline.

The built-in image digitizer selects a trace and does not reconstruct all 12 leads. Check the digitizer and image-quality notes in the result before interpreting it.

### WFDB research input

Upload a ZIP containing:

```text
100.hea
100.dat
100.atr
```

The backend reads the WFDB record, uses lead 0, applies the same 0.5–40 Hz preprocessing, creates 252-sample beat windows, calculates the 23 final features, and runs the frozen model.

## 5. Download the generated WFDB

After an image passes through either digitizer, the Prediction page provides **Download generated WFDB ZIP** for inspection.

## 6. Demo image

A synthetic paper-style test image is included at:

```text
demo/sample_ecg_paper_like.png
```

It is only a software test image and is not a patient ECG.

## Model

`models/final_ecg_hgb_model.joblib`

The API reads `model.classes_` dynamically, so probability columns are not assumed to be in a fixed order.

## Validation headline

Nested patient-wise hyperparameter validation with the frozen 23-feature set:

- Accuracy: 86.37%
- Balanced Accuracy: 59.72%
- Macro F1: 54.73%
- Weighted F1: 87.63%

These are research validation results on the project's dataset, not clinical performance claims.

## Dataset

The training project contains:

- 109,395 processed ECG beats
- 47 patients
- 48 records
- 5 classes: N, S, V, F, Q

The original dataset files are not included in this distribution.

## Important medical disclaimer

**Research / Educational Use Only.** This system has not been clinically validated and must not be used for diagnosis, treatment decisions, or emergency medical decisions. A patient should discuss an ECG with a qualified healthcare professional.

## Why ECG-image digitization is a separate research problem

Paper ECGs are common enough that PhysioNet's 2024 Challenge explicitly created a task for digitizing photographed/scanned ECGs into time-series data and classifying them. The challenge also released ECG-Image-Kit and an ECG-Image-Database to support this research.

Paper-image runtime digitization uses ECGLight when its official source, environment, and four YOLOv11 weights are installed. ECG-Image-Kit is reserved for synthetic image generation and QA. The legacy ECG-Digitiser checkout is isolated and is not selected by the application.
