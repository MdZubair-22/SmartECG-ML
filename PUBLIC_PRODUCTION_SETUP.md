# SmartECG-ML — serious local/public deployment setup

## 1. Core API environment

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

## 2. Install the ECGLight image digitization stack

Install Git, then:

```powershell
.\tools\setup_ecg_image_stack.ps1
```

The runtime engine is the official [SCAI-Lab ECGLight repository](https://github.com/SCAI-Lab/ECGLight). ECG-Image-Kit is used only for synthetic image generation and QA. The old vendored ECG-Digitiser checkout is isolated and is never used at runtime.

Create a separate Python 3.11 environment with ECGLight's documented dependencies. Then download its official model archive from the Polybox link in `third_party\ECGLight\README.md`. Place these four files exactly here:

```text
third_party/ECGLight/models/digitization_models/yolo11_full/weights/best.pt
third_party/ECGLight/models/digitization_models/yolo11_lead/weights/best.pt
third_party/ECGLight/models/digitization_models/yolo11_pulse/weights/best.pt
third_party/ECGLight/models/digitization_models/yolo11_patch/weights/best.pt
```

Find the Python executable:

```powershell
python -c "import sys; print(sys.executable)"
```

Set it for the current terminal:

```powershell
$env:SMART_ECG_ECGLIGHT_PATH="C:\path\to\SmartECG-ML_PRODUCTION\third_party\ECGLight"
$env:SMART_ECG_ECGLIGHT_PYTHON="C:\path\to\ecglight\python.exe"
```

For a permanent Windows user environment, set these variables in System Properties → Environment Variables.

## 3. Start SmartECG

Terminal 1:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

## 4. Verify

Open:

`http://127.0.0.1:8000/api/health`

You want:

```json
"status": "healthy"
```

and:

```json
"image_digitizer": {
  "production_ready": true
}
```

If `production_ready` is false, image uploads are disabled; SmartECG does not fall back to a different digitizer.

## 5. Supported public inputs

- JPG / JPEG
- PNG
- PDF (first page)
- WFDB ZIP (`.hea + .dat + .atr`) for research testing

## 6. Privacy architecture

Uploaded image bytes are processed in temporary directories and are not intentionally persisted by the API. For an internet-facing deployment, add HTTPS, authentication, request rate limiting, encrypted storage if any data is retained, access logging without ECG contents, deletion policies, and a formal privacy/consent process before accepting real patient records.
