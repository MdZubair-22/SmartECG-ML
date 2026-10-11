# SmartECG-ML

SmartECG-ML is a local research prototype for ECG beat classification. It provides a React web interface and a FastAPI backend that analyzes WFDB records or ECG images and PDFs, then applies a trained HistGradientBoosting classifier.

> **Research and education only.** This project is not clinically validated and must not be used to diagnose, treat, or make emergency medical decisions. Have ECGs interpreted by a qualified healthcare professional.

## Features

- Upload a WFDB ZIP with matching `.hea`, `.dat`, and `.atr` files.
- Upload a paper ECG as JPG, JPEG, PNG, or PDF.
- Digitize paper ECGs with the built-in image digitizer by default; optionally use ECGLight when its separate runtime and official model weights are installed.
- Review beat-level predictions, class probabilities, prediction distribution, and an overall model summary.
- Download the generated WFDB files after image digitization.

## Overall model summary

The summary is calculated from the model's predictions for analyzed beats:

| Summary | Rule |
| --- | --- |
| **Normal pattern detected** | All analyzed beats are classified `N`. |
| **Abnormal pattern detected** | At least one beat is classified `S`, `V`, or `F`. |
| **Inconclusive** | A beat is classified `Q` (Unknown / Other), with no `S`, `V`, or `F` prediction, or there are no usable predictions. |

These labels describe the model output only; they are not clinical conclusions. A paper-image digitizer can also affect the quality of the resulting beat predictions.

## Requirements

- Windows (commands below use PowerShell)
- Python 3.10 or 3.11 recommended
- Node.js and npm

The trained classifier and its feature configuration are included under `models/`. The large ECG datasets, generated analysis outputs, Python virtual environments, and separately licensed third-party source checkouts are not included in this repository. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) before installing or using third-party components.

## Run locally

Open two terminals at the project root.

### 1. Install backend dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### 2. Start the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The API is at <http://127.0.0.1:8000>. Interactive API documentation is at <http://127.0.0.1:8000/docs>.

### 3. Install and start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (usually <http://127.0.0.1:5173>). The Vite development server proxies `/api` requests to the backend on port 8000.

Alternatively, after installing dependencies, run `start_backend.bat` and `start_frontend.bat` from the project root in separate terminals.

## Image digitization engines

The built-in digitizer is the default because it does not wait on ECGLight model inference. It extracts a visible waveform trace from an ECG image and does **not** reconstruct all 12 leads. Use a clear, high-resolution, straight-on scan or photo with the waveform visible.

To prefer ECGLight instead, configure its separate Python 3.11 environment and four official digitization weights, then set:

```powershell
$env:SMART_ECG_PREFER_CLASSICAL_FALLBACK="0"
$env:SMART_ECG_ECGLIGHT_PATH="C:\path\to\third_party\ECGLight"
$env:SMART_ECG_ECGLIGHT_PYTHON="C:\path\to\ecglight\python.exe"
```

The ECGLight model archive is distributed separately upstream. Review the applicable upstream terms before use. Further setup details are in [PUBLIC_PRODUCTION_SETUP.md](./PUBLIC_PRODUCTION_SETUP.md).

## API quick reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | API, model, and image digitizer status |
| `GET` | `/api/model-info` | Model classes, feature list, and configuration |
| `GET` | `/api/performance` | Validation metrics and per-class metrics |
| `POST` | `/api/analyze-upload` | Analyze an ECG image/PDF or WFDB ZIP |
| `POST` | `/api/digitize-image` | Convert an ECG image/PDF to a WFDB ZIP |
| `POST` | `/api/predict` | Predict from a JSON object containing the required features |

For example, submit an image with `curl.exe`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/analyze-upload" `
  -F "file=@demo\sample_ecg_paper_like.png"
```

The sample image is synthetic test data, not a patient ECG.

## Project layout

```text
backend/       FastAPI API, ECG processing, and digitization
config/        Image digitizer settings
demo/          Synthetic sample ECG images
frontend/      React and Vite interface
models/        Trained classifier and feature configuration
notebooks/     Research and model-development scripts
tests/         Backend unit tests
tools/         Local ECG image-stack setup utilities
```

## Tests and build

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Build the frontend:

```powershell
cd frontend
npm run build
```

## Validation context

The reported nested patient-wise validation metrics (accuracy 86.37%, balanced accuracy 59.72%, macro F1 54.73%, weighted F1 87.63%) are research results for this project's dataset. They are not evidence of clinical safety or generalization to other populations, devices, or digitization conditions.

## More documentation

- [Local setup and workflow details](./README_LOCAL_SETUP.md)
- [User guide](./PUBLIC_USER_GUIDE.md)
- [Production/runtime setup](./PUBLIC_PRODUCTION_SETUP.md)
- [Third-party notices](./THIRD_PARTY_NOTICES.md)
