# SmartECG-ML Frontend

React + Vite frontend for the SmartECG-ML local research application.

## User inputs

The Prediction page accepts these file formats:

- JPG / JPEG — photo of a printed ECG
- PNG — scan/image of a printed ECG
- PDF — ECG document; first page is analyzed
- ZIP — WFDB research record containing `.hea`, `.dat`, `.atr`

WFDB ZIP analysis is available. Paper-image uploads are accepted only when `/api/health` reports `image_digitizer.production_ready: true`; that requires the ECGLight runtime and all four official YOLOv11 digitization weights. Otherwise the UI disables paper-image analysis and the backend rejects it without selecting a fallback engine.

## Run

```bash
npm install
npm run dev
```

The Vite proxy sends `/api` requests to `http://127.0.0.1:8000`.
