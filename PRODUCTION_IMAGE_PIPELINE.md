# SmartECG-ML paper-ECG pipeline

## Architecture

`JPG/PNG/PDF -> ECGLight YOLOv11 digitization -> calibrated 500 Hz leads -> Lead II WFDB -> 360 Hz preprocessing/QRS/beat extraction -> existing 23-feature HGB model`

Runtime source: `third_party/ECGLight` (https://github.com/SCAI-Lab/ECGLight). Its official `ECGImage` pipeline uses `yolo11_full`, `yolo11_lead`, `yolo11_pulse`, and `yolo11_patch`, then performs calibration/grid reconstruction and emits 500 Hz signals. SmartECG passes reconstructed Lead II through the existing WFDB and beat-classification pipeline.

ECG-Image-Kit is used only for synthetic image generation and QA. The legacy vendored ECG-Digitiser code is not a runtime option. Image uploads are refused until the ECGLight environment and all four official weights are present.

## Installation

1. Install Git.
2. Run `tools/setup_ecg_image_stack.ps1`.
3. Create an ECGLight-compatible Python 3.11 environment and install the imports listed by its `environment.yml`.
4. Download the official model archive linked in `third_party/ECGLight/README.md` and place the digitization models at `third_party/ECGLight/models/digitization_models/`.
5. Set `SMART_ECG_ECGLIGHT_PATH` and `SMART_ECG_ECGLIGHT_PYTHON`, then restart the API.
6. `/api/health` should report `image_digitizer.production_ready: true` only when all four weights and runtime imports are available.

## Why a separate environment?

The digitizer uses PyTorch and Ultralytics. Keeping it separate from the FastAPI environment avoids dependency conflicts with the existing scikit-learn model environment.

## Safety/clinical boundary

The generated `.atr` is an internal QRS-location annotation used to segment beats for the existing classifier. It is not a clinician annotation. The existing SmartECG classifier was trained on MIT-BIH beat labels and is not a validated 12-lead diagnostic model. Do not present its output as a medical diagnosis.
