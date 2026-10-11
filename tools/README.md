# Production image stack tools

Run from the project root in PowerShell:

```powershell
.\tools\setup_ecg_image_stack.ps1
```

This installs the upstream ECGLight and ECG-Image-Kit repositories. ECGLight is the runtime digitizer; ECG-Image-Kit is used only for synthetic images and QA. The legacy ECG-Digitiser checkout, if present, is not used.

Set `SMART_ECG_ECGLIGHT_PATH` to `third_party/ECGLight` and `SMART_ECG_ECGLIGHT_PYTHON` to the Python executable of the dedicated ECGLight environment. Download and place all four digitization weights as listed in `PUBLIC_PRODUCTION_SETUP.md`.
