# ECG image stack

SmartECG-ML separates image synthesis from production digitization:

1. **ECG-Image-Kit** — used only for synthetic ECG-image generation and QA.
   Repository: https://github.com/alphanumericslab/ecg-image-kit
2. **ECGLight** — the production ECG-image digitizer.
   Repository: https://github.com/SCAI-Lab/ECGLight

ECGLight uses four YOLOv11 checkpoints for full-lead boxes, lead labels, calibration pulses, and patch segmentation. It calibrates and reconstructs each lead as a 500 Hz waveform. The official model archive is linked from the ECGLight README and is distributed separately from source.

The vendored `third_party/ECG-Digitiser` directory is legacy source only. It is not imported, configured, or selected by SmartECG runtime code. ECGLight's upstream repository is licensed for non-commercial academic and research use; review its license before deployment.
