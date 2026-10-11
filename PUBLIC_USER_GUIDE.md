# SmartECG-ML — User Guide

## Current availability

WFDB ZIP and paper ECG image/PDF analysis are available. Paper ECGs use the built-in waveform digitizer by default. ECGLight can be selected when its runtime environment and official weights are installed.

## WFDB analysis

1. Open SmartECG-ML and choose **Analyze an ECG**.
2. Upload a ZIP containing matching `.hea`, `.dat`, and `.atr` files.
3. Review the overall model result, beat-class distribution, and probabilities.

The existing HGB classifier uses the unchanged 23-feature representation.

## Paper ECG analysis

Upload a JPG, PNG, or PDF. By default, the built-in digitizer extracts an ECG trace from the image and passes it through the SmartECG beat classifier. This method does not reconstruct all 12 leads. ECGLight can instead run YOLOv11 full-lead, lead-label, calibration-pulse, and patch segmentation and reconstruct calibrated 500 Hz signals; it requires a separate runtime environment and four official model weights. Set `SMART_ECG_PREFER_CLASSICAL_FALLBACK=0` to prefer ECGLight when it is configured.

The overall summary reports a normal pattern only when all analyzed beats are classified `N`; it reports an abnormal pattern when the model predicts `S`, `V`, or `F`; and it reports an inconclusive result when `Q` (Unknown / Other) occurs without an abnormal-class prediction.

## Image quality

- Photograph the whole ECG page.
- Keep the camera parallel to the paper.
- Use good lighting without shadows.
- Keep the red/gray grid visible.
- Avoid glare and fingers covering the waveform.
- Use the highest-resolution photo available.

## Medical safety

SmartECG-ML is a research/educational prototype. It is not clinically validated and should not be used to diagnose a patient or make treatment/emergency decisions.
