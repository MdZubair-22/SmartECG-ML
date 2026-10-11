$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path 'third_party' | Out-Null

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required.' }

if (-not (Test-Path 'third_party/ECG-Image-Kit/.git')) {
  git clone --depth 1 https://github.com/alphanumericslab/ecg-image-kit.git third_party/ECG-Image-Kit
}

if (-not (Test-Path 'third_party/ECGLight/.git')) {
  git clone --depth 1 https://github.com/SCAI-Lab/ECGLight.git third_party/ECGLight
}

Write-Host ''
Write-Host 'ECGLight and ECG-Image-Kit source checkouts are ready.' -ForegroundColor Green
Write-Host 'Create a Python 3.11 ECGLight environment using third_party/ECGLight/environment.yml.'
Write-Host 'Download the official model archive from the Polybox link in third_party/ECGLight/README.md.'
Write-Host 'Place the four digitization checkpoints at:'
Write-Host '  third_party/ECGLight/models/digitization_models/yolo11_full/weights/best.pt'
Write-Host '  third_party/ECGLight/models/digitization_models/yolo11_lead/weights/best.pt'
Write-Host '  third_party/ECGLight/models/digitization_models/yolo11_pulse/weights/best.pt'
Write-Host '  third_party/ECGLight/models/digitization_models/yolo11_patch/weights/best.pt'
Write-Host 'Set SMART_ECG_ECGLIGHT_PYTHON to that environment Python executable.'
Write-Host ''
