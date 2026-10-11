$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$kit = Join-Path $root 'third_party/ECG-Image-Kit'
$generator = Join-Path $root 'third_party/ECG-Image-Kit/codes/ecg-image-generator'
if (-not (Test-Path $kit)) { throw 'ECG-Image-Kit is not installed. Run tools/setup_ecg_image_stack.ps1 first.' }
if (-not (Test-Path $generator)) { throw 'ECG-Image-Kit generator is not installed.' }
Write-Host 'ECG-Image-Kit synthetic generator is available.' -ForegroundColor Green
Write-Host 'Use the generator README/environment for PTB-XL based synthetic images.'
Write-Host 'Recommended QA variations: rotation, black/white conversion, random header, calibration pulse, wrinkles/creases and shadows.'
