@echo off
cd /d "%~dp0"
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
)

if "%PYTHON_EXE%"=="" (
  echo ERROR: SmartECG API .venv not found.
  echo Create one with: python -m venv .venv
  pause
  exit /b 1
)

if "%SMART_ECG_ECGLIGHT_PATH%"=="" set "SMART_ECG_ECGLIGHT_PATH=%cd%\third_party\ECGLight"
if "%SMART_ECG_ECGLIGHT_PYTHON%"=="" set "SMART_ECG_ECGLIGHT_PYTHON=%cd%\.ecglight-venv\Scripts\python.exe"

%PYTHON_EXE% -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
