from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from .image_pipeline import load_document, _detect_r_peaks, FS

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "image_digitizer.json"
DEFAULT_REPO = ROOT / "third_party" / "ECGLight"
WEIGHT_PATHS = {
    "full": Path("models/digitization_models/yolo11_full/weights/best.pt"),
    "lead": Path("models/digitization_models/yolo11_lead/weights/best.pt"),
    "pulse": Path("models/digitization_models/yolo11_pulse/weights/best.pt"),
    "patch": Path("models/digitization_models/yolo11_patch/weights/best.pt"),
}


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def engine_config() -> dict:
    cfg = _load_config()
    repo = Path(os.getenv("SMART_ECG_ECGLIGHT_PATH", str(ROOT / cfg.get("ecglight_path", "third_party/ECGLight")))).expanduser()
    repo = (ROOT / repo).resolve() if not repo.is_absolute() else repo.resolve()
    py = os.getenv("SMART_ECG_ECGLIGHT_PYTHON", cfg.get("ecglight_python", ".ecglight-venv/Scripts/python.exe"))
    py_path = Path(py).expanduser()
    py_path = (ROOT / py_path).resolve() if not py_path.is_absolute() else py_path.resolve()
    python_ok = bool(py_path and py_path.is_file())
    required_source = [repo / "digitization.py", repo / "config.py", repo / "backend" / "digitization_runner.py"]
    source_ok = all(path.is_file() for path in required_source)
    weights = {name: repo / relative for name, relative in WEIGHT_PATHS.items()}
    missing_weights = [str(path) for path in weights.values() if not path.is_file()]
    imports_ok = False
    import_error = None
    if python_ok:
        try:
            probe = subprocess.run(
                [
                    str(py_path),
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); import cv2, numpy, pandas, scipy, skimage, torch, wfdb; from ultralytics import YOLO; from patched_yolo_infer import MakeCropsDetectThem; from digitization import ECGImage",
                    str(repo),
                ],
                cwd=str(repo if repo.exists() else ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            imports_ok = probe.returncode == 0
            if not imports_ok:
                import_error = (probe.stderr or probe.stdout).strip()[-1000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            import_error = str(exc)
    return {
        "configured": source_ok and python_ok,
        "engine": "ECGLight",
        "repo": str(repo),
        "python": str(py_path),
        "source_ok": source_ok,
        "python_ok": python_ok,
        "imports_ok": imports_ok,
        "import_error": import_error,
        "weights": {name: str(path) for name, path in weights.items()},
        "missing_weights": missing_weights,
        "weights_ok": not missing_weights,
    }


def status() -> dict:
    s = engine_config()
    s["production_ready"] = bool(
        s["source_ok"] and s["python_ok"] and s["imports_ok"] and s["weights_ok"]
    )
    return s


def _pdf_or_image_to_png(data: bytes, filename: str, output: Path) -> None:
    image, _ = load_document(data, filename)
    import cv2
    if not cv2.imwrite(str(output), image):
        raise RuntimeError("Could not prepare the ECG image for ECGLight.")


def _select_signal(record) -> tuple[np.ndarray, str]:
    preferred = ["II", "MLII", "I"]
    names = [str(x).upper() for x in record.sig_name]
    for wanted in preferred:
        if wanted in names:
            return np.asarray(record.p_signal[:, names.index(wanted)], dtype=np.float64), record.sig_name[names.index(wanted)]
    return np.asarray(record.p_signal[:, 0], dtype=np.float64), record.sig_name[0]


def _make_internal_wfdb(record_base: Path, out_zip: io.BytesIO) -> dict:
    rec = wfdb.rdrecord(str(record_base), physical=True)
    signal, lead = _select_signal(rec)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    source_fs = float(rec.fs)
    if source_fs <= 0:
        raise RuntimeError("Digitizer output has an invalid sampling frequency.")
    if abs(source_fs - FS) > 1e-6:
        from fractions import Fraction
        ratio = Fraction(FS / source_fs).limit_denominator(1000)
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    signal = signal.astype(np.float64)
    # Preserve morphology but remove gross offset; the downstream model performs beat-wise normalization.
    signal -= np.median(signal)
    if np.std(signal) < 1e-8:
        raise RuntimeError("Digitized ECG has insufficient signal variation.")
    peaks = _detect_r_peaks(signal)
    peaks = np.asarray([int(p) for p in peaks if 90 <= p < len(signal) - 162], dtype=int)
    if len(peaks) < 3:
        raise RuntimeError("Digitized ECG did not contain enough reliable QRS detections for beat-level analysis.")
    peaks = np.asarray([p for i, p in enumerate(peaks) if i == 0 or 0.25 * FS <= p - peaks[i-1] <= 3.0 * FS], dtype=int)

    with tempfile.TemporaryDirectory(prefix="smart_ecg_runtime_") as td:
        root = Path(td)
        name = "image_ecg"
        wfdb.wrsamp(str(root / name), fs=FS, units=["mV"], sig_name=["MLII"], p_signal=signal.reshape(-1, 1), fmt=["16"], adc_gain=[1000], baseline=[0], comments=["Digitized by ECGLight at 500 Hz", f"source_lead={lead}"])
        wfdb.wrann(str(root / name), "atr", sample=peaks, symbol=["N"] * len(peaks), fs=FS)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext in (".hea", ".dat", ".atr"):
                p = root / f"{name}{ext}"
                if not p.exists():
                    raise RuntimeError(f"Missing generated WFDB file: {p.name}")
                zf.write(p, p.name)
    return {
        "engine": "ECGLight",
        "source_lead": lead,
        "source_sampling_frequency": source_fs,
        "sampling_frequency": FS,
        "detected_beats": int(len(peaks)),
        "duration_seconds": round(len(signal) / FS, 3),
        "generated_wfdb": ["image_ecg.hea", "image_ecg.dat", "image_ecg.atr"],
        "annotation_note": "The .atr file contains automatically detected QRS locations marked N for beat segmentation only; it is not a clinician annotation or diagnosis.",
    }


def digitize_with_production_engine(data: bytes, filename: str) -> tuple[bytes, dict]:
    s = status()
    if not s["source_ok"]:
        raise RuntimeError(f"ECGLight source is missing or incomplete: {s['repo']}")
    if not s["python_ok"]:
        raise RuntimeError(f"ECGLight Python environment not found: {s['python']}")
    if not s["imports_ok"]:
        raise RuntimeError("ECGLight environment is missing one or more required imports: cv2, numpy, pandas, scipy, skimage, torch, ultralytics, patched_yolo_infer, wfdb.")
    if not s["weights_ok"]:
        raise RuntimeError("ECGLight digitization weights are missing:\n" + "\n".join(s["missing_weights"]))

    cfg = _load_config()
    timeout = int(os.getenv("SMART_ECG_ECGLIGHT_TIMEOUT", cfg.get("max_runtime_seconds", 300)))
    with tempfile.TemporaryDirectory(prefix="smart_ecg_input_") as td:
        root = Path(td)
        image_path = root / "ecg.png"
        output_path = root / "digitized.csv"
        _pdf_or_image_to_png(data, filename, image_path)
        runner = """import os, sys
from pathlib import Path
repo, image_path, output_path, full_path, lead_path, pulse_path, patch_path = sys.argv[1:]
os.chdir(repo)
sys.path.insert(0, repo)
from ultralytics import YOLO
from digitization import ECGImage
ecg = ECGImage(
    box_model=YOLO(full_path),
    segmentation_model=patch_path,
    lead_name_model=YOLO(lead_path),
    pulse_model=YOLO(pulse_path),
    image_path=image_path,
)
ecg.run_full_pipeline()
ecg.save_signals_as_csv(Path(output_path).stem, directory=str(Path(output_path).parent))
"""
        model_args = [s["weights"][name] for name in ("full", "lead", "pulse", "patch")]
        cmd = [str(s["python"]), "-c", runner, s["repo"], str(image_path), str(output_path), *model_args]
        try:
            proc = subprocess.run(cmd, cwd=str(s["repo"]), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ECGLight timed out after {timeout} seconds.") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[-2500:]
            raise RuntimeError(f"ECGLight failed (exit {proc.returncode}). {detail}")
        if not output_path.is_file():
            raise RuntimeError("ECGLight completed without exporting its digitized waveform CSV.")
        digitized = pd.read_csv(output_path)
        lead_columns = {str(column).strip().upper(): column for column in digitized.columns}
        if "II" not in lead_columns:
            raise RuntimeError(f"ECGLight output has no Lead II waveform. Leads found: {list(digitized.columns)}")
        signal = pd.to_numeric(digitized[lead_columns["II"]], errors="coerce").to_numpy(dtype=np.float64)
        finite_samples = np.flatnonzero(np.isfinite(signal))
        if len(finite_samples):
            signal = signal[: finite_samples[-1] + 1]
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        if len(signal) < 500 or np.std(signal) < 1e-8:
            raise RuntimeError("ECGLight Lead II waveform is empty or has insufficient signal variation.")
        record_base = root / "ecglight_lead_ii"
        wfdb.wrsamp(
            str(record_base), fs=500, units=["mV"], sig_name=["II"],
            p_signal=signal.reshape(-1, 1), fmt=["16"],
            comments=["ECGLight digitized waveform", "source_lead=II"],
        )
        buf = io.BytesIO()
        meta = _make_internal_wfdb(record_base, buf)
        meta.update({
            "source_type": "pdf" if filename.lower().endswith(".pdf") else "image",
            "digitized_leads": list(digitized.columns),
            "digitized_samples_lead_ii": int(len(signal)),
            "digitized_sampling_frequency": 500,
            "lead_note": "ECGLight reconstructs calibrated 500 Hz leads; Lead II is passed through the existing SmartECG beat-classification pipeline.",
        })
        return buf.getvalue(), meta
