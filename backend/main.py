
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .ecg_pipeline import analyze_wfdb_zip, CLASS_NAMES
from .image_service import digitize, engine_status

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_ecg_hgb_model.joblib"
FEATURES_PATH = ROOT / "models" / "final_feature_list.json"
CONFIG_PATH = ROOT / "models" / "final_model_config.json"
REPORT_DIR = ROOT / "results" / "final_report_package"

app = FastAPI(
    title="SmartECG-ML API",
    version="4.0.0",
    description="SmartECG-ML API with WFDB and production ECG-image digitization.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Feature list not found: {FEATURES_PATH}")

model = joblib.load(MODEL_PATH)
feature_data = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
FEATURES = (
    feature_data
    if isinstance(feature_data, list)
    else feature_data.get("features") or feature_data.get("selected_features")
)
if not FEATURES:
    raise ValueError("Could not read final feature list.")

CONFIG = (
    json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if CONFIG_PATH.exists() else {}
)
CLASSES = [str(c) for c in model.classes_]


class PredictionRequest(BaseModel):
    features: dict[str, float]


def feature_prediction(features: dict[str, float]) -> dict:
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"error": "Missing required features.", "missing_features": missing},
        )

    values = {}
    for feature in FEATURES:
        try:
            value = float(features[feature])
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail={"error": f"Invalid value for feature '{feature}'."},
            )
        if not np.isfinite(value):
            raise HTTPException(
                status_code=400,
                detail={"error": f"Feature '{feature}' must be finite."},
            )
        values[feature] = value

    X = pd.DataFrame([values], columns=FEATURES)
    pred = str(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    probability_map = {
        cls: float(p) for cls, p in zip(CLASSES, probs)
    }

    return {
        "success": True,
        "prediction": pred,
        "description": CLASS_NAMES.get(pred, "Unknown"),
        "probability": probability_map[pred],
        "probabilities": probability_map,
        "features_used": len(FEATURES),
    }


@app.get("/")
def root():
    return {
        "application": "SmartECG-ML",
        "status": "running",
        "version": "3.0.0",
        "model": type(model).__name__,
        "feature_count": len(FEATURES),
        "classes": CLASSES,
        "docs": "/docs",
        "image_digitizer": engine_status(),
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(FEATURES),
        "classes": CLASSES,
        "image_digitizer": engine_status(),
    }


@app.get("/api/model-info")
def model_info():
    return {
        "model": type(model).__name__,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "classes": CLASSES,
        "class_descriptions": CLASS_NAMES,
        "configuration": CONFIG,
    }


@app.post("/api/predict")
def predict(request: PredictionRequest):
    return feature_prediction(request.features)


@app.get("/api/test-prediction")
def test_prediction():
    return feature_prediction({f: 0.0 for f in FEATURES})




def _check_upload_size(data: bytes, filename: str) -> None:
    lower = (filename or "").lower()
    limit = 25 * 1024 * 1024 if lower.endswith(".pdf") else 15 * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload is larger than the {limit // (1024 * 1024)} MB limit for this file type.",
        )


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

@app.post("/api/predict-wfdb")
async def predict_wfdb(
    file: UploadFile = File(...),
    max_beats: int = Query(5000, ge=1, le=10000),
):
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Upload a ZIP containing WFDB .hea, .dat and .atr files.",
        )

    data = await file.read()
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Upload is larger than the 100 MB local-server limit.",
        )

    try:
        return analyze_wfdb_zip(
            data,
            model=model,
            feature_list=FEATURES,
            max_results=max_beats,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/analyze-upload")
async def analyze_upload(
    file: UploadFile = File(...),
    max_beats: int = Query(5000, ge=1, le=10000),
):
    """Analyze either a WFDB ZIP or a paper ECG image/PDF.

    Images/PDFs are digitized into temporary WFDB .hea/.dat/.atr files, zipped,
    and then passed through the same existing SmartECG-ML inference pipeline.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    data = await file.read()
    _check_upload_size(data, filename)

    try:
        if suffix == ".zip":
            result = await run_in_threadpool(
                analyze_wfdb_zip,
                data,
                model=model,
                feature_list=FEATURES,
                max_results=max_beats,
            )
            result["input_mode"] = "WFDB ZIP"
            result["conversion"] = None
            return result

        if suffix in ALLOWED_IMAGE_EXTENSIONS:
            wfdb_zip, conversion = await run_in_threadpool(digitize, data, filename)
            result = await run_in_threadpool(
                analyze_wfdb_zip,
                wfdb_zip,
                model=model,
                feature_list=FEATURES,
                max_results=max_beats,
            )
            result["input_mode"] = "ECG image/PDF → WFDB → SmartECG-ML"
            result["conversion"] = conversion
            return result

        raise HTTPException(
            status_code=400,
            detail="Unsupported file. Upload PNG, JPG, JPEG, PDF, or a WFDB ZIP containing .hea/.dat/.atr.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/digitize-image")
async def digitize_image(
    file: UploadFile = File(...),
):
    """Convert a paper ECG image/PDF to a WFDB ZIP for research/debugging."""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload PNG, JPG, JPEG, or PDF.")
    data = await file.read()
    _check_upload_size(data, filename)
    try:
        wfdb_zip, _ = await run_in_threadpool(digitize, data, filename)
        return Response(
            content=wfdb_zip,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=smart_ecg_digitized_wfdb.zip"},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/image-digitizer")
def image_digitizer_status():
    return engine_status()


@app.get("/api/performance")
def performance():
    result = {
        "accuracy": 0.863677,
        "balanced_accuracy": 0.597188,
        "macro_precision": 0.5269,
        "macro_recall": 0.5972,
        "macro_f1": 0.547298,
        "weighted_f1": 0.876315,
        "classes": {},
    }

    metrics_file = REPORT_DIR / "03_per_class_metrics.csv"
    if not metrics_file.exists():
        metrics_file = ROOT / "results" / "step47B_nested_class_metrics.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        for _, row in df.iterrows():
            cls = str(row.get("class", row.get("label", "")))
            result["classes"][cls] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "support": int(row["support"]),
            }

    return result
