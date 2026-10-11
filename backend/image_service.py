from __future__ import annotations

import os

from .image_pipeline import digitize_ecg_image
from .production_digitizer import status as production_status, digitize_with_production_engine


def engine_status() -> dict:
    status = production_status()
    status["fallback_available"] = True
    status["fallback_enabled"] = os.getenv("SMART_ECG_ALLOW_CLASSICAL_FALLBACK", "1").lower() not in {"0", "false", "no"}
    status["active_engine"] = (
        "classical_fallback"
        if _prefer_classical_fallback() and status["fallback_enabled"]
        else status["engine"]
    )
    return status


def _prefer_classical_fallback() -> bool:
    configured = os.getenv("SMART_ECG_PREFER_CLASSICAL_FALLBACK")
    if configured is None:
        from .production_digitizer import _load_config

        configured = str(_load_config().get("prefer_classical_fallback", True))
    return configured.lower() not in {"0", "false", "no"}


def _run_classical_fallback(data: bytes, filename: str, reason: str | None = None):
    wfdb_zip, metadata = digitize_ecg_image(data, filename)
    metadata = dict(metadata)
    metadata["engine"] = "classical_fallback"
    if reason:
        metadata["fallback_reason"] = reason
        metadata["production_engine_error"] = reason
    return wfdb_zip, metadata


def digitize(data: bytes, filename: str):
    allow_fallback = os.getenv("SMART_ECG_ALLOW_CLASSICAL_FALLBACK", "1").lower() not in {"0", "false", "no"}

    if allow_fallback and _prefer_classical_fallback():
        return _run_classical_fallback(data, filename, "Classical fallback is preferred for responsive image analysis.")

    try:
        return digitize_with_production_engine(data, filename)
    except Exception as production_error:
        if not allow_fallback:
            raise RuntimeError(f"ECGLight production digitization failed and fallback is disabled: {production_error}") from production_error

        try:
            return _run_classical_fallback(data, filename, str(production_error))
        except Exception as fallback_error:
            raise RuntimeError(
                f"ECGLight production digitization failed and classical fallback failed. "
                f"Production error: {production_error}. Fallback error: {fallback_error}"
            ) from fallback_error
