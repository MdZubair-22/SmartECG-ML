from __future__ import annotations

import io
import math
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import wfdb
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt, resample_poly

FS = 360
DEFAULT_DURATION_SECONDS = 10.0


def load_document(data: bytes, filename: str) -> tuple[np.ndarray, dict]:
    """Load the first PDF page or an image into BGR OpenCV format."""
    name = (filename or "upload").lower()
    if name.endswith(".pdf") or data[:4] == b"%PDF":
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            if len(doc) == 0:
                raise ValueError("The PDF has no pages.")
            page = doc[0]
            # 180 dpi is a good local compromise between OCR-like detail and memory.
            pix = page.get_pixmap(matrix=fitz.Matrix(180 / 72, 180 / 72), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img, {"source_type": "pdf", "pages": len(doc), "page_used": 1}
        except Exception as exc:
            raise ValueError(f"Could not read the PDF: {exc}") from exc

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The uploaded file is not a readable JPG/PNG image or PDF.")
    return img, {"source_type": "image", "pages": 1, "page_used": 1}


def _deskew(image: np.ndarray) -> np.ndarray:
    """Small deskew correction based on non-white ink; conservative by design."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = gray < 220
    ys, xs = np.where(ink)
    if len(xs) < 500:
        return image
    angle = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 8:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)



def _waveform_component(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], str]:
    """Isolate the widest ECG trace component.

    Real paper ECGs usually have a long connected trace, while the red/gray
    grid and printed text are either a different color or much less horizontally
    continuous. The method therefore prefers a component with a large horizontal
    span rather than simply taking the darkest pixels.
    """
    b, g, r = [c.astype(np.int16) for c in cv2.split(image)]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    red_grid = (r > g * 1.10) & (r > b * 1.10) & (r > 130)
    # Colored trace (blue/green/cyan) and dark neutral trace (black).
    colored = (sat > 45) & ~red_grid & (gray < 235)
    neutral = (gray < 145) & ~red_grid
    mask = colored | neutral

    # Remove page borders and long straight grid lines. The trace survives because
    # it is not a long horizontal/vertical line.
    u = mask.astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, image.shape[1] // 35), 1))
    hlines = cv2.morphologyEx(u, cv2.MORPH_OPEN, hk)
    # Do not remove vertical structures here: steep QRS complexes are vertical
    # in the image. Red grids are already suppressed by color; remaining page
    # borders are handled by connected-component span selection.
    u = u & ~hlines
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Suppress a little of the header/footer before connected components.
    h, w = u.shape
    u[: int(h * 0.05)] = 0
    u[int(h * 0.97):] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(u, 8)
    candidates = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        xspan = cw / max(1, w)
        if area < 80 or xspan < 0.10:
            continue
        # Strongly reward horizontal coverage. This picks a 10-second rhythm strip
        # when present on a 12-lead page, otherwise the widest lead strip.
        score = (xspan ** 2) * math.log1p(float(area))
        candidates.append((score, i, x, y, cw, ch, area))

    if not candidates:
        raise ValueError(
            "Could not isolate an ECG waveform from this image. "
            "Use a clear photo/scan with the ECG trace and paper grid visible."
        )

    candidates.sort(reverse=True)
    _, idx, x, y, cw, ch, area = candidates[0]
    component = labels == idx
    method = "connected-component waveform extraction"
    return component, (int(x), int(y), int(cw), int(ch)), method


def _trace_component(component: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a connected ECG trace into a 1-D pixel waveform.

    Steep QRS complexes can occupy many pixels in one image column. For those
    columns we use the local baseline plus the appropriate upper/lower endpoint,
    preserving the sharp peak rather than taking the median of a vertical stroke.
    """
    x0, y0, w, h = bbox
    ys, xs = np.where(component)
    if len(xs) == 0:
        raise ValueError("Empty ECG waveform component.")

    xmin, xmax = int(xs.min()), int(xs.max())
    width = xmax - xmin + 1
    centers = np.full(width, np.nan, dtype=np.float64)
    mins = np.full(width, np.nan, dtype=np.float64)
    maxs = np.full(width, np.nan, dtype=np.float64)
    spans = np.zeros(width, dtype=np.float64)

    for x in range(xmin, xmax + 1):
        yy = ys[xs == x]
        if len(yy):
            centers[x - xmin] = float(np.median(yy))
            mins[x - xmin] = float(np.min(yy))
            maxs[x - xmin] = float(np.max(yy))
            spans[x - xmin] = float(np.max(yy) - np.min(yy))

    valid = np.isfinite(centers)
    if valid.sum() < max(100, int(width * 0.35)):
        raise ValueError("The ECG trace is too fragmented to digitize reliably.")

    # Fill small gaps and estimate the baseline from non-vertical columns.
    centers[~valid] = np.nanmedian(centers[valid])
    centers = gaussian_filter1d(centers, sigma=1.2)
    non_spike = spans < max(7.0, h * 0.035)
    baseline = float(np.nanmedian(centers[non_spike])) if np.any(non_spike) else float(np.nanmedian(centers))

    signal = centers.copy()
    spike = spans > max(10.0, h * 0.05)
    # Group adjacent spike columns.
    idxs = np.where(spike)[0]
    groups: list[tuple[int, int]] = []
    if len(idxs):
        s = prev = int(idxs[0])
        for q in idxs[1:]:
            q = int(q)
            if q > prev + 1:
                groups.append((s, prev))
                s = q
            prev = q
        groups.append((s, prev))

    for s, e in groups:
        lo = float(np.nanmin(mins[s:e + 1]))
        hi = float(np.nanmax(maxs[s:e + 1]))
        # The endpoint farther from the local baseline is the likely QRS peak.
        peak = lo if abs(baseline - lo) >= abs(hi - baseline) else hi
        mid = (s + e) / 2.0
        half = max(1.0, (e - s + 1) / 2.0)
        for k in range(s, e + 1):
            weight = max(0.0, 1.0 - abs(k - mid) / half)
            signal[k] = signal[k] * (1.0 - weight) + peak * weight

    # Interpolate any remaining missing columns.
    signal = np.interp(np.arange(width), np.flatnonzero(np.isfinite(signal)), signal[np.isfinite(signal)])
    return signal.astype(np.float64), np.array([xmin, xmax], dtype=int)

def _estimate_time_scale(image: np.ndarray, width: int, component_width: int | None = None) -> tuple[float, str]:
    """Estimate seconds across the selected waveform; use ECG-paper fallback."""
    # If the trace spans most of the page, it is very likely the common 10-second
    # rhythm strip. Prefer this layout rule over noisy grid estimates.
    if component_width is not None:
        ratio = component_width / max(1, image.shape[1])
        if ratio >= 0.62:
            return 10.0, "10-second rhythm-strip layout estimate"

    # Red grid detection. If enough vertical grid peaks are found, 1 small box
    # is assumed to be 0.04 s at the standard 25 mm/s paper speed.
    b, g, r = [c.astype(np.float32) for c in cv2.split(image)]
    red = ((r > g * 1.10) & (r > b * 1.10) & (r > 130)).astype(np.uint8)
    projection = red.sum(axis=0).astype(np.float32)
    projection = gaussian_filter1d(projection, sigma=2)
    peaks, props = find_peaks(projection, distance=max(5, width // 300), prominence=max(2, projection.max() * 0.04))
    if len(peaks) >= 8:
        gaps = np.diff(peaks)
        gaps = gaps[(gaps > 2) & (gaps < np.percentile(gaps, 90) * 2)]
        if len(gaps) >= 6:
            px_per_small_box = float(np.median(gaps))
            seconds = width / max(px_per_small_box, 1e-6) * 0.04
            if 2.0 <= seconds <= 30.0:
                return seconds, "red-grid estimate at 25 mm/s"
    if component_width is not None:
        ratio = component_width / max(1, image.shape[1])
        if ratio >= 0.62:
            return 10.0, "10-second rhythm-strip layout fallback"
        if 0.14 <= ratio <= 0.42:
            return 2.5, "2.5-second multi-lead strip layout fallback"
    return DEFAULT_DURATION_SECONDS, "10-second ECG fallback"


def _resample_trace(path: np.ndarray, duration_seconds: float) -> np.ndarray:
    # Map pixels to a time series. Invert y so upward ECG is positive.
    x = np.linspace(0, duration_seconds, len(path), endpoint=False)
    y = -path.astype(np.float64)
    y -= np.median(y)
    # Remove slow camera/page tilt while preserving ECG morphology.
    baseline = gaussian_filter1d(y, sigma=max(5, int(0.6 * len(y) / duration_seconds)))
    y = y - baseline
    target_len = max(FS * 4, int(round(duration_seconds * FS)))
    new_x = np.linspace(0, duration_seconds, target_len, endpoint=False)
    signal = np.interp(new_x, x, y)
    # Normalize amplitude for a stable physical WFDB representation.
    signal = signal / (np.std(signal) + 1e-8)
    signal = np.clip(signal, -6, 6)
    return signal.astype(np.float64)


def _detect_r_peaks(signal: np.ndarray) -> np.ndarray:
    """Use WFDB XQRS first, then a robust absolute-peak fallback."""
    try:
        from wfdb import processing
        detector = processing.XQRS(sig=signal, fs=FS, verbose=False)
        detector.detect()
        peaks = np.asarray(detector.qrs_inds, dtype=int)
        if len(peaks) >= 3:
            return peaks
    except Exception:
        pass

    # Bandpass, absolute envelope, adaptive prominence and refractory period.
    x = signal - gaussian_filter1d(signal, sigma=max(1, int(0.5 * FS)))
    envelope = np.abs(x)
    envelope = gaussian_filter1d(envelope, sigma=max(1, int(0.025 * FS)))
    distance = int(0.25 * FS)
    prominence = max(np.std(envelope) * 0.8, np.percentile(envelope, 75) * 0.15)
    peaks, _ = find_peaks(envelope, distance=distance, prominence=prominence)
    return peaks.astype(int)


def _write_wfdb_zip(signal: np.ndarray, peaks: np.ndarray, metadata: dict) -> bytes:
    with tempfile.TemporaryDirectory(prefix="smart_ecg_image_wfdb_") as tmp:
        root = Path(tmp)
        record_name = "image_ecg"
        # Use mV units and 16-bit storage. The existing model z-scores each beat,
        # so exact amplitude calibration is not needed for this research adapter.
        wfdb.wrsamp(
            record_name,
            fs=FS,
            units=["mV"],
            sig_name=["MLII"],
            p_signal=signal.reshape(-1, 1),
            fmt=["16"],
            adc_gain=[1000],
            baseline=[0],
            write_dir=str(root),
            comments=[
                "SmartECG-ML image digitization adapter",
                f"source_type={metadata.get('source_type', 'image')}",
                f"time_scale={metadata.get('time_scale', '')}",
            ],
        )
        if len(peaks) < 2:
            raise ValueError("Could not detect enough ECG beats in the uploaded image.")
        wfdb.wrann(
            record_name,
            "atr",
            sample=np.asarray(peaks, dtype=int),
            symbol=["N"] * len(peaks),
            fs=FS,
            write_dir=str(root),
        )
        # wrann may already use the absolute record path; ensure all three exist.
        required = [root / "image_ecg.hea", root / "image_ecg.dat", root / "image_ecg.atr"]
        missing = [str(p.name) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"WFDB conversion failed; missing {missing}.")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in required:
                zf.write(p, p.name)
        return buf.getvalue()


def digitize_ecg_image(data: bytes, filename: str) -> tuple[bytes, dict]:
    """Convert an ECG image/PDF into a synthetic WFDB ZIP for the existing model.

    The generated .atr contains automatically detected beat positions using the
    symbol ``N`` only as a placeholder. It is NOT a clinician-provided diagnosis.
    The existing SmartECG-ML classifier supplies the final beat class.
    """
    image, meta = load_document(data, filename)
    if image.shape[0] < 200 or image.shape[1] < 400:
        raise ValueError("ECG image is too small. Upload a clear scan/photo with the waveform visible.")

    image = _deskew(image)
    component, bbox, extraction_method = _waveform_component(image)
    trace, x_range = _trace_component(component, bbox)
    duration, scale_method = _estimate_time_scale(image, len(trace), int(x_range[1] - x_range[0] + 1))
    signal = _resample_trace(trace, duration)
    peaks = _detect_r_peaks(signal)

    valid = []
    for p in peaks:
        if p < 90 or p + 162 >= len(signal):
            continue
        if valid and not (0.25 * FS <= p - valid[-1] <= 3.0 * FS):
            continue
        valid.append(int(p))
    peaks = np.asarray(valid, dtype=int)
    if len(peaks) < 3:
        raise ValueError(
            "The ECG waveform could not be digitized reliably enough to detect at least 3 beats. "
            "Try a clearer, straight-on photo or a higher-resolution scan with the entire waveform visible."
        )

    meta.update({
        "digitized": True,
        "sampling_frequency": FS,
        "duration_seconds": round(len(signal) / FS, 3),
        "waveform_bbox": [int(v) for v in bbox],
        "detected_beats": int(len(peaks)),
        "time_scale": scale_method,
        "extraction_method": extraction_method,
        "lead_note": "The adapter selects the widest ECG trace it can isolate; on common 12-lead printouts this is often the bottom rhythm strip. It does not reconstruct all 12 leads.",
        "generated_wfdb": ["image_ecg.hea", "image_ecg.dat", "image_ecg.atr"],
        "annotation_note": "The generated .atr marks automatically detected beat locations with N placeholders; these are not clinician labels.",
    })
    return _write_wfdb_zip(signal, peaks, meta), meta
