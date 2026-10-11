
from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis, skew


FS = 360
PRE_SAMPLES = int(0.25 * FS)   # 90
POST_SAMPLES = int(0.45 * FS)  # 162
WINDOW_SIZE = PRE_SAMPLES + POST_SAMPLES  # 252
ANNOTATION_INDEX = 90

LOWCUT = 0.5
HIGHCUT = 40.0
FILTER_ORDER = 4
LEAD_INDEX = 0

BEAT_SYMBOLS = {
    "N", "L", "R", "B",
    "A", "a", "J", "S",
    "V", "r", "F",
    "e", "j", "n", "E", "/",
    "f", "Q", "?"
}

CLASS_NAMES = {
    "N": "Normal",
    "S": "Supraventricular",
    "V": "Ventricular",
    "F": "Fusion",
    "Q": "Unknown / Other",
}

ABNORMAL_CLASSES = {"S", "V", "F"}


def summarize_predictions(prediction_distribution: dict[str, int]) -> dict:
    """Summarize beat predictions without treating unknown beats as normal."""
    abnormal_beats = sum(
        count for label, count in prediction_distribution.items()
        if label in ABNORMAL_CLASSES
    )
    inconclusive_beats = int(prediction_distribution.get("Q", 0))
    normal_beats = int(prediction_distribution.get("N", 0))

    if abnormal_beats:
        status = "abnormal"
        label = "Abnormal pattern detected"
        explanation = (
            f"{abnormal_beats} analyzed beat(s) were classified as "
            "supraventricular, ventricular, or fusion."
        )
    elif inconclusive_beats:
        status = "inconclusive"
        label = "Inconclusive"
        explanation = (
            f"{inconclusive_beats} analyzed beat(s) were classified as "
            "unknown/other, so this result cannot be summarized as normal."
        )
    elif normal_beats:
        status = "normal"
        label = "Normal pattern detected"
        explanation = (
            "All analyzed beats were classified as normal by the model."
        )
    else:
        status = "inconclusive"
        label = "Inconclusive"
        explanation = "No beat predictions were available to summarize."

    return {
        "status": status,
        "label": label,
        "explanation": explanation,
        "normal_beats": normal_beats,
        "abnormal_beats": abnormal_beats,
        "inconclusive_beats": inconclusive_beats,
    }


def safe_extract_zip(data: bytes, destination: Path) -> None:
    """Extract a user ZIP without allowing path traversal."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        files = [i for i in zf.infolist() if not i.is_dir()]
        if not files:
            raise ValueError("The ZIP file is empty.")

        for info in files:
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe path detected in uploaded ZIP.")

            target = (destination / name).resolve()
            if not str(target).startswith(str(destination.resolve()) + os.sep):
                raise ValueError("Unsafe ZIP path detected.")

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def find_records(root: Path) -> list[Path]:
    """Find WFDB records by .hea files and require matching data/annotation files."""
    records = []
    for hea in sorted(root.rglob("*.hea")):
        stem = hea.with_suffix("")
        dat = stem.with_suffix(".dat")
        atr = stem.with_suffix(".atr")
        if dat.exists() and atr.exists():
            records.append(stem)

    if not records:
        raise ValueError(
            "No complete WFDB record found. ZIP must contain matching "
            ".hea, .dat and .atr files."
        )
    return records


def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    nyquist = fs / 2.0
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist
    if not 0 < low < high < 1:
        raise ValueError(f"Unsupported sampling frequency: {fs}")
    sos = butter(FILTER_ORDER, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def morphology_features(signal: np.ndarray) -> dict[str, float]:
    """Exact morphology logic used by notebooks/24_extract_morphology_features.py."""
    x = signal[:ANNOTATION_INDEX + 1]

    minimum = float(np.min(x))
    maximum = float(np.max(x))
    amplitude_range = maximum - minimum
    mean_value = float(np.mean(x))
    std_value = float(np.std(x))
    rms = float(np.sqrt(np.mean(x ** 2)))

    annotation_amplitude = float(signal[ANNOTATION_INDEX])

    pre_10 = signal[max(0, ANNOTATION_INDEX - 10):ANNOTATION_INDEX]
    pre_20 = signal[max(0, ANNOTATION_INDEX - 20):ANNOTATION_INDEX]
    pre_30 = signal[max(0, ANNOTATION_INDEX - 30):ANNOTATION_INDEX]

    pre10_mean = float(np.mean(pre_10))
    pre20_mean = float(np.mean(pre_20))
    pre30_mean = float(np.mean(pre_30))

    annotation_vs_pre10 = annotation_amplitude - pre10_mean
    annotation_vs_pre20 = annotation_amplitude - pre20_mean
    annotation_vs_pre30 = annotation_amplitude - pre30_mean

    slope_10 = (
        (annotation_amplitude - pre_10[0]) / len(pre_10)
        if len(pre_10) >= 2 else 0.0
    )
    slope_20 = (
        (annotation_amplitude - pre_20[0]) / len(pre_20)
        if len(pre_20) >= 2 else 0.0
    )
    slope_30 = (
        (annotation_amplitude - pre_30[0]) / len(pre_30)
        if len(pre_30) >= 2 else 0.0
    )

    derivative = np.diff(x)
    if len(derivative) > 0:
        max_positive_slope = float(np.max(derivative))
        max_negative_slope = float(np.min(derivative))
        mean_absolute_slope = float(np.mean(np.abs(derivative)))
    else:
        max_positive_slope = max_negative_slope = mean_absolute_slope = 0.0

    total_energy = float(np.sum(x ** 2))
    pre10_energy = float(np.sum(pre_10 ** 2))
    pre20_energy = float(np.sum(pre_20 ** 2))
    pre30_energy = float(np.sum(pre_30 ** 2))

    if total_energy > 0:
        pre10_energy_ratio = pre10_energy / total_energy
        pre20_energy_ratio = pre20_energy / total_energy
        pre30_energy_ratio = pre30_energy / total_energy
    else:
        pre10_energy_ratio = pre20_energy_ratio = pre30_energy_ratio = 0.0

    midpoint = len(x) // 2
    first_half = x[:midpoint]
    second_half = x[midpoint:]
    first_energy = float(np.sum(first_half ** 2))
    second_energy = float(np.sum(second_half ** 2))
    denominator = first_energy + second_energy
    energy_asymmetry = (
        (first_energy - second_energy) / denominator
        if denominator > 0 else 0.0
    )

    return {
        "morph_annotation_amplitude": annotation_amplitude,
        "morph_min": minimum,
        "morph_max": maximum,
        "morph_range": amplitude_range,
        "morph_mean": mean_value,
        "morph_std": std_value,
        "morph_rms": rms,
        "morph_pre10_mean": pre10_mean,
        "morph_pre20_mean": pre20_mean,
        "morph_pre30_mean": pre30_mean,
        "morph_annotation_pre10_diff": annotation_vs_pre10,
        "morph_annotation_pre20_diff": annotation_vs_pre20,
        "morph_annotation_pre30_diff": annotation_vs_pre30,
        "morph_slope10": slope_10,
        "morph_slope20": slope_20,
        "morph_slope30": slope_30,
        "morph_max_positive_slope": max_positive_slope,
        "morph_max_negative_slope": max_negative_slope,
        "morph_mean_abs_slope": mean_absolute_slope,
        "morph_pre10_energy_ratio": pre10_energy_ratio,
        "morph_pre20_energy_ratio": pre20_energy_ratio,
        "morph_pre30_energy_ratio": pre30_energy_ratio,
        "morph_energy_asymmetry": energy_asymmetry,
    }


def causal_features(normalized_signal: np.ndarray) -> dict[str, float]:
    """Exact relevant logic from notebooks/16 + notebooks/20."""
    mean_value = float(np.mean(normalized_signal))
    std_value = float(np.std(normalized_signal))
    min_value = float(np.min(normalized_signal))
    max_value = float(np.max(normalized_signal))
    peak_to_peak = max_value - min_value
    rms = float(np.sqrt(np.mean(normalized_signal ** 2)))
    energy = float(np.sum(normalized_signal ** 2))
    absolute_area = float(np.sum(np.abs(normalized_signal)))
    max_abs = float(np.max(np.abs(normalized_signal)))
    skewness = float(skew(normalized_signal))
    kurt = float(kurtosis(normalized_signal))

    pre_signal = normalized_signal[:ANNOTATION_INDEX]
    post_signal = normalized_signal[ANNOTATION_INDEX:]
    pre_energy = float(np.sum(pre_signal ** 2))
    post_energy = float(np.sum(post_signal ** 2))
    pre_mean = float(np.mean(pre_signal))
    post_mean = float(np.mean(post_signal))

    zero_crossings = int(np.sum(
        np.diff(np.signbit(normalized_signal))
    ))

    fft_signal = normalized_signal - mean_value
    fft_values = np.fft.rfft(fft_signal)
    power = np.abs(fft_values) ** 2
    frequencies = np.fft.rfftfreq(WINDOW_SIZE, d=1 / FS)
    total_power = np.sum(power)
    safe_power = np.where(
        total_power > 0, power, 1e-12
    )
    dominant_indices = np.argmax(power)
    dominant_frequency = float(frequencies[dominant_indices])

    probability = safe_power / np.sum(safe_power)
    spectral_entropy = float(
        -np.sum(probability * np.log2(probability))
    )

    energy_ratio = pre_energy / (post_energy + 1e-8)

    return {
        "energy_ratio": float(energy_ratio),
        "dominant_frequency": dominant_frequency,
        "spectral_entropy": spectral_entropy,
    }


def build_features(
    filtered_beat: np.ndarray,
    normalized_beat: np.ndarray,
    prev_rr: float,
) -> dict[str, float]:
    """Combine the exact causal + morphology components used by the final model."""
    if not (0.25 <= prev_rr <= 3.0):
        raise ValueError("Previous RR outside training range.")

    prev_hr = 60.0 / prev_rr
    values = {}
    values.update(causal_features(normalized_beat))
    values["prev_rr"] = float(prev_rr)
    values["prev_hr"] = float(prev_hr)
    values.update(morphology_features(filtered_beat))
    return values


def analyze_wfdb_zip(
    data: bytes,
    model,
    feature_list: list[str],
    max_results: int = 5000,
) -> dict:
    """
    Analyze a WFDB ZIP using the same:
    MLII lead -> 0.5-40 Hz filter -> 252 sample beat ->
    per-beat z-score normalization -> exact feature calculations -> model.
    """
    with tempfile.TemporaryDirectory(prefix="smart_ecg_") as tmp:
        root = Path(tmp)
        safe_extract_zip(data, root)
        records = find_records(root)

        results = []
        waveform = None
        waveform_sample = None
        skipped = 0

        for record_path in records:
            record = wfdb.rdrecord(str(record_path))
            annotation = wfdb.rdann(str(record_path), "atr")

            if record.p_signal is None:
                raise ValueError(f"Record {record_path.name} has no physical signal.")

            fs = float(record.fs)
            signal = np.asarray(record.p_signal[:, LEAD_INDEX], dtype=np.float64)

            filtered = bandpass_filter(signal, fs)

            # Preserve the training pipeline's 360 Hz assumption.
            if int(round(fs)) != FS:
                raise ValueError(
                    f"Record {record_path.name} uses {fs} Hz. "
                    f"The trained pipeline expects {FS} Hz."
                )

            samples = [
                int(s) for s, sym in zip(annotation.sample, annotation.symbol)
                if sym in BEAT_SYMBOLS
            ]
            symbols = [
                str(sym) for sym in annotation.symbol
                if sym in BEAT_SYMBOLS
            ]

            prev_sample = None
            for sample, symbol in zip(samples, symbols):
                if len(results) >= max_results:
                    break

                # Match notebooks/10 and /14 boundary rules.
                start = sample - PRE_SAMPLES
                end = sample + POST_SAMPLES
                if start < 0 or end > len(signal):
                    skipped += 1
                    prev_sample = sample
                    continue

                if prev_sample is None:
                    skipped += 1
                    prev_sample = sample
                    continue

                prev_rr = (sample - prev_sample) / FS
                prev_sample = sample

                if not (0.25 <= prev_rr <= 3.0):
                    skipped += 1
                    continue

                filtered_beat = filtered[start:end]
                if len(filtered_beat) != WINDOW_SIZE:
                    skipped += 1
                    continue

                mean = np.mean(filtered_beat)
                std = np.std(filtered_beat)
                if std < 1e-8:
                    skipped += 1
                    continue

                normalized_beat = (
                    (filtered_beat - mean) / std
                ).astype(np.float64)

                features = build_features(
                    filtered_beat,
                    normalized_beat,
                    prev_rr,
                )

                X = pd.DataFrame(
                    [[features[f] for f in feature_list]],
                    columns=feature_list,
                )

                pred = str(model.predict(X)[0])
                probs = model.predict_proba(X)[0]
                prob_map = {
                    cls: float(p)
                    for cls, p in zip(model.classes_, probs)
                }

                result = {
                    "record": record_path.stem,
                    "sample": sample,
                    "time_seconds": round(sample / FS, 4),
                    "annotation": symbol,
                    "prediction": pred,
                    "description": CLASS_NAMES.get(pred, "Unknown"),
                    "probability": prob_map[pred],
                    "probabilities": prob_map,
                }
                results.append(result)

                if waveform is None:
                    waveform = filtered_beat.tolist()
                    waveform_sample = sample

            if len(results) >= max_results:
                break

        if not results:
            raise ValueError(
                "No valid beats could be analyzed. Check that the ZIP contains "
                "valid WFDB .hea/.dat/.atr files at 360 Hz."
            )

        prediction_counts = (
            pd.Series([r["prediction"] for r in results])
            .value_counts()
            .to_dict()
        )

        return {
            "success": True,
            "records_found": len(records),
            "beats_analyzed": len(results),
            "beats_skipped": skipped,
            "max_results": max_results,
            "prediction_distribution": {
                str(k): int(v) for k, v in prediction_counts.items()
            },
            "overall_assessment": summarize_predictions({
                str(k): int(v) for k, v in prediction_counts.items()
            }),
            "first_result": results[0],
            "results": results,
            "waveform": {
                "sample": waveform_sample,
                "sampling_frequency": FS,
                "values": waveform,
            },
        }
