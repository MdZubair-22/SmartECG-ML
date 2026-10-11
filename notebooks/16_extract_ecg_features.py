import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis


# ============================================================
# PATHS
# ============================================================

SIGNAL_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\normalized_ecg_beats.npz"

ANNOTATION_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_dataset_patientwise.csv"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\ecg_features.csv"


# ============================================================
# SETTINGS
# ============================================================

FS = 360

WINDOW_SIZE = 252

R_INDEX = int(0.25 * FS)


# ============================================================
# LOAD ECG SIGNALS
# ============================================================

data = np.load(SIGNAL_PATH)

signals = data["signals"]

labels = data["labels"]

records = data["records"].astype(str)

patients = data["patients"]

samples = data["samples"].astype(int)


print("=" * 70)
print("FAST ECG FEATURE EXTRACTION")
print("=" * 70)

print("Signals:", signals.shape)

print("Labels:", labels.shape)


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

annotations = pd.read_csv(
    ANNOTATION_PATH
)

annotations["record"] = (
    annotations["record"]
    .astype(str)
)

annotations["sample"] = (
    annotations["sample"]
    .astype(int)
)


# ============================================================
# CALCULATE RR INTERVALS
# ============================================================

annotations["prev_sample"] = (
    annotations
    .groupby("record")["sample"]
    .shift(1)
)

annotations["next_sample"] = (
    annotations
    .groupby("record")["sample"]
    .shift(-1)
)


annotations["prev_rr"] = (
    annotations["sample"]
    - annotations["prev_sample"]
) / FS


annotations["next_rr"] = (
    annotations["next_sample"]
    - annotations["sample"]
) / FS


annotations["prev_hr"] = (
    60 / annotations["prev_rr"]
)

annotations["next_hr"] = (
    60 / annotations["next_rr"]
)


annotations["mean_rr"] = (
    annotations["prev_rr"]
    +
    annotations["next_rr"]
) / 2


# ============================================================
# CREATE SIGNAL METADATA
# ============================================================

meta = pd.DataFrame({

    "record": records,

    "patient_id": patients,

    "sample": samples,

    "class": labels
})


# ============================================================
# MERGE RR FEATURES
# ============================================================

rr_columns = [
    "record",
    "sample",
    "prev_rr",
    "next_rr",
    "mean_rr",
    "prev_hr",
    "next_hr"
]


meta = meta.merge(
    annotations[rr_columns],
    on=["record", "sample"],
    how="left"
)


print("\nRR information merged successfully.")


# ============================================================
# BASIC STATISTICAL FEATURES
# ============================================================

print("\nCalculating statistical features...")


mean_value = np.mean(
    signals,
    axis=1
)

std_value = np.std(
    signals,
    axis=1
)

min_value = np.min(
    signals,
    axis=1
)

max_value = np.max(
    signals,
    axis=1
)

peak_to_peak = (
    max_value - min_value
)

rms = np.sqrt(
    np.mean(
        signals ** 2,
        axis=1
    )
)

energy = np.sum(
    signals ** 2,
    axis=1
)

absolute_area = np.sum(
    np.abs(signals),
    axis=1
)

max_abs = np.max(
    np.abs(signals),
    axis=1
)


# ============================================================
# SHAPE FEATURES
# ============================================================

print("Calculating morphology features...")


skewness = skew(
    signals,
    axis=1
)

kurt = kurtosis(
    signals,
    axis=1
)


# ============================================================
# R-PEAK RELATED FEATURES
# ============================================================

pre_signal = signals[
    :,
    :R_INDEX
]

post_signal = signals[
    :,
    R_INDEX:
]


pre_energy = np.sum(
    pre_signal ** 2,
    axis=1
)

post_energy = np.sum(
    post_signal ** 2,
    axis=1
)


pre_mean = np.mean(
    pre_signal,
    axis=1
)

post_mean = np.mean(
    post_signal,
    axis=1
)


# ============================================================
# ZERO CROSSINGS
# ============================================================

print("Calculating zero crossings...")


zero_crossings = np.sum(
    np.diff(
        np.signbit(signals),
        axis=1
    ),
    axis=1
)


# ============================================================
# FREQUENCY FEATURES
# ============================================================

print("Calculating frequency features...")


# Remove mean before FFT
fft_signal = (
    signals
    - mean_value[:, None]
)


fft_values = np.fft.rfft(
    fft_signal,
    axis=1
)


power = (
    np.abs(fft_values) ** 2
)


frequencies = np.fft.rfftfreq(
    WINDOW_SIZE,
    d=1 / FS
)


total_power = np.sum(
    power,
    axis=1
)


# Avoid division by zero
safe_power = np.where(
    total_power[:, None] > 0,
    power,
    1e-12
)


# Dominant frequency
dominant_indices = np.argmax(
    power,
    axis=1
)


dominant_frequency = (
    frequencies[
        dominant_indices
    ]
)


# Spectral entropy
probability = (
    safe_power
    /
    np.sum(
        safe_power,
        axis=1,
        keepdims=True
    )
)


spectral_entropy = -np.sum(
    probability
    *
    np.log2(probability),
    axis=1
)


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

feature_df = pd.DataFrame({

    "record": records,

    "patient_id": patients,

    "sample": samples,

    "mean": mean_value,

    "std": std_value,

    "min": min_value,

    "max": max_value,

    "peak_to_peak": peak_to_peak,

    "rms": rms,

    "energy": energy,

    "absolute_area": absolute_area,

    "skewness": skewness,

    "kurtosis": kurt,

    "max_abs": max_abs,

    "zero_crossings": zero_crossings,

    "pre_energy": pre_energy,

    "post_energy": post_energy,

    "pre_mean": pre_mean,

    "post_mean": post_mean,

    "dominant_frequency": dominant_frequency,

    "spectral_entropy": spectral_entropy,

    "prev_rr": meta["prev_rr"].values,

    "next_rr": meta["next_rr"].values,

    "mean_rr": meta["mean_rr"].values,

    "prev_hr": meta["prev_hr"].values,

    "next_hr": meta["next_hr"].values,

    "class": labels
})


# ============================================================
# REMOVE INVALID RR VALUES
# ============================================================

print("\nMissing values before cleaning:")

print(
    feature_df.isna().sum()
)


feature_df = feature_df.dropna()


# ============================================================
# REMOVE INFINITE VALUES
# ============================================================

feature_df = feature_df.replace(
    [np.inf, -np.inf],
    np.nan
)

feature_df = feature_df.dropna()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)

print("FEATURE EXTRACTION COMPLETE")

print("=" * 70)

print(
    "Final samples:",
    len(feature_df)
)

print(
    "Total columns:",
    len(feature_df.columns)
)

print(
    "ML features:",
    len(feature_df.columns) - 4
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    feature_df["class"]
    .value_counts()
)


# ============================================================
# FEATURE LIST
# ============================================================

print("\nFeature columns:")

for column in feature_df.columns:

    print(
        column
    )


# ============================================================
# SAVE
# ============================================================

feature_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)

print("FEATURE DATASET SAVED")

print("=" * 70)

print(
    OUTPUT_PATH
)