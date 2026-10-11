import os
import numpy as np
import pandas as pd
import wfdb


# ============================================================
# PATHS
# ============================================================

FILTERED_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\filtered_records"

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_dataset_patientwise.csv"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\filtered_ecg_beats.npz"


# ============================================================
# SETTINGS
# ============================================================

PRE_SECONDS = 0.25
POST_SECONDS = 0.45

SAMPLING_FREQUENCY = 360

PRE_SAMPLES = int(
    PRE_SECONDS * SAMPLING_FREQUENCY
)

POST_SAMPLES = int(
    POST_SECONDS * SAMPLING_FREQUENCY
)

WINDOW_SIZE = PRE_SAMPLES + POST_SAMPLES


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("FILTERED ECG BEAT EXTRACTION")
print("=" * 70)

print("Annotated beats:", len(df))
print("Window size:", WINDOW_SIZE)


# ============================================================
# STORAGE
# ============================================================

signals = []
labels = []
records = []
patients = []
samples = []


# ============================================================
# CACHE FILTERED RECORDS
# ============================================================

record_cache = {}


for record_name in df["record"].unique():

    record_name = str(record_name)

    print(
        "Loading filtered record:",
        record_name
    )

    path = os.path.join(
        FILTERED_PATH,
        f"{record_name}.npy"
    )

    signal = np.load(path)

    record_cache[record_name] = signal


# ============================================================
# EXTRACT WINDOWS
# ============================================================

for _, row in df.iterrows():

    record_name = str(row["record"])

    sample = int(row["sample"])

    signal = record_cache[record_name]

    start = sample - PRE_SAMPLES
    end = sample + POST_SAMPLES

    # Skip boundary beats
    if start < 0 or end > len(signal):
        continue

    beat = signal[start:end]

    # Verify exact size
    if len(beat) != WINDOW_SIZE:
        continue

    signals.append(beat)

    labels.append(row["class"])

    records.append(record_name)

    patients.append(row["patient_id"])

    samples.append(sample)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

signals = np.asarray(
    signals,
    dtype=np.float32
)

labels = np.asarray(labels)

records = np.asarray(records)

patients = np.asarray(patients)

samples = np.asarray(samples)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(
    "Number of beats:",
    len(signals)
)

print(
    "Signal shape:",
    signals.shape
)

print(
    "Labels shape:",
    labels.shape
)

print("\nClass distribution:")

unique, counts = np.unique(
    labels,
    return_counts=True
)

for label, count in zip(
    unique,
    counts
):

    print(
        f"{label}: {count}"
    )


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT_PATH,
    signals=signals,
    labels=labels,
    records=records,
    patients=patients,
    samples=samples
)

print("\nSaved to:")

print(OUTPUT_PATH)