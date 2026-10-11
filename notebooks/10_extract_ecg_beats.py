import os
import wfdb
import pandas as pd
import numpy as np


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\raw"

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_dataset_patientwise.csv"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\ecg_beats.npz"


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

# Use MLII
LEAD_INDEX = 0

# 0.25 seconds before annotation
PRE_SECONDS = 0.25

# 0.45 seconds after annotation
POST_SECONDS = 0.45


# --------------------------------------------------
# LOAD ANNOTATION DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("ECG BEAT WINDOW EXTRACTION")
print("=" * 70)

print("Total annotated beats:", len(df))


# --------------------------------------------------
# STORAGE
# --------------------------------------------------

signals = []
labels = []
records = []
patients = []
samples = []


# --------------------------------------------------
# CACHE ECG RECORDS
# --------------------------------------------------

record_cache = {}


# --------------------------------------------------
# PROCESS RECORDS
# --------------------------------------------------

for record_name in df["record"].unique():

    print("Loading record:", record_name)

    record_path = os.path.join(
        DATA_PATH,
        str(record_name)
    )

    record = wfdb.rdrecord(record_path)

    fs = record.fs

    signal = record.p_signal[:, LEAD_INDEX]

    record_cache[record_name] = (
        signal,
        fs
    )


# --------------------------------------------------
# EXTRACT WINDOWS
# --------------------------------------------------

for index, row in df.iterrows():

    record_name = row["record"]

    sample = int(row["sample"])

    signal, fs = record_cache[
        record_name
    ]

    pre_samples = int(
        PRE_SECONDS * fs
    )

    post_samples = int(
        POST_SECONDS * fs
    )

    start = sample - pre_samples

    end = sample + post_samples

    # Skip boundary beats
    if start < 0 or end >= len(signal):
        continue

    beat = signal[start:end]

    signals.append(beat)

    labels.append(row["class"])

    records.append(record_name)

    patients.append(row["patient_id"])

    samples.append(sample)


# --------------------------------------------------
# CONVERT TO ARRAY
# --------------------------------------------------

signals = np.array(
    signals,
    dtype=np.float32
)

labels = np.array(labels)

records = np.array(records)

patients = np.array(patients)

samples = np.array(samples)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)

print("EXTRACTION COMPLETE")

print("=" * 70)

print("Number of beats:", len(signals))

print("Window samples:", signals.shape[1])

print("Sampling frequency: 360 Hz")

print("Signal shape:", signals.shape)

print("Labels shape:", labels.shape)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

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
