import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\raw"

FILTERED_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\filtered_records"

RECORD = "100"

# Show first 10 seconds
SECONDS = 10


# ============================================================
# LOAD RAW ECG
# ============================================================

record_path = os.path.join(
    DATA_PATH,
    RECORD
)

record = wfdb.rdrecord(record_path)

fs = record.fs

raw_signal = record.p_signal[:, 0]


# ============================================================
# LOAD FILTERED ECG
# ============================================================

filtered_file = os.path.join(
    FILTERED_PATH,
    f"{RECORD}.npy"
)

filtered_signal = np.load(
    filtered_file
)


# ============================================================
# SELECT SAME TIME RANGE
# ============================================================

samples = int(
    SECONDS * fs
)

time = np.arange(samples) / fs


raw_segment = raw_signal[:samples]

filtered_segment = filtered_signal[:samples]


# ============================================================
# PLOT RAW
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    time,
    raw_segment
)

plt.title(
    f"Record {RECORD} - Raw ECG (MLII)"
)

plt.xlabel("Time (seconds)")

plt.ylabel("Amplitude (mV)")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT FILTERED
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    time,
    filtered_segment
)

plt.title(
    f"Record {RECORD} - Filtered ECG (0.5–40 Hz)"
)

plt.xlabel("Time (seconds)")

plt.ylabel("Amplitude")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# RAW VS FILTERED
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    time,
    raw_segment,
    label="Raw"
)

plt.plot(
    time,
    filtered_segment,
    label="Filtered"
)

plt.title(
    f"Record {RECORD} - Raw vs Filtered ECG"
)

plt.xlabel("Time (seconds)")

plt.ylabel("Amplitude")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# STATISTICS
# ============================================================

print("=" * 70)

print("RAW VS FILTERED ECG")

print("=" * 70)

print("\nSampling frequency:", fs)

print("\nRAW")

print("Mean:", np.mean(raw_segment))

print("Std:", np.std(raw_segment))

print("Min:", np.min(raw_segment))

print("Max:", np.max(raw_segment))


print("\nFILTERED")

print(
    "Mean:",
    np.mean(filtered_segment)
)

print(
    "Std:",
    np.std(filtered_segment)
)

print(
    "Min:",
    np.min(filtered_segment)
)

print(
    "Max:",
    np.max(filtered_segment)
)