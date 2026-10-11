import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\filtered_ecg_beats.npz"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\normalized_ecg_beats.npz"


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(INPUT_PATH)

signals = data["signals"]
labels = data["labels"]
records = data["records"]
patients = data["patients"]
samples = data["samples"]


print("=" * 70)
print("ECG BEAT NORMALIZATION")
print("=" * 70)

print("Original signal shape:", signals.shape)


# ============================================================
# Z-SCORE NORMALIZATION
# ============================================================

means = np.mean(
    signals,
    axis=1,
    keepdims=True
)

stds = np.std(
    signals,
    axis=1,
    keepdims=True
)


# Prevent division by zero
stds[stds < 1e-8] = 1.0


normalized_signals = (
    signals - means
) / stds


normalized_signals = normalized_signals.astype(
    np.float32
)


# ============================================================
# CHECK
# ============================================================

print("\nNormalization check:")

print(
    "Example original mean:",
    np.mean(signals[0])
)

print(
    "Example original std:",
    np.std(signals[0])
)

print(
    "Example normalized mean:",
    np.mean(normalized_signals[0])
)

print(
    "Example normalized std:",
    np.std(normalized_signals[0])
)


# ============================================================
# GLOBAL CHECK
# ============================================================

print("\nGlobal normalized statistics:")

print(
    "Mean:",
    np.mean(normalized_signals)
)

print(
    "Std:",
    np.std(normalized_signals)
)

print(
    "Min:",
    np.min(normalized_signals)
)

print(
    "Max:",
    np.max(normalized_signals)
)


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT_PATH,
    signals=normalized_signals,
    labels=labels,
    records=records,
    patients=patients,
    samples=samples
)


print("\n" + "=" * 70)

print("NORMALIZATION COMPLETE")

print("=" * 70)

print("Normalized signal shape:")
print(normalized_signals.shape)

print("\nSaved to:")
print(OUTPUT_PATH)