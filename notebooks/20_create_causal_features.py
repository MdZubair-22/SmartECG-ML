import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\ml_dataset.csv"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\causal_ecg_features.csv"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "record": str,
        "patient_id": str
    }
)


print("=" * 70)
print("CAUSAL ECG FEATURE DATASET")
print("=" * 70)

print(
    "Original rows:",
    len(df)
)


# ============================================================
# ROBUST RR PROCESSING
# ============================================================

# Physiologically reasonable range used for the
# baseline causal rhythm feature.

MIN_RR = 0.25
MAX_RR = 3.0


# Keep previous RR only when it is within the
# accepted range.

df["prev_rr_clean"] = df["prev_rr"].where(
    (
        df["prev_rr"] >= MIN_RR
    )
    &
    (
        df["prev_rr"] <= MAX_RR
    ),
    np.nan
)


# Calculate previous heart rate from cleaned RR.

df["prev_hr_clean"] = (
    60 / df["prev_rr_clean"]
)


# ============================================================
# NEW MORPHOLOGY FEATURE
# ============================================================

# Energy distribution before/after reference point.

EPSILON = 1e-8


df["energy_ratio"] = (
    df["pre_energy"]
    /
    (
        df["post_energy"]
        + EPSILON
    )
)


# ============================================================
# SELECT CAUSAL FEATURES
# ============================================================


causal_features = [

    "min",
    "max",
    "peak_to_peak",
    "absolute_area",
    "skewness",
    "kurtosis",
    "zero_crossings",
    "max_abs",
    "energy_ratio",
    "dominant_frequency",
    "spectral_entropy",
    "prev_rr",
    "prev_hr"
]


metadata = [
    "record",
    "patient_id",
    "sample",
    "class",
    "fold"
]


final_columns = (
    metadata
    +
    causal_features
)


causal_df = df[
    final_columns
].copy()


# Rename cleaned features to final names.

causal_df = causal_df.rename(
    columns={
        "prev_rr_clean": "prev_rr",
        "prev_hr_clean": "prev_hr"
    }
)


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

causal_df = causal_df.replace(
    [np.inf, -np.inf],
    np.nan
)


print("\nMissing values:")

print(
    causal_df.isna().sum()
)


# For the first beat of a record, previous RR
# is unavailable.

before = len(causal_df)

causal_df = causal_df.dropna(
    subset=[
        "prev_rr",
        "prev_hr"
    ]
)

removed = before - len(causal_df)


print(
    "\nRows removed because previous RR was unavailable/invalid:",
    removed
)


# ============================================================
# CHECK
# ============================================================

print("\n" + "=" * 70)

print("CAUSAL FEATURE VALIDATION")

print("=" * 70)


print(
    "Rows:",
    len(causal_df)
)

print(
    "Columns:",
    len(causal_df.columns)
)

print(
    "Number of ML features:",
    len(causal_features)
)


print("\nML features:")

for feature in causal_features:

    print(
        " -",
        feature
    )


# ============================================================
# INVALID VALUE CHECK
# ============================================================

numeric_features = causal_df[
    causal_features
]


print("\nNaN values:")

print(
    numeric_features.isna()
    .sum()
    .sum()
)


print(
    "Infinite values:",
    np.isinf(
        numeric_features
    ).sum().sum()
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)

print("CLASS DISTRIBUTION")

print("=" * 70)

print(
    causal_df["class"]
    .value_counts()
)


# ============================================================
# FOLD DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)

print("FOLD DISTRIBUTION")

print("=" * 70)

print(
    causal_df["fold"]
    .value_counts()
    .sort_index()
)


# ============================================================
# PATIENT LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)

print("PATIENT LEAKAGE CHECK")

print("=" * 70)


patient_fold_counts = (
    causal_df
    .groupby("patient_id")["fold"]
    .nunique()
)


leakage = (
    patient_fold_counts[
        patient_fold_counts > 1
    ]
)


if len(leakage) == 0:

    print(
        "SUCCESS: No patient appears in multiple folds."
    )

else:

    print(
        "ERROR: Patient leakage detected."
    )

    print(leakage)

    raise ValueError(
        "Patient leakage detected."
    )


# ============================================================
# SAVE
# ============================================================

causal_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)

print("CAUSAL DATASET READY")

print("=" * 70)

print(
    "Saved to:"
)

print(
    OUTPUT_PATH
)