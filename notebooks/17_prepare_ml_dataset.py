import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

FEATURE_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\ecg_features.csv"

FOLD_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\patient_3folds.csv"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\ml_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PREPARING ML DATASET")
print("=" * 70)


features = pd.read_csv(
    FEATURE_PATH,
    dtype={"patient_id": str}
)

folds = pd.read_csv(
    FOLD_PATH,
    dtype={"patient_id": str}
)


print("\nFeature rows:", len(features))

print("Feature columns:", len(features.columns))

print("Fold rows:", len(folds))


# ============================================================
# CREATE UNIQUE PATIENT → FOLD TABLE
# ============================================================

patient_folds = (
    folds[
        ["patient_id", "fold"]
    ]
    .drop_duplicates()
    .sort_values("patient_id")
)


print("\nUnique patient-fold mappings:")

print(
    len(patient_folds)
)


# ============================================================
# VERIFY EVERY PATIENT HAS ONLY ONE FOLD
# ============================================================

patient_fold_counts = (
    patient_folds
    .groupby("patient_id")["fold"]
    .nunique()
)


invalid_patients = (
    patient_fold_counts[
        patient_fold_counts > 1
    ]
)


print("\nPatients assigned to multiple folds:")

print(
    len(invalid_patients)
)


if len(invalid_patients) > 0:

    print("\nERROR: Some patients have multiple folds.")

    print(invalid_patients)

    raise ValueError(
        "Patient fold assignment is invalid."
    )

else:

    print(
        "SUCCESS: Every patient belongs to exactly one fold."
    )


# ============================================================
# VERIFY NUMBER OF PATIENTS
# ============================================================

print(
    "\nUnique patients in fold mapping:",
    patient_folds["patient_id"].nunique()
)


print(
    "Unique patients in feature dataset:",
    features["patient_id"].nunique()
)


# ============================================================
# MERGE ONE ROW PER PATIENT
# ============================================================

ml_df = features.merge(
    patient_folds,
    on="patient_id",
    how="left",
    validate="many_to_one"
)


# ============================================================
# CHECK MERGE
# ============================================================

print("\nAfter merge:")

print(
    "Rows:",
    len(ml_df)
)

print(
    "Missing folds:",
    ml_df["fold"].isna().sum()
)


if ml_df["fold"].isna().sum() > 0:

    print(
        "\nWARNING: Some feature rows do not have a fold."
    )

    missing_patients = (
        ml_df.loc[
            ml_df["fold"].isna(),
            "patient_id"
        ]
        .unique()
    )

    print(
        "Missing patient IDs:",
        missing_patients
    )

    raise ValueError(
        "Fold assignment missing for some patients."
    )


# ============================================================
# CONVERT FOLD TO INTEGER
# ============================================================

ml_df["fold"] = (
    ml_df["fold"]
    .astype(int)
)


# ============================================================
# PATIENT LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)


patient_fold_check = (
    ml_df
    .groupby("patient_id")["fold"]
    .nunique()
)


leaking_patients = (
    patient_fold_check[
        patient_fold_check > 1
    ]
)


if len(leaking_patients) == 0:

    print(
        "SUCCESS: No patient appears in multiple folds."
    )

else:

    print(
        "ERROR: Patient leakage detected!"
    )

    print(
        leaking_patients
    )

    raise ValueError(
        "Patient leakage detected."
    )


# ============================================================
# PATIENTS PER FOLD
# ============================================================

print("\n" + "=" * 70)
print("PATIENTS PER FOLD")
print("=" * 70)


patients_per_fold = (
    ml_df
    .groupby("fold")["patient_id"]
    .nunique()
)


print(
    patients_per_fold
)


# ============================================================
# SAMPLES PER FOLD
# ============================================================

print("\n" + "=" * 70)
print("SAMPLES PER FOLD")
print("=" * 70)


samples_per_fold = (
    ml_df["fold"]
    .value_counts()
    .sort_index()
)


print(
    samples_per_fold
)


# ============================================================
# CLASS DISTRIBUTION BY FOLD
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION BY FOLD")
print("=" * 70)


class_distribution = pd.crosstab(
    ml_df["fold"],
    ml_df["class"]
)


class_distribution = (
    class_distribution
    .reindex(
        columns=["N", "S", "V", "F", "Q"],
        fill_value=0
    )
)


print(
    class_distribution
)


# ============================================================
# CHECK REQUIRED ML FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)


metadata_columns = [
    "record",
    "patient_id",
    "sample",
    "fold",
    "class"
]


feature_columns = [
    column
    for column in ml_df.columns
    if column not in metadata_columns
]


print(
    "Number of ML features:",
    len(feature_columns)
)


print("\nML features:")

for feature in feature_columns:

    print(
        " -",
        feature
    )


# ============================================================
# CHECK NaN / INFINITE VALUES
# ============================================================

print("\nChecking invalid values...")


nan_count = (
    ml_df[feature_columns]
    .isna()
    .sum()
    .sum()
)


infinite_count = (
    np.isinf(
        ml_df[feature_columns]
        .select_dtypes(
            include=np.number
        )
    )
    .sum()
    .sum()
)


print(
    "NaN values:",
    nan_count
)

print(
    "Infinite values:",
    infinite_count
)


# ============================================================
# SAVE
# ============================================================

ml_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("ML DATASET READY")
print("=" * 70)


print(
    "Rows:",
    len(ml_df)
)

print(
    "Columns:",
    len(ml_df.columns)
)

print(
    "ML features:",
    len(feature_columns)
)

print("\nSaved to:")

print(
    OUTPUT_PATH
)