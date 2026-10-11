import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

CAUSAL_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\causal_ecg_features.csv"

MORPH_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\causal_morphology_features.csv"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\enhanced_causal_ecg_features.csv"


print("=" * 70)
print("MERGING CAUSAL + MORPHOLOGY FEATURES")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

causal = pd.read_csv(
    CAUSAL_PATH,
    dtype={
        "record": str,
        "patient_id": str
    }
)

morph = pd.read_csv(
    MORPH_PATH,
    dtype={
        "record": str,
        "patient_id": str
    }
)


print(
    "\nCausal dataset:",
    causal.shape
)

print(
    "Morphology dataset:",
    morph.shape
)


# ============================================================
# MERGE KEYS
# ============================================================

KEYS = [
    "record",
    "patient_id",
    "sample"
]


# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\nChecking duplicate keys...")


causal_duplicates = causal.duplicated(
    KEYS
).sum()

morph_duplicates = morph.duplicated(
    KEYS
).sum()


print(
    "Causal duplicates:",
    causal_duplicates
)

print(
    "Morphology duplicates:",
    morph_duplicates
)


if causal_duplicates > 0:

    raise ValueError(
        "Duplicate keys found in causal dataset."
    )


if morph_duplicates > 0:

    raise ValueError(
        "Duplicate keys found in morphology dataset."
    )


# ============================================================
# MORPHOLOGY COLUMNS
# ============================================================

morph_features = [

    column

    for column in morph.columns

    if column not in KEYS
]


# ============================================================
# MERGE
# ============================================================

print(
    "\nMerging using:"
)

print(
    "record + patient_id + sample"
)


enhanced = causal.merge(

    morph[
        KEYS + morph_features
    ],

    on=KEYS,

    how="inner",

    validate="one_to_one"

)


print(
    "\nMerged dataset:",
    enhanced.shape
)


# ============================================================
# CHECK ROW RETENTION
# ============================================================

print(
    "\nRows retained:",
    len(enhanced)
)

print(
    "Rows in causal dataset:",
    len(causal)
)


retention = (
    len(enhanced)
    /
    len(causal)
    *
    100
)


print(
    "Retention:",
    round(
        retention,
        2
    ),
    "%"
)


# ============================================================
# CHECK TARGET
# ============================================================

print(
    "\nClass distribution:"
)

print(
    enhanced["class"]
    .value_counts()
)


# ============================================================
# CHECK FOLD
# ============================================================

print(
    "\nFold distribution:"
)

print(
    enhanced["fold"]
    .value_counts()
)


# ============================================================
# FEATURE LIST
# ============================================================

metadata = [
    "record",
    "patient_id",
    "sample",
    "class",
    "fold"
]


ml_features = [

    column

    for column in enhanced.columns

    if column not in metadata
]


print(
    "\nTotal ML features:",
    len(ml_features)
)


print(
    "\nML features:"
)

for feature in ml_features:

    print(
        " -",
        feature
    )


# ============================================================
# VALIDATION
# ============================================================

print(
    "\nChecking missing values..."
)

nan_count = (
    enhanced[
        ml_features
    ]
    .isna()
    .sum()
    .sum()
)


print(
    "NaN values:",
    nan_count
)


inf_count = np.isinf(
    enhanced[
        ml_features
    ]
    .select_dtypes(
        include=np.number
    )
).sum().sum()


print(
    "Infinite values:",
    inf_count
)


# ============================================================
# PATIENT LEAKAGE CHECK
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PATIENT LEAKAGE CHECK"
)

print("=" * 70
)


patient_fold_counts = (
    enhanced
    .groupby(
        "patient_id"
    )["fold"]
    .nunique()
)


leakage = (
    patient_fold_counts
    > 1
)


print(
    "Patients:",
    patient_fold_counts.shape[0]
)

print(
    "Patients in multiple folds:",
    leakage.sum()
)


if leakage.sum() == 0:

    print(
        "SUCCESS: No patient appears in multiple folds."
    )

else:

    print(
        "WARNING: Patient leakage detected!"
    )


# ============================================================
# SAVE
# ============================================================

enhanced.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\n" + "=" * 70
)

print(
    "ENHANCED DATASET READY"
)

print("=" * 70)


print(
    "Rows:",
    len(enhanced)
)

print(
    "Columns:",
    len(enhanced.columns)
)

print(
    "ML features:",
    len(ml_features)
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)