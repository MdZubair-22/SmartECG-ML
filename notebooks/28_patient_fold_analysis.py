import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML"
)

INPUT_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "enhanced_causal_ecg_features.csv"
)

RESULTS_DIR = BASE_DIR / "results"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 28A - PATIENT / FOLD ANALYSIS")
print("=" * 70)

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

print("\nDataset:")
print(df.shape)

# ============================================================
# NORMALIZE PATIENT ID
# ============================================================

print("\n" + "=" * 70)
print("PATIENT ID NORMALIZATION")
print("=" * 70)

df["patient_id"] = (
    df["patient_id"]
    .astype(str)
    .str.strip()
)

print(
    "\nUnique patients:",
    df["patient_id"].nunique()
)

# ============================================================
# CHECK PATIENT → FOLD MAPPING
# ============================================================

patient_fold_counts = (
    df.groupby("patient_id")["fold"]
    .nunique()
)

multiple_fold_patients = (
    patient_fold_counts[
        patient_fold_counts > 1
    ]
)

print(
    "\nPatients appearing in multiple folds:",
    len(multiple_fold_patients)
)

if len(multiple_fold_patients) == 0:

    print(
        "SUCCESS: Every patient belongs to exactly one fold."
    )

else:

    print(
        "WARNING: Patient leakage detected!"
    )

    print(
        multiple_fold_patients
    )

# ============================================================
# PATIENT → FOLD MAPPING
# ============================================================

patient_fold = (
    df.groupby("patient_id")["fold"]
    .first()
    .reset_index()
)

print("\n" + "=" * 70)
print("PATIENT → FOLD MAPPING")
print("=" * 70)

for fold in sorted(
    patient_fold["fold"].unique()
):

    patients = (
        patient_fold[
            patient_fold["fold"] == fold
        ]["patient_id"]
        .tolist()
    )

    print(
        f"\nFOLD {fold}"
    )

    print("-" * 50)

    print(
        "Number of patients:",
        len(patients)
    )

    print(
        patients
    )

# ============================================================
# F CLASS BY PATIENT
# ============================================================

f_df = df[
    df["class"] == "F"
].copy()

f_patient = (
    f_df.groupby(
        ["patient_id", "fold"]
    )
    .size()
    .reset_index(
        name="F_samples"
    )
    .sort_values(
        "F_samples",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("F CLASS BY PATIENT")
print("=" * 70)

print(
    f_patient
)

# ============================================================
# F CLASS BY FOLD
# ============================================================

f_fold = (
    f_df.groupby("fold")
    .size()
)

print("\n" + "=" * 70)
print("F CLASS BY FOLD")
print("=" * 70)

print(
    f_fold
)

# ============================================================
# F TRAINING AVAILABILITY
# ============================================================

print("\n" + "=" * 70)
print("F TRAINING AVAILABILITY")
print("=" * 70)

for test_fold in sorted(
    df["fold"].unique()
):

    train_df = df[
        df["fold"] != test_fold
    ]

    test_df = df[
        df["fold"] == test_fold
    ]

    train_f = (
        train_df["class"] == "F"
    ).sum()

    test_f = (
        test_df["class"] == "F"
    ).sum()

    train_f_patients = (
        train_df[
            train_df["class"] == "F"
        ]["patient_id"]
        .nunique()
    )

    test_f_patients = (
        test_df[
            test_df["class"] == "F"
        ]["patient_id"]
        .nunique()
    )

    print(
        f"\nTEST FOLD {test_fold}"
    )

    print(
        "Training F samples:",
        train_f
    )

    print(
        "Testing F samples:",
        test_f
    )

    print(
        "Training F patients:",
        train_f_patients
    )

    print(
        "Testing F patients:",
        test_f_patients
    )

# ============================================================
# CLASS DISTRIBUTION BY FOLD
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION BY FOLD")
print("=" * 70)

class_fold = pd.crosstab(
    df["fold"],
    df["class"]
)

print(
    class_fold
)

# ============================================================
# UNIQUE PATIENTS PER CLASS
# ============================================================

print("\n" + "=" * 70)
print("UNIQUE PATIENTS CONTAINING EACH CLASS")
print("=" * 70)

class_patient_counts = (
    df.groupby("class")["patient_id"]
    .nunique()
)

print(
    class_patient_counts
)

# ============================================================
# CLASS / PATIENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION BY PATIENT/FOLD")
print("=" * 70)

class_fold_patient = (
    df.groupby(
        ["fold", "class"]
    )["patient_id"]
    .nunique()
    .unstack(
        fill_value=0
    )
)

print(
    class_fold_patient
)

# ============================================================
# SAVE RESULTS
# ============================================================

patient_fold.to_csv(
    RESULTS_DIR /
    "step28_patient_fold_mapping.csv",
    index=False
)

f_patient.to_csv(
    RESULTS_DIR /
    "step28_F_patient_fold_analysis.csv",
    index=False
)

class_fold_patient.to_csv(
    RESULTS_DIR /
    "step28_class_patient_distribution.csv"
)

print("\n" + "=" * 70)
print("STEP 28A COMPLETE")
print("=" * 70)

print("\nSaved:")

print(
    "results/step28_patient_fold_mapping.csv"
)

print(
    "results/step28_F_patient_fold_analysis.csv"
)

print(
    "results/step28_class_patient_distribution.csv"
)