"""
STEP 47A — DATASET & PATIENT INTEGRITY CHECK

Purpose:
    Investigate the discrepancy in the reported number of patients
    between previous project steps.

    Step 43 reported:
        49 patients

    Step 45 reported:
        47 patients

    This script checks:
        1. patient_id data types
        2. unique patient IDs
        3. missing patient IDs
        4. duplicate rows
        5. patient/fold consistency
        6. patient appearing in multiple folds
        7. samples per patient
        8. samples per patient per fold
        9. class distribution per patient
       10. record/patient relationships
       11. suspicious patient IDs
       12. duplicate sample identifiers

IMPORTANT:
    This script DOES NOT modify the dataset.
    It only performs an integrity audit.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enhanced_causal_ecg_features.csv"
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. START
# ============================================================

print("=" * 80)
print("STEP 47A — DATASET & PATIENT INTEGRITY CHECK")
print("=" * 80)

start_time = time.time()


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n[1/10] Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Rows: {len(df):,}"
)


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

print("\n[2/10] Checking required identifiers...")

required = [
    "patient_id",
    "record",
    "sample",
    "class",
    "fold",
]

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print(
    "Required identifier columns found."
)


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n[3/10] Inspecting identifier data types...")

print(
    "\nColumn data types:"
)

for col in required:

    print(
        f"{col:15s}: "
        f"{df[col].dtype}"
    )


# ============================================================
# 6. MISSING IDENTIFIERS
# ============================================================

print("\n[4/10] Checking missing identifiers...")

missing_summary = {}

for col in required:

    count = int(
        df[col].isna().sum()
    )

    missing_summary[col] = count

    print(
        f"{col:15s}: "
        f"{count:,} missing"
    )


# ============================================================
# 7. UNIQUE PATIENT ANALYSIS
# ============================================================

print("\n[5/10] Investigating patient IDs...")

patient_ids = (
    df["patient_id"]
    .dropna()
    .unique()
)

print(
    f"\nUnique patient IDs: "
    f"{len(patient_ids)}"
)

print("\nPatient IDs:")

for patient in sorted(
    patient_ids,
    key=lambda x: str(x),
):

    print(
        f"  {repr(patient)}"
    )


# ============================================================
# 8. PATIENT ID VALUE TYPES
# ============================================================

print(
    "\nPatient ID Python value types:"
)

type_counts = (
    df["patient_id"]
    .map(lambda x: type(x).__name__)
    .value_counts()
)

print(type_counts)


# ============================================================
# 9. PATIENT NORMALIZATION CHECK
# ============================================================

print(
    "\nChecking for IDs that differ only by formatting..."
)

patient_as_string = (
    df["patient_id"]
    .astype(str)
    .str.strip()
)

normalized_groups = (
    df.assign(
        patient_id_string=patient_as_string
    )
    .groupby("patient_id_string")[
        "patient_id"
    ]
    .nunique()
)

suspicious_normalized = (
    normalized_groups[
        normalized_groups > 1
    ]
)

if len(suspicious_normalized) == 0:

    print(
        "No formatting-based patient ID "
        "duplicates detected."
    )

else:

    print(
        "Potential formatting-based duplicates:"
    )

    print(
        suspicious_normalized
    )


# ============================================================
# 10. PATIENT → FOLD CONSISTENCY
# ============================================================

print(
    "\n[6/10] Checking patient-wise fold consistency..."
)

patient_fold_counts = (
    df.groupby("patient_id")["fold"]
    .nunique()
)

patients_multiple_folds = (
    patient_fold_counts[
        patient_fold_counts > 1
    ]
)

print(
    f"Patients appearing in exactly one fold: "
    f"{(patient_fold_counts == 1).sum()}"
)

print(
    f"Patients appearing in multiple folds: "
    f"{len(patients_multiple_folds)}"
)

if len(patients_multiple_folds) > 0:

    print(
        "\nWARNING — Patient leakage candidates:"
    )

    print(
        patients_multiple_folds
    )

else:

    print(
        "PASS — Every patient belongs to exactly one fold."
    )


# ============================================================
# 11. PATIENT/FOLD TABLE
# ============================================================

patient_fold_table = (
    df.groupby(
        ["patient_id", "fold"]
    )
    .size()
    .reset_index(
        name="sample_count"
    )
)

print(
    "\nPatient/fold distribution:"
)

print(
    patient_fold_table.to_string(
        index=False
    )
)


# ============================================================
# 12. DUPLICATE ROWS
# ============================================================

print(
    "\n[7/10] Checking duplicate rows..."
)

duplicate_rows = int(
    df.duplicated().sum()
)

print(
    f"Exact duplicate rows: "
    f"{duplicate_rows:,}"
)


# ============================================================
# 13. RECORD/PATIENT RELATIONSHIP
# ============================================================

print(
    "\nChecking record → patient relationship..."
)

record_patient_counts = (
    df.groupby("record")["patient_id"]
    .nunique()
)

records_multiple_patients = (
    record_patient_counts[
        record_patient_counts > 1
    ]
)

print(
    f"Unique records: "
    f"{df['record'].nunique()}"
)

print(
    f"Records linked to multiple patients: "
    f"{len(records_multiple_patients)}"
)

if len(records_multiple_patients) > 0:

    print(
        "\nWARNING:"
    )

    print(
        records_multiple_patients
    )

else:

    print(
        "PASS — Each record maps to one patient."
    )


# ============================================================
# 14. PATIENT → RECORD RELATIONSHIP
# ============================================================

patient_record_counts = (
    df.groupby("patient_id")["record"]
    .nunique()
)

print(
    "\nRecords per patient:"
)

print(
    patient_record_counts.describe()
)


# ============================================================
# 15. SAMPLE IDENTIFIER ANALYSIS
# ============================================================

print(
    "\n[8/10] Checking sample identifiers..."
)

print(
    f"Unique sample values: "
    f"{df['sample'].nunique():,}"
)

duplicate_sample_count = (
    df["sample"]
    .duplicated()
    .sum()
)

print(
    f"Rows with duplicated sample values: "
    f"{duplicate_sample_count:,}"
)

if duplicate_sample_count > 0:

    print(
        "\nIMPORTANT:"
    )

    print(
        "Sample values are not globally unique."
    )

    print(
        "They must NOT be used alone to merge "
        "patient information."
    )

    sample_counts = (
        df["sample"]
        .value_counts()
    )

    print(
        "\nTop repeated sample values:"
    )

    print(
        sample_counts.head(10)
    )


# ============================================================
# 16. SAMPLE UNIQUENESS WITH RECORD
# ============================================================

print(
    "\nChecking (record, sample) uniqueness..."
)

record_sample_duplicates = int(
    df.duplicated(
        subset=[
            "record",
            "sample",
        ]
    ).sum()
)

print(
    f"Duplicate (record, sample) pairs: "
    f"{record_sample_duplicates:,}"
)


# ============================================================
# 17. CLASS DISTRIBUTION
# ============================================================

print(
    "\n[9/10] Checking class distribution..."
)

class_distribution = (
    df["class"]
    .value_counts()
    .sort_index()
)

print(
    class_distribution
)


# ============================================================
# 18. CLASS DISTRIBUTION PER FOLD
# ============================================================

fold_class_distribution = (
    pd.crosstab(
        df["fold"],
        df["class"]
    )
    .reindex(
        columns=[
            "N",
            "S",
            "V",
            "F",
            "Q",
        ],
        fill_value=0,
    )
)

print(
    "\nClass distribution by fold:"
)

print(
    fold_class_distribution
)


# ============================================================
# 19. PATIENT CLASS DISTRIBUTION
# ============================================================

patient_class_distribution = (
    pd.crosstab(
        df["patient_id"],
        df["class"]
    )
    .reindex(
        columns=[
            "N",
            "S",
            "V",
            "F",
            "Q",
        ],
        fill_value=0,
    )
)

print(
    "\nPatient class distribution:"
)

print(
    patient_class_distribution
)


# ============================================================
# 20. PATIENT SAMPLE COUNTS
# ============================================================

patient_sample_counts = (
    df.groupby("patient_id")
    .size()
    .sort_values(
        ascending=False
    )
)

print(
    "\nSamples per patient:"
)

print(
    patient_sample_counts.describe()
)

print(
    "\nPatients with smallest sample counts:"
)

print(
    patient_sample_counts.tail(10)
)


# ============================================================
# 21. FOLD SAMPLE COUNTS
# ============================================================

fold_counts = (
    df["fold"]
    .value_counts()
    .sort_index()
)

print(
    "\nSamples per fold:"
)

print(
    fold_counts
)


# ============================================================
# 22. PATIENTS PER FOLD
# ============================================================

patients_per_fold = (
    df.groupby("fold")[
        "patient_id"
    ]
    .nunique()
)

print(
    "\nPatients per fold:"
)

print(
    patients_per_fold
)


# ============================================================
# 23. IDENTIFY POTENTIAL PATIENT DISCREPANCY
# ============================================================

print(
    "\n[10/10] Investigating 47 vs 49 patient discrepancy..."
)

actual_patient_count = (
    df["patient_id"]
    .nunique()
)

print(
    f"\nCurrent CSV unique patients: "
    f"{actual_patient_count}"
)

if actual_patient_count == 47:

    print(
        "\nCurrent dataset contains 47 patients."
    )

elif actual_patient_count == 49:

    print(
        "\nCurrent dataset contains 49 patients."
    )

else:

    print(
        "\nCurrent dataset contains "
        f"{actual_patient_count} patients."
    )


# ============================================================
# 24. SAVE AUDIT TABLES
# ============================================================

print(
    "\nSaving audit results..."
)


PATIENT_FOLD_PATH = (
    RESULTS_DIR
    / "step47A_patient_fold_table.csv"
)

patient_fold_table.to_csv(
    PATIENT_FOLD_PATH,
    index=False,
)


PATIENT_CLASS_PATH = (
    RESULTS_DIR
    / "step47A_patient_class_distribution.csv"
)

patient_class_distribution.to_csv(
    PATIENT_CLASS_PATH
)


FOLD_CLASS_PATH = (
    RESULTS_DIR
    / "step47A_fold_class_distribution.csv"
)

fold_class_distribution.to_csv(
    FOLD_CLASS_PATH
)


PATIENT_COUNT_PATH = (
    RESULTS_DIR
    / "step47A_patient_sample_counts.csv"
)

patient_sample_counts.rename(
    "sample_count"
).to_csv(
    PATIENT_COUNT_PATH
)


# ============================================================
# 25. CREATE JSON SUMMARY
# ============================================================

audit_summary = {

    "dataset": {
        "path": str(DATA_PATH),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    },

    "patients": {
        "unique_count": int(
            df["patient_id"].nunique()
        ),
        "ids": [
            str(x)
            for x in sorted(
                df["patient_id"]
                .dropna()
                .unique(),
                key=lambda x: str(x),
            )
        ],
    },

    "identifier_types": {
        col: str(
            df[col].dtype
        )
        for col in required
    },

    "missing_identifiers":
        missing_summary,

    "duplicate_rows":
        duplicate_rows,

    "record_analysis": {
        "unique_records": int(
            df["record"].nunique()
        ),
        "records_multiple_patients":
            int(
                len(records_multiple_patients)
            ),
    },

    "sample_analysis": {
        "unique_samples": int(
            df["sample"].nunique()
        ),
        "duplicate_sample_rows": int(
            duplicate_sample_count
        ),
        "duplicate_record_sample_pairs":
            int(
                record_sample_duplicates
            ),
    },

    "patient_fold_analysis": {
        "patients_one_fold": int(
            (
                patient_fold_counts == 1
            ).sum()
        ),
        "patients_multiple_folds": int(
            len(patients_multiple_folds)
        ),
    },

    "folds": {
        str(fold): {
            "samples": int(
                fold_counts.get(
                    fold,
                    0
                )
            ),
            "patients": int(
                patients_per_fold.get(
                    fold,
                    0
                )
            ),
        }
        for fold in sorted(
            df["fold"].unique()
        )
    },

    "class_distribution": {
        str(cls): int(count)
        for cls, count
        in class_distribution.items()
    },

    "integrity": {

        "patient_fold_leakage": (
            len(
                patients_multiple_folds
            ) > 0
        ),

        "record_patient_conflict": (
            len(
                records_multiple_patients
            ) > 0
        ),

        "duplicate_rows": (
            duplicate_rows > 0
        ),

        "missing_patient_ids": (
            missing_summary[
                "patient_id"
            ] > 0
        ),
    },

    "outputs": {

        "patient_fold_table":
            str(PATIENT_FOLD_PATH),

        "patient_class_distribution":
            str(PATIENT_CLASS_PATH),

        "fold_class_distribution":
            str(FOLD_CLASS_PATH),

        "patient_sample_counts":
            str(PATIENT_COUNT_PATH),
    },
}


JSON_PATH = (
    RESULTS_DIR
    / "step47A_dataset_patient_integrity.json"
)

with open(
    JSON_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        audit_summary,
        f,
        indent=4,
    )


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

elapsed = time.time() - start_time

print("\n")
print("=" * 80)
print("STEP 47A COMPLETE")
print("=" * 80)

print(
    f"\nDataset rows: "
    f"{len(df):,}"
)

print(
    f"Unique patients: "
    f"{df['patient_id'].nunique()}"
)

print(
    f"Unique records: "
    f"{df['record'].nunique()}"
)

print(
    f"Unique samples: "
    f"{df['sample'].nunique():,}"
)

print(
    f"Duplicate rows: "
    f"{duplicate_rows:,}"
)

print(
    f"Patients in multiple folds: "
    f"{len(patients_multiple_folds)}"
)

print(
    f"Records linked to multiple patients: "
    f"{len(records_multiple_patients)}"
)

print(
    "\nAudit files:"
)

print(
    PATIENT_FOLD_PATH
)

print(
    PATIENT_CLASS_PATH
)

print(
    FOLD_CLASS_PATH
)

print(
    PATIENT_COUNT_PATH
)

print(
    JSON_PATH
)

print(
    f"\nExecution time: "
    f"{elapsed / 60:.2f} minutes"
)

print(
    "\nNo dataset files were modified."
)

print(
    "\nSTEP 47A finished successfully."
)