import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML"
)

INPUT_PATH = (
    BASE_DIR /
    "results" /
    "enhanced_oof_predictions.csv"
)

RESULTS_DIR = BASE_DIR / "results"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ENHANCED ECG MODEL ERROR ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "record",
    "patient_id",
    "sample",
    "fold",
    "true_class",
    "predicted_class"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print("\nBasic validation:")
print("Missing values:")
print(df[required_columns].isna().sum())

# ============================================================
# PATIENT LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

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
    "\nPatients in multiple folds:",
    len(patients_multiple_folds)
)

if len(patients_multiple_folds) == 0:
    print(
        "SUCCESS: No patient appears in multiple folds."
    )
else:
    print(
        "WARNING: Patient leakage detected!"
    )

# ============================================================
# OVERALL METRICS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

y_true = df["true_class"]
y_pred = df["predicted_class"]

accuracy = (
    (y_true == y_pred).mean()
)

balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_acc:.4f}"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    labels=["N", "S", "V", "F", "Q"],
    zero_division=0
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

classes = ["N", "S", "V", "F", "Q"]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\nRows = True class"
)

print(
    "Columns = Predicted class\n"
)

print(cm_df)

cm_path = (
    RESULTS_DIR /
    "step27_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

# ============================================================
# ERROR DATASET
# ============================================================

df["correct"] = (
    df["true_class"] ==
    df["predicted_class"]
)

errors = df[
    ~df["correct"]
].copy()

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print(
    "\nTotal samples:",
    len(df)
)

print(
    "Correct predictions:",
    df["correct"].sum()
)

print(
    "Incorrect predictions:",
    len(errors)
)

print(
    "Error rate:",
    len(errors) / len(df)
)

# ============================================================
# ERROR COUNT BY TRUE CLASS
# ============================================================

print("\n" + "=" * 70)
print("ERRORS BY TRUE CLASS")
print("=" * 70)

error_by_class = (
    errors["true_class"]
    .value_counts()
    .reindex(classes, fill_value=0)
)

total_by_class = (
    df["true_class"]
    .value_counts()
    .reindex(classes, fill_value=0)
)

error_percentage = (
    error_by_class /
    total_by_class *
    100
)

error_summary = pd.DataFrame({
    "total_samples": total_by_class,
    "errors": error_by_class,
    "error_percentage": error_percentage
})

print(error_summary)

error_summary_path = (
    RESULTS_DIR /
    "step27_class_error_analysis.csv"
)

error_summary.to_csv(
    error_summary_path
)

# ============================================================
# MISCLASSIFICATION PAIRS
# ============================================================

print("\n" + "=" * 70)
print("TOP MISCLASSIFICATION PAIRS")
print("=" * 70)

misclassification_pairs = (
    errors.groupby(
        ["true_class", "predicted_class"]
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False
    )
)

print(
    misclassification_pairs.head(20)
)

pairs_path = (
    RESULTS_DIR /
    "step27_misclassification_pairs.csv"
)

misclassification_pairs.to_csv(
    pairs_path,
    index=False
)

# ============================================================
# CLASS-WISE PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTIONS BY TRUE CLASS")
print("=" * 70)

prediction_distribution = pd.crosstab(
    df["true_class"],
    df["predicted_class"]
)

prediction_distribution = (
    prediction_distribution
    .reindex(
        index=classes,
        columns=classes,
        fill_value=0
    )
)

print(
    prediction_distribution
)

prediction_distribution.to_csv(
    RESULTS_DIR /
    "step27_prediction_distribution.csv"
)

# ============================================================
# F CLASS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("F CLASS ERROR ANALYSIS")
print("=" * 70)

f_df = df[
    df["true_class"] == "F"
].copy()

print(
    "\nTotal F samples:",
    len(f_df)
)

print(
    "\nF prediction distribution:"
)

print(
    f_df["predicted_class"]
    .value_counts()
)

print(
    "\nF errors by predicted class:"
)

print(
    f_df[
        f_df["predicted_class"] != "F"
    ]["predicted_class"]
    .value_counts()
)

# F class by patient
f_patient = (
    f_df.groupby(
        "patient_id"
    )
    .agg(
        total_F=("true_class", "size"),
        correctly_detected=(
            "predicted_class",
            lambda x: (x == "F").sum()
        )
    )
)

f_patient["recall"] = (
    f_patient["correctly_detected"] /
    f_patient["total_F"]
)

f_patient = (
    f_patient
    .sort_values(
        "total_F",
        ascending=False
    )
)

print(
    "\nF performance by patient:"
)

print(
    f_patient
)

f_patient.to_csv(
    RESULTS_DIR /
    "step27_F_patient_analysis.csv"
)

# ============================================================
# S CLASS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("S CLASS ERROR ANALYSIS")
print("=" * 70)

s_df = df[
    df["true_class"] == "S"
].copy()

print(
    "\nTotal S samples:",
    len(s_df)
)

print(
    "\nS prediction distribution:"
)

print(
    s_df["predicted_class"]
    .value_counts()
)

print(
    "\nS errors by predicted class:"
)

print(
    s_df[
        s_df["predicted_class"] != "S"
    ]["predicted_class"]
    .value_counts()
)

# ============================================================
# Q CLASS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("Q CLASS ERROR ANALYSIS")
print("=" * 70)

q_df = df[
    df["true_class"] == "Q"
].copy()

print(
    "\nTotal Q samples:",
    len(q_df)
)

print(
    "\nQ prediction distribution:"
)

print(
    q_df["predicted_class"]
    .value_counts()
)

# ============================================================
# V CLASS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("V CLASS ANALYSIS")
print("=" * 70)

v_df = df[
    df["true_class"] == "V"
].copy()

print(
    "\nTotal V samples:",
    len(v_df)
)

print(
    "\nV prediction distribution:"
)

print(
    v_df["predicted_class"]
    .value_counts()
)

# ============================================================
# ERROR RATE BY PATIENT
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-WISE ERROR ANALYSIS")
print("=" * 70)

patient_summary = (
    df.groupby("patient_id")
    .agg(
        total_samples=("true_class", "size"),
        errors=(
            "correct",
            lambda x: (~x).sum()
        )
    )
)

patient_summary["error_rate"] = (
    patient_summary["errors"] /
    patient_summary["total_samples"]
)

patient_summary = (
    patient_summary
    .sort_values(
        "error_rate",
        ascending=False
    )
)

print(
    "\nPatients with highest error rates:"
)

print(
    patient_summary.head(20)
)

patient_summary.to_csv(
    RESULTS_DIR /
    "step27_patient_error_analysis.csv"
)

# ============================================================
# ERROR RATE BY FOLD
# ============================================================

print("\n" + "=" * 70)
print("FOLD-WISE ERROR ANALYSIS")
print("=" * 70)

fold_summary = (
    df.groupby("fold")
    .agg(
        total_samples=("true_class", "size"),
        errors=(
            "correct",
            lambda x: (~x).sum()
        )
    )
)

fold_summary["accuracy"] = (
    1 -
    fold_summary["errors"] /
    fold_summary["total_samples"]
)

print(
    fold_summary
)

fold_summary.to_csv(
    RESULTS_DIR /
    "step27_fold_error_analysis.csv"
)

# ============================================================
# CLASS PERFORMANCE BY FOLD
# ============================================================

print("\n" + "=" * 70)
print("CLASS RECALL BY FOLD")
print("=" * 70)

fold_class_recall = []

for fold in sorted(df["fold"].unique()):

    fold_df = df[
        df["fold"] == fold
    ]

    for cls in classes:

        class_rows = fold_df[
            fold_df["true_class"] == cls
        ]

        if len(class_rows) == 0:
            recall = np.nan
        else:
            recall = (
                class_rows["predicted_class"]
                == cls
            ).mean()

        fold_class_recall.append({
            "fold": fold,
            "class": cls,
            "samples": len(class_rows),
            "recall": recall
        })

fold_class_recall_df = pd.DataFrame(
    fold_class_recall
)

print(
    fold_class_recall_df
)

fold_class_recall_df.to_csv(
    RESULTS_DIR /
    "step27_fold_class_recall.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 27 COMPLETE")
print("=" * 70)

print(
    "\nGenerated files:"
)

files = [
    "step27_confusion_matrix.csv",
    "step27_class_error_analysis.csv",
    "step27_misclassification_pairs.csv",
    "step27_prediction_distribution.csv",
    "step27_F_patient_analysis.csv",
    "step27_patient_error_analysis.csv",
    "step27_fold_error_analysis.csv",
    "step27_fold_class_recall.csv"
]

for filename in files:
    print(
        f" - results/{filename}"
    )

print(
    "\nUse these results to decide the next targeted improvement."
)