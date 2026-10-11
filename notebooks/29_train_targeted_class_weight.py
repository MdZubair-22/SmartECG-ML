import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/enhanced_causal_ecg_features.csv"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_RESULTS = os.path.join(
    RESULTS_DIR,
    "targeted_weighted_rf_results.csv"
)

OUTPUT_OOF = os.path.join(
    RESULTS_DIR,
    "targeted_weighted_oof_predictions.csv"
)

OUTPUT_IMPORTANCE = os.path.join(
    RESULTS_DIR,
    "targeted_weighted_feature_importance.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 29 - TARGETED CLASS-WEIGHTED RANDOM FOREST")
print("=" * 70)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("\nDataset shape:")
print(df.shape)

# ============================================================
# NORMALIZE PATIENT IDs
# ============================================================

df["patient_id"] = df["patient_id"].astype(str)

# ============================================================
# FEATURES
# ============================================================

EXCLUDE_COLUMNS = [
    "record",
    "patient_id",
    "sample",
    "class",
    "fold"
]

FEATURES = [
    c for c in df.columns
    if c not in EXCLUDE_COLUMNS
]

TARGET = "class"

print("\nNumber of ML features:", len(FEATURES))

print("\nFeatures:")
for feature in FEATURES:
    print(" -", feature)

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

print("NaN values:", df[FEATURES].isna().sum().sum())
print(
    "Infinite values:",
    np.isinf(df[FEATURES].to_numpy()).sum()
)

print("\nClasses:")
print(df[TARGET].value_counts())

# ============================================================
# PATIENT LEAKAGE CHECK
# ============================================================

patient_fold_counts = (
    df.groupby("patient_id")["fold"]
    .nunique()
)

leaking_patients = patient_fold_counts[
    patient_fold_counts > 1
]

print("\nPatient leakage check:")

if len(leaking_patients) == 0:
    print("SUCCESS: No patient appears in multiple folds.")
else:
    print("WARNING: Patient leakage detected!")
    print(leaking_patients)

# ============================================================
# TARGETED CLASS WEIGHTS
# ============================================================

# Baseline model had very poor F and S recall.
#
# We increase their importance while keeping N relatively
# dominant and avoiding an extremely aggressive weighting.

CLASS_WEIGHTS = {
    "N": 1.0,
    "S": 2.5,
    "V": 1.0,
    "F": 8.0,
    "Q": 1.2
}

print("\n" + "=" * 70)
print("TARGETED CLASS WEIGHTS")
print("=" * 70)

for cls, weight in CLASS_WEIGHTS.items():
    print(f"{cls}: {weight}")

# ============================================================
# CROSS-VALIDATION
# ============================================================

fold_results = []
oof_predictions = []

folds = sorted(df["fold"].unique())

for test_fold in folds:

    print("\n" + "=" * 70)
    print(f"TEST FOLD: {test_fold}")
    print("=" * 70)

    train_df = df[df["fold"] != test_fold].copy()
    test_df = df[df["fold"] == test_fold].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\nTraining samples:", len(train_df))
    print("Testing samples:", len(test_df))

    print("\nTraining distribution:")
    print(y_train.value_counts())

    print("\nTesting distribution:")
    print(y_test.value_counts())

    # ========================================================
    # MODEL
    # ========================================================

    print("\nTraining targeted weighted Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=CLASS_WEIGHTS,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Training complete.")

    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = model.predict(X_test)

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(y_test, y_pred)

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print("\nResults:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=["N", "S", "V", "F", "Q"],
            zero_division=0
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["N", "S", "V", "F", "Q"]
    )

    print("Confusion Matrix:")

    print(
        pd.DataFrame(
            cm,
            index=["N", "S", "V", "F", "Q"],
            columns=["N", "S", "V", "F", "Q"]
        )
    )

    # ========================================================
    # STORE FOLD RESULT
    # ========================================================

    fold_results.append({
        "fold": test_fold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })

    # ========================================================
    # OUT-OF-FOLD PREDICTIONS
    # ========================================================

    fold_oof = test_df[
        ["record", "patient_id", "sample", "fold"]
    ].copy()

    fold_oof["true_class"] = y_test.values
    fold_oof["predicted_class"] = y_pred

    oof_predictions.append(fold_oof)

# ============================================================
# COMBINE OOF PREDICTIONS
# ============================================================

oof_df = pd.concat(
    oof_predictions,
    ignore_index=True
)

oof_df.to_csv(
    OUTPUT_OOF,
    index=False
)

# ============================================================
# OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL TARGETED WEIGHTED RANDOM FOREST")
print("=" * 70)

y_true_all = oof_df["true_class"]
y_pred_all = oof_df["predicted_class"]

overall_accuracy = accuracy_score(
    y_true_all,
    y_pred_all
)

overall_balanced_accuracy = balanced_accuracy_score(
    y_true_all,
    y_pred_all
)

overall_macro_f1 = f1_score(
    y_true_all,
    y_pred_all,
    average="macro"
)

overall_weighted_f1 = f1_score(
    y_true_all,
    y_pred_all,
    average="weighted"
)

print(
    f"\nOverall Accuracy: "
    f"{overall_accuracy:.4f}"
)

print(
    f"Overall Balanced Accuracy: "
    f"{overall_balanced_accuracy:.4f}"
)

print(
    f"Overall Macro F1: "
    f"{overall_macro_f1:.4f}"
)

print(
    f"Overall Weighted F1: "
    f"{overall_weighted_f1:.4f}"
)

# ============================================================
# OVERALL CLASSIFICATION REPORT
# ============================================================

print("\nOverall Classification Report:")

print(
    classification_report(
        y_true_all,
        y_pred_all,
        labels=["N", "S", "V", "F", "Q"],
        zero_division=0
    )
)

# ============================================================
# OVERALL CONFUSION MATRIX
# ============================================================

overall_cm = confusion_matrix(
    y_true_all,
    y_pred_all,
    labels=["N", "S", "V", "F", "Q"]
)

print("\nOverall Confusion Matrix:")

print(
    pd.DataFrame(
        overall_cm,
        index=["N", "S", "V", "F", "Q"],
        columns=["N", "S", "V", "F", "Q"]
    )
)

# ============================================================
# FOLD SUMMARY
# ============================================================

results_df = pd.DataFrame(fold_results)

print("\nFold Summary:")
print(results_df)

print(
    "\nMean Macro F1:",
    round(results_df["macro_f1"].mean(), 4)
)

print(
    "Std Macro F1:",
    round(results_df["macro_f1"].std(), 4)
)

# ============================================================
# TRAIN FINAL MODEL FOR FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL FEATURE IMPORTANCE")
print("=" * 70)

X_all = df[FEATURES]
y_all = df[TARGET]

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight=CLASS_WEIGHTS,
    random_state=42,
    n_jobs=-1
)

print("\nTraining final model...")

final_model.fit(
    X_all,
    y_all
)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": final_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\nTop 20 Features:")

print(
    importance_df.head(20).to_string(index=False)
)

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

importance_df.to_csv(
    OUTPUT_IMPORTANCE,
    index=False
)

print("\n" + "=" * 70)
print("STEP 29 COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    f" - {OUTPUT_RESULTS}"
)

print(
    f" - {OUTPUT_OOF}"
)

print(
    f" - {OUTPUT_IMPORTANCE}"
)