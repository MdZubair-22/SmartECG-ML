"""
Step 35 - Gradient Boosting Comparison
---------------------------------------
Patient-wise 3-fold evaluation of HistGradientBoostingClassifier.

Compares:
1. Standard Random Forest
2. HistGradientBoostingClassifier

Dataset:
data/processed/enhanced_causal_ecg_features.csv

Features:
38 ECG features

Classes:
N, S, V, F, Q
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_causal_ecg_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

CLASS_ORDER = ["N", "S", "V", "F", "Q"]

META_COLUMNS = [
    "record",
    "patient_id",
    "sample",
    "class",
    "fold"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("STEP 35 - GRADIENT BOOSTING MODEL COMPARISON")
print("=" * 80)

print(f"\nLoading dataset:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

required_columns = set(META_COLUMNS)

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# PREPARE FEATURES
# ============================================================

feature_columns = [
    c for c in df.columns
    if c not in META_COLUMNS
]

print(f"\nNumber of ML features: {len(feature_columns)}")
print("Features:")

for i, feature in enumerate(feature_columns, 1):
    print(f"  {i:02d}. {feature}")


X = df[feature_columns].copy()
y = df["class"].copy()

folds = df["fold"].astype(int).values
patients = df["patient_id"].values

print("\nClass distribution:")
print(y.value_counts().reindex(CLASS_ORDER))

print("\nFold distribution:")
print(df["fold"].value_counts().sort_index())

print(f"\nUnique patients: {df['patient_id'].nunique()}")


# ============================================================
# VALIDATION
# ============================================================

print("\nChecking patient leakage...")

leakage_found = False

for fold in sorted(df["fold"].unique()):

    test_patients = set(
        df.loc[df["fold"] == fold, "patient_id"]
    )

    other_patients = set(
        df.loc[df["fold"] != fold, "patient_id"]
    )

    overlap = test_patients.intersection(other_patients)

    if overlap:
        leakage_found = True
        print(
            f"WARNING: Fold {fold} has "
            f"{len(overlap)} overlapping patients"
        )

if not leakage_found:
    print("✓ No patient leakage detected")


# ============================================================
# STORAGE
# ============================================================

models = {
    "random_forest": [],
    "hist_gradient_boosting": []
}

predictions = {
    "random_forest": np.empty(len(df), dtype=object),
    "hist_gradient_boosting": np.empty(len(df), dtype=object)
}

fold_metrics = {
    "random_forest": [],
    "hist_gradient_boosting": []
}


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_random_forest():

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def create_gradient_boosting():

    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [make_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ============================================================
# PATIENT-WISE 3-FOLD OOF
# ============================================================

unique_folds = sorted(df["fold"].unique())

print("\n" + "=" * 80)
print("PATIENT-WISE 3-FOLD EVALUATION")
print("=" * 80)

for test_fold in unique_folds:

    print("\n" + "-" * 80)
    print(f"TEST FOLD {test_fold}")
    print("-" * 80)

    train_mask = folds != test_fold
    test_mask = folds == test_fold

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")

    print(
        f"Training patients: "
        f"{df.loc[train_mask, 'patient_id'].nunique()}"
    )

    print(
        f"Test patients: "
        f"{df.loc[test_mask, 'patient_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\n[1/2] Training Random Forest...")

    start = time.time()

    rf = create_random_forest()

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    elapsed = time.time() - start

    predictions["random_forest"][test_mask] = rf_pred

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_bal = balanced_accuracy_score(y_test, rf_pred)
    rf_macro = f1_score(
        y_test,
        rf_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )
    rf_weighted = f1_score(
        y_test,
        rf_pred,
        labels=CLASS_ORDER,
        average="weighted",
        zero_division=0
    )

    print(f"Training time: {elapsed:.1f} sec")
    print(f"Accuracy:           {rf_acc:.4f}")
    print(f"Balanced Accuracy:  {rf_bal:.4f}")
    print(f"Macro F1:            {rf_macro:.4f}")
    print(f"Weighted F1:         {rf_weighted:.4f}")

    fold_metrics["random_forest"].append({
        "fold": int(test_fold),
        "accuracy": float(rf_acc),
        "balanced_accuracy": float(rf_bal),
        "macro_f1": float(rf_macro),
        "weighted_f1": float(rf_weighted)
    })

    # --------------------------------------------------------
    # HistGradientBoosting
    # --------------------------------------------------------

    print("\n[2/2] Training HistGradientBoosting...")

    start = time.time()

    gb = create_gradient_boosting()

    gb.fit(X_train, y_train)

    gb_pred = gb.predict(X_test)

    elapsed = time.time() - start

    predictions["hist_gradient_boosting"][test_mask] = gb_pred

    gb_acc = accuracy_score(y_test, gb_pred)
    gb_bal = balanced_accuracy_score(y_test, gb_pred)
    gb_macro = f1_score(
        y_test,
        gb_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )
    gb_weighted = f1_score(
        y_test,
        gb_pred,
        labels=CLASS_ORDER,
        average="weighted",
        zero_division=0
    )

    print(f"Training time: {elapsed:.1f} sec")
    print(f"Accuracy:           {gb_acc:.4f}")
    print(f"Balanced Accuracy:  {gb_bal:.4f}")
    print(f"Macro F1:            {gb_macro:.4f}")
    print(f"Weighted F1:         {gb_weighted:.4f}")

    fold_metrics["hist_gradient_boosting"].append({
        "fold": int(test_fold),
        "accuracy": float(gb_acc),
        "balanced_accuracy": float(gb_bal),
        "macro_f1": float(gb_macro),
        "weighted_f1": float(gb_weighted)
    })


# ============================================================
# FINAL OOF EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL OUT-OF-FOLD RESULTS")
print("=" * 80)


comparison = []


for model_name in [
    "random_forest",
    "hist_gradient_boosting"
]:

    pred = predictions[model_name]

    acc = accuracy_score(y, pred)

    bal = balanced_accuracy_score(
        y,
        pred
    )

    macro = f1_score(
        y,
        pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )

    weighted = f1_score(
        y,
        pred,
        labels=CLASS_ORDER,
        average="weighted",
        zero_division=0
    )

    comparison.append({
        "model": model_name,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted)
    })

    print("\n" + "-" * 80)
    print(model_name.upper())
    print("-" * 80)

    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced Accuracy:  {bal:.4f}")
    print(f"Macro F1:            {macro:.4f}")
    print(f"Weighted F1:         {weighted:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            pred,
            labels=CLASS_ORDER,
            zero_division=0,
            digits=4
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y,
        pred,
        labels=CLASS_ORDER
    )

    print(
        pd.DataFrame(
            cm,
            index=CLASS_ORDER,
            columns=CLASS_ORDER
        )
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(comparison)

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# PER-CLASS RECALL
# ============================================================

print("\n" + "=" * 80)
print("PER-CLASS RECALL")
print("=" * 80)

recall_rows = []

for model_name in [
    "random_forest",
    "hist_gradient_boosting"
]:

    pred = predictions[model_name]

    report = classification_report(
        y,
        pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0
    )

    row = {
        "model": model_name
    }

    for cls in CLASS_ORDER:
        row[f"{cls}_recall"] = report[cls]["recall"]

    recall_rows.append(row)


recall_df = pd.DataFrame(recall_rows)

print(
    recall_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output_predictions = df[
    ["record", "patient_id", "sample", "class", "fold"]
].copy()

output_predictions["random_forest_prediction"] = predictions[
    "random_forest"
]

output_predictions["hist_gradient_boosting_prediction"] = predictions[
    "hist_gradient_boosting"
]

prediction_path = RESULTS_DIR / "step35_predictions.csv"

output_predictions.to_csv(
    prediction_path,
    index=False,
    chunksize=10000
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_output = {
    "dataset": str(DATA_PATH),
    "n_samples": int(len(df)),
    "n_features": int(len(feature_columns)),
    "features": feature_columns,
    "classes": CLASS_ORDER,
    "n_patients": int(df["patient_id"].nunique()),
    "folds": [int(v) for v in unique_folds],
    "comparison": comparison,
    "per_class_recall": recall_df.to_dict(orient="records"),
    "fold_metrics": fold_metrics
}

metrics_path = RESULTS_DIR / "step35_gradient_boosting_metrics.json"

with open(metrics_path, "w") as f:
    json.dump(
        make_json_safe(metrics_output),
        f,
        indent=2
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("STEP 35 COMPLETED")
print("=" * 80)

print("\nSaved:")
print(f"  {prediction_path}")
print(f"  {metrics_path}")

print("\nNext step:")
print("Compare the models using Balanced Accuracy + Macro F1")
print("and inspect S/F/Q recall before selecting the next approach.")