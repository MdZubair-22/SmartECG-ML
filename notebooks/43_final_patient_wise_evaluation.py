"""
STEP 43 — Final Frozen Patient-Wise Evaluation

Purpose:
    Final post-selection OOF evaluation using:
    - Fixed 23-feature subset from Step 39
    - Best HGB hyperparameters from Step 42
    - Balanced class weights from Step 36/39
    - Strict patient-wise 3-fold OOF evaluation

IMPORTANT:
    This is a frozen post-selection OOF evaluation.
    Step 42 hyperparameters were selected using OOF results, so this
    should NOT be described as an independently unbiased test estimate.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "enhanced_causal_ecg_features.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

CLASSES = ["N", "S", "V", "F", "Q"]

# Fixed 23-feature subset selected in Step 39
FEATURES = [
    # Rhythm
    "prev_rr",
    "prev_hr",

    # Spectral
    "spectral_entropy",
    "dominant_frequency",
    "energy_ratio",

    # Morphology
    "morph_mean_abs_slope",
    "morph_range",
    "morph_max",
    "morph_std",
    "morph_slope30",
    "morph_slope20",
    "morph_annotation_amplitude",
    "morph_rms",
    "morph_annotation_pre30_diff",
    "morph_pre30_energy_ratio",
    "morph_pre10_mean",
    "morph_annotation_pre20_diff",
    "morph_pre20_energy_ratio",
    "morph_max_positive_slope",
    "morph_pre10_energy_ratio",
    "morph_slope10",
    "morph_max_negative_slope",
    "morph_mean",
]

# Step 42 best configuration by balanced accuracy
HGB_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": RANDOM_STATE,
}

# Step 36 balanced weighting
CLASS_WEIGHTS = {
    "N": 1.0,
    "S": 3.0,
    "V": 1.0,
    "F": 5.0,
    "Q": 1.5,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("STEP 43 — FINAL PATIENT-WISE EVALUATION")
print("=" * 80)

start_time = time.time()

print("\n[1/7] Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

required_columns = [
    "patient_id",
    "fold",
    "class",
] + FEATURES

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns:\n{missing}"
    )

print(f"Features used: {len(FEATURES)}")
print(f"Patients: {df['patient_id'].nunique()}")
print(f"Folds: {sorted(df['fold'].unique())}")


# ============================================================
# DATA QUALITY
# ============================================================

print("\n[2/7] Checking data quality...")

X = df[FEATURES].copy()
y = df["class"].astype(str)
folds = df["fold"].astype(int)

missing_count = int(X.isna().sum().sum())
infinite_count = int(np.isinf(X.to_numpy()).sum())

print(f"Missing values: {missing_count}")
print(f"Infinite values: {infinite_count}")

if missing_count > 0:
    raise ValueError("Dataset contains missing feature values.")

if infinite_count > 0:
    raise ValueError("Dataset contains infinite feature values.")


# ============================================================
# PATIENT-WISE OOF
# ============================================================

print("\n[3/7] Running strict patient-wise 3-fold OOF...")

unique_folds = sorted(folds.unique())

if unique_folds != [0, 1, 2]:
    raise ValueError(
        f"Expected folds [0, 1, 2], found {unique_folds}"
    )

oof_pred = np.empty(len(df), dtype=object)
oof_proba = np.zeros(
    (len(df), len(CLASSES)),
    dtype=np.float32
)

fold_results = []


for test_fold in unique_folds:

    print("\n" + "-" * 70)
    print(f"TEST FOLD: {test_fold}")
    print("-" * 70)

    train_mask = folds != test_fold
    test_mask = folds == test_fold

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"Training samples: {len(train_idx):,}")
    print(f"Testing samples : {len(test_idx):,}")

    train_patients = set(df.iloc[train_idx]["patient_id"])
    test_patients = set(df.iloc[test_idx]["patient_id"])

    overlap = train_patients.intersection(test_patients)

    print(f"Training patients: {len(train_patients)}")
    print(f"Testing patients : {len(test_patients)}")
    print(f"Patient overlap  : {len(overlap)}")

    if overlap:
        raise RuntimeError(
            f"PATIENT LEAKAGE DETECTED in fold {test_fold}: "
            f"{sorted(overlap)[:10]}"
        )

    # --------------------------------------------------------
    # Sample weights
    # --------------------------------------------------------

    sample_weights = y_train.map(CLASS_WEIGHTS).to_numpy()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HistGradientBoostingClassifier(
        **HGB_PARAMS
    )

    print("Training HGB...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    oof_pred[test_idx] = pred

    # HistGradientBoosting classes should match CLASSES
    model_class_to_index = {
        cls: i for i, cls in enumerate(model.classes_)
    }

    for local_i, global_i in enumerate(test_idx):

        for cls in CLASSES:
            if cls in model_class_to_index:
                oof_proba[
                    global_i,
                    CLASSES.index(cls)
                ] = probabilities[
                    local_i,
                    model_class_to_index[cls]
                ]

    # --------------------------------------------------------
    # Fold metrics
    # --------------------------------------------------------

    fold_acc = accuracy_score(y_test, pred)

    fold_bal = balanced_accuracy_score(
        y_test,
        pred
    )

    fold_macro_f1 = f1_score(
        y_test,
        pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    fold_weighted_f1 = f1_score(
        y_test,
        pred,
        labels=CLASSES,
        average="weighted",
        zero_division=0,
    )

    print(f"Accuracy          : {fold_acc:.4f}")
    print(f"Balanced Accuracy : {fold_bal:.4f}")
    print(f"Macro F1          : {fold_macro_f1:.4f}")
    print(f"Weighted F1       : {fold_weighted_f1:.4f}")

    fold_results.append({
        "fold": int(test_fold),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_patients": int(len(train_patients)),
        "test_patients": int(len(test_patients)),
        "accuracy": float(fold_acc),
        "balanced_accuracy": float(fold_bal),
        "macro_f1": float(fold_macro_f1),
        "weighted_f1": float(fold_weighted_f1),
    })


# ============================================================
# OVERALL METRICS
# ============================================================

print("\n[4/7] Calculating final OOF metrics...")

overall_accuracy = accuracy_score(
    y,
    oof_pred
)

overall_balanced_accuracy = balanced_accuracy_score(
    y,
    oof_pred
)

overall_macro_f1 = f1_score(
    y,
    oof_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0,
)

overall_weighted_f1 = f1_score(
    y,
    oof_pred,
    labels=CLASSES,
    average="weighted",
    zero_division=0,
)

overall_macro_precision = precision_score(
    y,
    oof_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0,
)

overall_macro_recall = recall_score(
    y,
    oof_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 80)
print("FINAL OOF PERFORMANCE")
print("=" * 80)

print(f"Accuracy           : {overall_accuracy:.4f}")
print(f"Balanced Accuracy  : {overall_balanced_accuracy:.4f}")
print(f"Macro Precision    : {overall_macro_precision:.4f}")
print(f"Macro Recall       : {overall_macro_recall:.4f}")
print(f"Macro F1           : {overall_macro_f1:.4f}")
print(f"Weighted F1        : {overall_weighted_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n[5/7] Per-class metrics...")

report = classification_report(
    y,
    oof_pred,
    labels=CLASSES,
    target_names=CLASSES,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).T

print("\n")
print(
    report_df[
        ["precision", "recall", "f1-score", "support"]
    ].round(4)
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n[6/7] Confusion matrix...")

cm = confusion_matrix(
    y,
    oof_pred,
    labels=CLASSES,
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{c}" for c in CLASSES],
    columns=[f"Predicted_{c}" for c in CLASSES],
)

print("\n")
print(cm_df)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\n[7/7] Saving final results...")

prediction_df = df[
    [
        "record",
        "patient_id",
        "sample",
        "class",
        "fold",
    ]
].copy()

prediction_df["prediction"] = oof_pred

for i, cls in enumerate(CLASSES):
    prediction_df[f"prob_{cls}"] = oof_proba[:, i]

prediction_path = (
    RESULTS_DIR /
    "step43_final_patient_wise_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = (
    RESULTS_DIR /
    "step43_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    RESULTS_DIR /
    "step43_classification_report.csv"
)

report_df.to_csv(report_path)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_df = pd.DataFrame(fold_results)

fold_path = (
    RESULTS_DIR /
    "step43_fold_metrics.csv"
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# SAVE ROC/PR DATA
# ============================================================

roc_pr_df = prediction_df[
    [
        "record",
        "patient_id",
        "sample",
        "class",
        "fold",
    ]
].copy()

for cls in CLASSES:
    roc_pr_df[f"prob_{cls}"] = prediction_df[f"prob_{cls}"]

roc_pr_path = (
    RESULTS_DIR /
    "step43_roc_pr_data.csv"
)

roc_pr_df.to_csv(
    roc_pr_path,
    index=False
)


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

def _to_jsonable(obj):
    """Recursively convert NumPy/array values to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


metrics = {
    "step": 43,
    "evaluation": "frozen_post_selection_patient_wise_3fold_OOF",

    "dataset": {
        "path": str(DATA_PATH),
        "samples": int(len(df)),
        "features_total": int(len(df.columns)),
        "features_used": int(len(FEATURES)),
        "patients": int(df["patient_id"].nunique()),
        "folds": unique_folds,
    },

    "features": FEATURES,

    "model": {
        "type": "HistGradientBoostingClassifier",
        "parameters": HGB_PARAMS,
        "class_weights": CLASS_WEIGHTS,
    },

    "metrics": {
        "accuracy": float(overall_accuracy),
        "balanced_accuracy": float(
            overall_balanced_accuracy
        ),
        "macro_precision": float(
            overall_macro_precision
        ),
        "macro_recall": float(
            overall_macro_recall
        ),
        "macro_f1": float(overall_macro_f1),
        "weighted_f1": float(
            overall_weighted_f1
        ),
    },

    "classification_report": report,

    "confusion_matrix": cm.tolist(),

    "classes": CLASSES,

    "fold_metrics": fold_results,

    "outputs": {
        "predictions": str(prediction_path),
        "confusion_matrix": str(cm_path),
        "classification_report": str(report_path),
        "fold_metrics": str(fold_path),
        "roc_pr_data": str(roc_pr_path),
    },

    "methodological_note": (
        "This is a frozen post-selection OOF evaluation. "
        "The Step 42 hyperparameters were selected using OOF "
        "results, therefore this result should not be interpreted "
        "as an independently unbiased test estimate."
    ),
}

json_path = (
    RESULTS_DIR /
    "step43_final_patient_wise_metrics.json"
)

json_payload = _to_jsonable(metrics)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        json_payload,
        f,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

elapsed = time.time() - start_time

print("\n" + "=" * 80)
print("STEP 43 COMPLETE")
print("=" * 80)

print(f"\nAccuracy          : {overall_accuracy * 100:.2f}%")
print(
    f"Balanced Accuracy : "
    f"{overall_balanced_accuracy * 100:.2f}%"
)
print(
    f"Macro Precision   : "
    f"{overall_macro_precision * 100:.2f}%"
)
print(
    f"Macro Recall      : "
    f"{overall_macro_recall * 100:.2f}%"
)
print(
    f"Macro F1          : "
    f"{overall_macro_f1 * 100:.2f}%"
)
print(
    f"Weighted F1       : "
    f"{overall_weighted_f1 * 100:.2f}%"
)

print("\nPer-class metrics:")

for cls in CLASSES:
    print(
        f"{cls}: "
        f"Precision={report[cls]['precision']:.4f}, "
        f"Recall={report[cls]['recall']:.4f}, "
        f"F1={report[cls]['f1-score']:.4f}"
    )

print("\nFiles saved:")

print(prediction_path)
print(cm_path)
print(report_path)
print(fold_path)
print(roc_pr_path)
print(json_path)

print(f"\nExecution time: {elapsed / 60:.2f} minutes")
print("\nStep 43 finished successfully.")