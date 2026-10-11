"""
STEP 37 - PATIENT-WISE HGB THRESHOLD OPTIMIZATION

Model:
HistGradientBoostingClassifier

Class weights:
N = 1.0
S = 3.0
V = 1.0
F = 5.0
Q = 1.5

Evaluation:
Patient-wise 3-fold OOF

For each test fold:
    Train on training folds
    Select thresholds on validation fold
    Apply thresholds ONCE to untouched test fold

This prevents threshold-selection leakage.
"""

from pathlib import Path
import json
import time
from itertools import product

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "enhanced_causal_ecg_features.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

CLASS_ORDER = ["N", "S", "V", "F", "Q"]
CLASS_TO_INT = {label: idx for idx, label in enumerate(CLASS_ORDER)}
INT_TO_CLASS = {idx: label for label, idx in CLASS_TO_INT.items()}

CLASS_WEIGHTS = {
    "N": 1.0,
    "S": 3.0,
    "V": 1.0,
    "F": 5.0,
    "Q": 1.5
}

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
print("STEP 37 - HGB PATIENT-WISE THRESHOLD OPTIMIZATION")
print("=" * 80)

print("\nLoading:")
print(DATA_PATH)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(f"\nDataset shape: {df.shape}")

patient_fold_counts = df.groupby("patient_id")["fold"].nunique()
leakage = patient_fold_counts[patient_fold_counts > 1]
if len(leakage) > 0:
    raise ValueError(f"Patient leakage detected across folds: {leakage.to_dict()}")
print("\nPatient leakage check: ✓ no patient appears in multiple folds")


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    c for c in df.columns
    if c not in META_COLUMNS
]

X = df[feature_columns].copy()
y = df["class"].copy()

folds = df["fold"].astype(int).values

print(f"Features: {len(feature_columns)}")

print("\nClass distribution:")
print(
    y.value_counts()
    .reindex(CLASS_ORDER)
)


# ============================================================
# MODEL
# ============================================================

def create_model():

    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

def make_sample_weights(labels):

    return np.array(
        [
            CLASS_WEIGHTS[str(label)]
            for label in labels
        ],
        dtype=np.float64
    )


def reorder_probabilities(probabilities, model_classes):
    """Reorder model.predict_proba columns to the canonical ECG class order."""
    if list(model_classes) == CLASS_ORDER:
        return np.asarray(probabilities, dtype=np.float64)

    ordered = np.zeros_like(np.asarray(probabilities, dtype=np.float64), dtype=np.float64)
    for target_idx, cls in enumerate(CLASS_ORDER):
        source_idx = list(model_classes).index(cls)
        ordered[:, target_idx] = probabilities[:, source_idx]
    return ordered


def compute_confusion_metrics(y_true, y_pred, n_classes=5):
    """Compute balanced accuracy and macro F1 from confusion matrix without slow per-call report generation."""
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(conf, (y_true, y_pred), 1)

    row_sums = conf.sum(axis=1)
    col_sums = conf.sum(axis=0)
    diag = np.diag(conf)

    tp = diag.astype(np.float64)
    fp = col_sums.astype(np.float64) - tp
    fn = row_sums.astype(np.float64) - tp

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fp) > 0
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fn) > 0
    )

    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(precision + recall) > 0
    )

    balanced_accuracy = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    return {
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "confusion_matrix": conf
    }


def atomic_json_write(path, payload):
    temp_path = Path(str(path) + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if path.exists():
        path.unlink()
    temp_path.replace(path)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def apply_thresholds(probabilities, thresholds):

    adjusted = probabilities.copy()

    for i, cls in enumerate(CLASS_ORDER):

        adjusted[:, i] = (
            probabilities[:, i]
            / thresholds[cls]
        )

    indices = np.argmax(
        adjusted,
        axis=1
    )

    return np.array(
        [
            CLASS_ORDER[i]
            for i in indices
        ],
        dtype=object
    )


def search_thresholds(
    probabilities,
    true_labels
):
    """Fast threshold search using vectorized metric computation.

    We still enforce N and V recall constraints to prevent threshold tuning from
    damaging the dominant classes while optimizing the macro-F1 / balanced-accuracy
    objective.
    """

    threshold_ranges = {
        "N": np.arange(0.90, 1.21, 0.05, dtype=np.float64),
        "S": np.arange(0.50, 1.01, 0.05, dtype=np.float64),
        "V": np.arange(0.90, 1.11, 0.05, dtype=np.float64),
        "F": np.arange(0.30, 1.01, 0.05, dtype=np.float64),
        "Q": np.arange(0.60, 1.01, 0.05, dtype=np.float64),
    }

    labels = np.asarray(true_labels, dtype=object)
    true_codes = np.array([CLASS_TO_INT[str(label)] for label in labels], dtype=np.int8)
    probability_array = np.asarray(probabilities, dtype=np.float64)

    baseline_pred_idx = np.argmax(probability_array, axis=1)
    baseline_pred = np.array([CLASS_ORDER[i] for i in baseline_pred_idx], dtype=object)
    baseline_report = classification_report(
        labels,
        baseline_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0
    )
    baseline_n_recall = baseline_report["N"]["recall"]
    baseline_v_recall = baseline_report["V"]["recall"]

    min_n_recall = max(0.80, baseline_n_recall - 0.08)
    min_v_recall = max(0.75, baseline_v_recall - 0.08)

    print("\nThreshold protection:")
    print(f"Minimum N recall: {min_n_recall:.4f}")
    print(f"Minimum V recall: {min_v_recall:.4f}")

    best = None
    total_tested = 0
    valid_tested = 0

    class_ids = np.array([CLASS_TO_INT[c] for c in CLASS_ORDER], dtype=np.int8)
    threshold_product = product(
        threshold_ranges["N"],
        threshold_ranges["S"],
        threshold_ranges["V"],
        threshold_ranges["F"],
        threshold_ranges["Q"],
    )

    total_combinations = int(
        len(threshold_ranges["N"]) * len(threshold_ranges["S"]) *
        len(threshold_ranges["V"]) * len(threshold_ranges["F"]) *
        len(threshold_ranges["Q"])
    )

    for n_t, s_t, v_t, f_t, q_t in threshold_product:
        total_tested += 1

        thresholds = {
            "N": float(n_t),
            "S": float(s_t),
            "V": float(v_t),
            "F": float(f_t),
            "Q": float(q_t),
        }
        threshold_vector = np.array([
            thresholds["N"],
            thresholds["S"],
            thresholds["V"],
            thresholds["F"],
            thresholds["Q"],
        ], dtype=np.float64)

        adjusted = probability_array / threshold_vector
        pred_idx = adjusted.argmax(axis=1)

        conf_metrics = compute_confusion_metrics(true_codes, pred_idx, n_classes=len(CLASS_ORDER))
        n_recall = conf_metrics["recall"][CLASS_TO_INT["N"]]
        v_recall = conf_metrics["recall"][CLASS_TO_INT["V"]]

        if n_recall < min_n_recall or v_recall < min_v_recall:
            if total_tested % 1000 == 0:
                print(
                    f"Progress: {total_tested:,}/{total_combinations:,} "
                    f"({100.0 * total_tested / total_combinations:.1f}%)"
                )
            continue

        valid_tested += 1
        balanced = conf_metrics["balanced_accuracy"]
        macro = conf_metrics["macro_f1"]
        objective = 0.60 * macro + 0.40 * balanced

        if best is None or objective > best["objective"]:
            best = {
                "thresholds": thresholds.copy(),
                "objective": float(objective),
                "balanced_accuracy": float(balanced),
                "macro_f1": float(macro),
                "n_recall": float(n_recall),
                "v_recall": float(v_recall),
            }

        if total_tested % 1000 == 0:
            print(
                f"Progress: {total_tested:,}/{total_combinations:,} "
                f"({100.0 * total_tested / total_combinations:.1f}%) | "
                f"Valid: {valid_tested:,} | Best objective: {best['objective']:.5f}"
            )

    print(f"\nThreshold combinations tested: {total_tested:,}")
    print(f"Valid combinations: {valid_tested:,}")

    if best is None:
        print("\nNo valid threshold combination found.")
        fallback = {cls: 1.0 for cls in CLASS_ORDER}
        return fallback

    print("\nBest validation thresholds:")
    for cls in CLASS_ORDER:
        print(f"  {cls}: {best['thresholds'][cls]:.3f}")
    print(f"\nValidation Balanced Accuracy: {best['balanced_accuracy']:.4f}")
    print(f"Validation Macro F1: {best['macro_f1']:.4f}")
    print(f"Validation Objective: {best['objective']:.4f}")

    return best["thresholds"]


# ============================================================
# STORAGE
# ============================================================

baseline_oof = np.empty(
    len(df),
    dtype=object
)

optimized_oof = np.empty(
    len(df),
    dtype=object
)

fold_results = []

fold_thresholds = {}

unique_folds = sorted(
    df["fold"].unique()
)


# ============================================================
# PATIENT-WISE EVALUATION
# ============================================================

for test_fold in unique_folds:

    print("\n")
    print("=" * 80)
    print(f"TEST FOLD {test_fold}")
    print("=" * 80)

    # --------------------------------------------------------
    # Choose validation fold
    # --------------------------------------------------------

    other_folds = [
        f for f in unique_folds
        if f != test_fold
    ]

    # For 3 folds:
    #
    # test 0 -> validation 1, train 2
    # test 1 -> validation 0, train 2
    # test 2 -> validation 0, train 1

    if test_fold == 0:

        validation_fold = 1
        train_folds = [2]

    elif test_fold == 1:

        validation_fold = 0
        train_folds = [2]

    else:

        validation_fold = 0
        train_folds = [1]

    train_mask = np.isin(
        folds,
        train_folds
    )

    validation_mask = (
        folds == validation_fold
    )

    test_mask = (
        folds == test_fold
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print(
        f"Training folds: {train_folds}"
    )

    print(
        f"Validation fold: {validation_fold}"
    )

    print(
        f"Test fold: {test_fold}"
    )

    print(
        f"\nTrain samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation samples: "
        f"{len(X_validation):,}"
    )

    print(
        f"Test samples: "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\nTraining HGB...")

    start = time.time()

    model = create_model()

    sample_weights = make_sample_weights(
        y_train
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    print(
        f"Training time: "
        f"{time.time() - start:.1f} sec"
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating validation probabilities..."
    )

    validation_prob = reorder_probabilities(
        model.predict_proba(X_validation),
        model.classes_
    )

    validation_baseline_pred = model.predict(
        X_validation
    )

    # --------------------------------------------------------
    # Search thresholds ONLY on validation
    # --------------------------------------------------------

    print(
        "\nSearching thresholds on validation fold..."
    )

    thresholds = search_thresholds(
        validation_prob,
        y_validation
    )

    fold_thresholds[str(test_fold)] = thresholds

    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating untouched test predictions..."
    )

    test_prob = reorder_probabilities(
        model.predict_proba(X_test),
        model.classes_
    )

    test_baseline_pred = model.predict(
        X_test
    )

    test_optimized_pred = apply_thresholds(
        test_prob,
        thresholds
    )

    # Store OOF
    baseline_oof[test_mask] = test_baseline_pred
    optimized_oof[test_mask] = test_optimized_pred

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    baseline_acc = accuracy_score(
        y_test,
        test_baseline_pred
    )

    baseline_bal = balanced_accuracy_score(
        y_test,
        test_baseline_pred
    )

    baseline_macro = f1_score(
        y_test,
        test_baseline_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )

    optimized_acc = accuracy_score(
        y_test,
        test_optimized_pred
    )

    optimized_bal = balanced_accuracy_score(
        y_test,
        test_optimized_pred
    )

    optimized_macro = f1_score(
        y_test,
        test_optimized_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )

    print("\nTEST RESULTS")

    print("\nBaseline:")
    print(
        f"  Accuracy:          {baseline_acc:.4f}"
    )
    print(
        f"  Balanced Accuracy: {baseline_bal:.4f}"
    )
    print(
        f"  Macro F1:          {baseline_macro:.4f}"
    )

    print("\nOptimized:")
    print(
        f"  Accuracy:          {optimized_acc:.4f}"
    )
    print(
        f"  Balanced Accuracy: {optimized_bal:.4f}"
    )
    print(
        f"  Macro F1:          {optimized_macro:.4f}"
    )

    fold_results.append({
        "test_fold": int(test_fold),
        "validation_fold": int(validation_fold),
        "train_folds": train_folds,
        "baseline_accuracy": float(baseline_acc),
        "baseline_balanced_accuracy": float(baseline_bal),
        "baseline_macro_f1": float(baseline_macro),
        "optimized_accuracy": float(optimized_acc),
        "optimized_balanced_accuracy": float(optimized_bal),
        "optimized_macro_f1": float(optimized_macro),
        "thresholds": thresholds
    })


# ============================================================
# FINAL OOF RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("FINAL PATIENT-WISE OOF RESULTS")
print("=" * 80)


# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

baseline_acc = accuracy_score(
    y,
    baseline_oof
)

baseline_bal = balanced_accuracy_score(
    y,
    baseline_oof
)

baseline_macro = f1_score(
    y,
    baseline_oof,
    labels=CLASS_ORDER,
    average="macro",
    zero_division=0
)

baseline_weighted = f1_score(
    y,
    baseline_oof,
    labels=CLASS_ORDER,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# Optimized
# ------------------------------------------------------------

optimized_acc = accuracy_score(
    y,
    optimized_oof
)

optimized_bal = balanced_accuracy_score(
    y,
    optimized_oof
)

optimized_macro = f1_score(
    y,
    optimized_oof,
    labels=CLASS_ORDER,
    average="macro",
    zero_division=0
)

optimized_weighted = f1_score(
    y,
    optimized_oof,
    labels=CLASS_ORDER,
    average="weighted",
    zero_division=0
)


print("\nBASELINE HGB")

print(
    f"Accuracy:          {baseline_acc:.4f}"
)

print(
    f"Balanced Accuracy: {baseline_bal:.4f}"
)

print(
    f"Macro F1:          {baseline_macro:.4f}"
)

print(
    f"Weighted F1:       {baseline_weighted:.4f}"
)


print("\nTHRESHOLD-OPTIMIZED HGB")

print(
    f"Accuracy:          {optimized_acc:.4f}"
)

print(
    f"Balanced Accuracy: {optimized_bal:.4f}"
)

print(
    f"Macro F1:          {optimized_macro:.4f}"
)

print(
    f"Weighted F1:       {optimized_weighted:.4f}"
)


# ============================================================
# CLASS REPORTS
# ============================================================

print("\n")
print("=" * 80)
print("CLASS RECALL COMPARISON")
print("=" * 80)

baseline_report = classification_report(
    y,
    baseline_oof,
    labels=CLASS_ORDER,
    output_dict=True,
    zero_division=0
)

optimized_report = classification_report(
    y,
    optimized_oof,
    labels=CLASS_ORDER,
    output_dict=True,
    zero_division=0
)

recall_table = []

for cls in CLASS_ORDER:

    recall_table.append({
        "class": cls,
        "baseline_recall":
            baseline_report[cls]["recall"],
        "optimized_recall":
            optimized_report[cls]["recall"],
        "change":
            (
                optimized_report[cls]["recall"]
                - baseline_report[cls]["recall"]
            )
    })

recall_df = pd.DataFrame(
    recall_table
)

print(
    recall_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\nBASELINE CONFUSION MATRIX")

baseline_cm = confusion_matrix(
    y,
    baseline_oof,
    labels=CLASS_ORDER
)

print(
    pd.DataFrame(
        baseline_cm,
        index=CLASS_ORDER,
        columns=CLASS_ORDER
    )
)


print("\nOPTIMIZED CONFUSION MATRIX")

optimized_cm = confusion_matrix(
    y,
    optimized_oof,
    labels=CLASS_ORDER
)

print(
    pd.DataFrame(
        optimized_cm,
        index=CLASS_ORDER,
        columns=CLASS_ORDER
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output = df[
    [
        "record",
        "patient_id",
        "sample",
        "class",
        "fold"
    ]
].copy()

output[
    "hgb_baseline_prediction"
] = baseline_oof

output[
    "hgb_threshold_prediction"
] = optimized_oof


prediction_path = (
    RESULTS_DIR
    / "step37_hgb_threshold_predictions.csv"
)

output.to_csv(
    prediction_path,
    index=False,
    chunksize=10000
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model": "HistGradientBoostingClassifier",

    "class_weights": CLASS_WEIGHTS,

    "dataset": str(DATA_PATH),

    "n_samples": int(len(df)),

    "n_features": int(len(feature_columns)),

    "features": feature_columns,

    "classes": CLASS_ORDER,

    "baseline": {
        "accuracy": float(baseline_acc),
        "balanced_accuracy": float(baseline_bal),
        "macro_f1": float(baseline_macro),
        "weighted_f1": float(baseline_weighted)
    },

    "optimized": {
        "accuracy": float(optimized_acc),
        "balanced_accuracy": float(optimized_bal),
        "macro_f1": float(optimized_macro),
        "weighted_f1": float(optimized_weighted)
    },

    "recall_comparison":
        recall_df.to_dict(
            orient="records"
        ),

    "fold_results":
        fold_results,

    "fold_thresholds":
        fold_thresholds
}


metrics_path = (
    RESULTS_DIR
    / "step37_hgb_threshold_metrics.json"
)

atomic_json_write(
    metrics_path,
    metrics
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 80)
print("STEP 37 COMPLETED")
print("=" * 80)

print("\nSaved:")

print(prediction_path)

print(metrics_path)

print("\nReference:")
print(
    "Step 35 HGB: "
    "86.24% Accuracy / "
    "56.36% Balanced Accuracy / "
    "52.48% Macro F1"
)

print(
    "Step 36 Balanced HGB: "
    "85.17% Accuracy / "
    "57.18% Balanced Accuracy / "
    "52.79% Macro F1"
)

print(
    "\nImportant: threshold selection was performed "
    "only on validation folds."
)