"""
STEP 36 - CLASS-BALANCED HISTGRADIENTBOOSTING

Goal:
Improve minority-class recall (especially S and F)
while maintaining V and Q performance.

Evaluation:
Patient-wise 3-fold OOF

Dataset:
data/processed/enhanced_causal_ecg_features.csv
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
print("STEP 36 - CLASS-BALANCED HISTGRADIENTBOOSTING")
print("=" * 80)

print("\nLoading:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"\nDataset shape: {df.shape}")


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

print(f"\nFeatures: {len(feature_columns)}")

print("\nClasses:")
print(
    y.value_counts()
    .reindex(CLASS_ORDER)
)


# ============================================================
# MODEL WEIGHT STRATEGIES
# ============================================================

# We deliberately use MODERATE weights.
#
# Very large weights can cause the model to over-predict
# minority classes and damage N/V/Q performance.

WEIGHT_STRATEGIES = {

    "baseline": {
        "N": 1.0,
        "S": 1.0,
        "V": 1.0,
        "F": 1.0,
        "Q": 1.0
    },

    "moderate": {
        "N": 1.0,
        "S": 2.0,
        "V": 1.0,
        "F": 3.0,
        "Q": 1.2
    },

    "balanced": {
        "N": 1.0,
        "S": 3.0,
        "V": 1.0,
        "F": 5.0,
        "Q": 1.5
    },

    "strong_minority": {
        "N": 1.0,
        "S": 4.0,
        "V": 1.0,
        "F": 7.0,
        "Q": 1.5
    }
}


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

def make_sample_weights(labels, class_weights):

    return np.array(
        [
            class_weights[str(label)]
            for label in labels
        ],
        dtype=np.float64
    )


# ============================================================
# STORAGE
# ============================================================

all_results = []

all_predictions = {}

unique_folds = sorted(df["fold"].unique())


# ============================================================
# RUN EACH WEIGHT STRATEGY
# ============================================================

for strategy_name, class_weights in WEIGHT_STRATEGIES.items():

    print("\n")
    print("=" * 80)
    print(f"WEIGHT STRATEGY: {strategy_name.upper()}")
    print("=" * 80)

    print("Weights:")
    print(class_weights)

    oof_predictions = np.empty(
        len(df),
        dtype=object
    )

    fold_results = []

    # --------------------------------------------------------
    # 3-FOLD PATIENT-WISE OOF
    # --------------------------------------------------------

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

        print(
            f"Train samples: {len(X_train):,}"
        )

        print(
            f"Test samples:  {len(X_test):,}"
        )

        # Create sample weights
        sample_weights = make_sample_weights(
            y_train,
            class_weights
        )

        print(
            "\nTraining class counts:"
        )

        print(
            y_train.value_counts()
            .reindex(CLASS_ORDER)
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        start = time.time()

        model = create_model()

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )

        elapsed = time.time() - start

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        pred = model.predict(X_test)

        oof_predictions[test_mask] = pred

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        acc = accuracy_score(
            y_test,
            pred
        )

        bal = balanced_accuracy_score(
            y_test,
            pred
        )

        macro = f1_score(
            y_test,
            pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0
        )

        weighted = f1_score(
            y_test,
            pred,
            labels=CLASS_ORDER,
            average="weighted",
            zero_division=0
        )

        print(
            f"\nTraining time: {elapsed:.1f} sec"
        )

        print(
            f"Accuracy:          {acc:.4f}"
        )

        print(
            f"Balanced Accuracy: {bal:.4f}"
        )

        print(
            f"Macro F1:          {macro:.4f}"
        )

        print(
            f"Weighted F1:       {weighted:.4f}"
        )

        report = classification_report(
            y_test,
            pred,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0
        )

        print("\nRecall:")

        for cls in CLASS_ORDER:
            print(
                f"  {cls}: "
                f"{report[cls]['recall']:.4f}"
            )

        fold_results.append({
            "fold": int(test_fold),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
            "macro_f1": float(macro),
            "weighted_f1": float(weighted)
        })

    # ========================================================
    # FINAL OOF
    # ========================================================

    print("\n")
    print("-" * 80)
    print(f"FINAL OOF - {strategy_name.upper()}")
    print("-" * 80)

    acc = accuracy_score(
        y,
        oof_predictions
    )

    bal = balanced_accuracy_score(
        y,
        oof_predictions
    )

    macro = f1_score(
        y,
        oof_predictions,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )

    weighted = f1_score(
        y,
        oof_predictions,
        labels=CLASS_ORDER,
        average="weighted",
        zero_division=0
    )

    print(
        f"\nAccuracy:          {acc:.4f}"
    )

    print(
        f"Balanced Accuracy: {bal:.4f}"
    )

    print(
        f"Macro F1:          {macro:.4f}"
    )

    print(
        f"Weighted F1:       {weighted:.4f}"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = classification_report(
        y,
        oof_predictions,
        labels=CLASS_ORDER,
        zero_division=0,
        digits=4
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y,
        oof_predictions,
        labels=CLASS_ORDER
    )

    print("Confusion Matrix:")

    print(
        pd.DataFrame(
            cm,
            index=CLASS_ORDER,
            columns=CLASS_ORDER
        )
    )

    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    report_dict = classification_report(
        y,
        oof_predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0
    )

    recalls = {
        cls: float(report_dict[cls]["recall"])
        for cls in CLASS_ORDER
    }

    print("\nFinal Recall:")

    for cls in CLASS_ORDER:
        print(
            f"  {cls}: {recalls[cls]:.4f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    all_predictions[strategy_name] = oof_predictions

    all_results.append({
        "strategy": strategy_name,
        "N_weight": class_weights["N"],
        "S_weight": class_weights["S"],
        "V_weight": class_weights["V"],
        "F_weight": class_weights["F"],
        "Q_weight": class_weights["Q"],
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "N_recall": recalls["N"],
        "S_recall": recalls["S"],
        "V_recall": recalls["V"],
        "F_recall": recalls["F"],
        "Q_recall": recalls["Q"],
        "fold_results": fold_results
    })


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("STEP 36 FINAL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(all_results)

display_columns = [
    "strategy",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "N_recall",
    "S_recall",
    "V_recall",
    "F_recall",
    "Q_recall"
]

print(
    results_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_output = df[
    [
        "record",
        "patient_id",
        "sample",
        "class",
        "fold"
    ]
].copy()

for strategy_name, pred in all_predictions.items():

    prediction_output[
        f"prediction_{strategy_name}"
    ] = pred


prediction_path = (
    RESULTS_DIR
    / "step36_balanced_gb_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False,
    chunksize=10000
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = (
    RESULTS_DIR
    / "step36_balanced_gb_metrics.json"
)

with open(metrics_path, "w") as f:

    json.dump(
        {
            "dataset": str(DATA_PATH),
            "n_samples": int(len(df)),
            "n_features": int(len(feature_columns)),
            "classes": CLASS_ORDER,
            "strategies": all_results
        },
        f,
        indent=2
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 80)
print("STEP 36 COMPLETED")
print("=" * 80)

print("\nSaved:")
print(prediction_path)
print(metrics_path)

print("\nReference from Step 35:")
print("Accuracy:           0.8624")
print("Balanced Accuracy:  0.5636")
print("Macro F1:            0.5248")
print("Weighted F1:         0.8698")

print("\nCompare especially:")
print("  S recall")
print("  F recall")
print("  Balanced Accuracy")
print("  Macro F1")
print("  Q recall")