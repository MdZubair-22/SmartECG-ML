"""
STEP 39 - FEATURE ABLATION EXPERIMENT

Compare different feature subsets using the proper
patient-wise 3-fold OOF setup.

Model:
HistGradientBoostingClassifier

Class weights:
N=1, S=3, V=1, F=5, Q=1.5

IMPORTANT:
Each test fold is trained using the OTHER TWO folds.
No test-fold feature selection is performed.
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
    classification_report
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
# LOAD
# ============================================================

print("=" * 80)
print("STEP 39 - FEATURE ABLATION")
print("=" * 80)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(
    f"\nDataset shape: {df.shape}"
)

all_features = [
    c for c in df.columns
    if c not in META_COLUMNS
]

X_all = df[all_features]
y = df["class"]

folds = df["fold"].astype(int).values


# ============================================================
# FEATURE GROUPS
# ============================================================

basic_features = [
    "min",
    "max",
    "peak_to_peak",
    "absolute_area",
    "skewness",
    "kurtosis",
    "zero_crossings",
    "max_abs"
]

rhythm_features = [
    "prev_rr",
    "prev_hr"
]

spectral_features = [
    "energy_ratio",
    "dominant_frequency",
    "spectral_entropy"
]

morphology_features = [
    c for c in all_features
    if c.startswith("morph_")
]


# Important features identified by Step 38
top_features = [
    "prev_rr",
    "spectral_entropy",
    "max",
    "prev_hr",
    "morph_energy_asymmetry",
    "morph_pre30_energy_ratio",
    "morph_zero_crossings",
    "morph_min",
    "morph_max_negative_slope",
    "morph_slope30",
    "energy_ratio",
    "morph_pre30_mean",
    "skewness",
    "morph_mean_abs_slope",
    "morph_pre10_mean"
]


# Strong morphology features for S/F
sf_morphology = [
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
    "morph_mean"
]


# ============================================================
# HIGH-CORRELATION REDUCED SET
# ============================================================

# Remove highly redundant morphology variables while keeping
# representative features.

reduced_features = [
    "min",
    "max",
    "peak_to_peak",
    "absolute_area",
    "skewness",
    "kurtosis",
    "zero_crossings",
    "max_abs",
    "energy_ratio",
    "dominant_frequency",
    "spectral_entropy",
    "prev_rr",
    "prev_hr",

    "morph_annotation_amplitude",
    "morph_min",
    "morph_max",
    "morph_mean",
    "morph_std",
    "morph_slope10",
    "morph_slope30",
    "morph_max_positive_slope",
    "morph_max_negative_slope",
    "morph_pre10_energy_ratio",
    "morph_pre30_energy_ratio",
    "morph_zero_crossings",
    "morph_distance_to_peak",
    "morph_energy_asymmetry"
]


# ============================================================
# DEFINE EXPERIMENTS
# ============================================================

FEATURE_SETS = {

    "all_38": all_features,

    "top_15": top_features,

    "rhythm_spectral": (
        rhythm_features
        + spectral_features
    ),

    "rhythm_spectral_basic": (
        rhythm_features
        + spectral_features
        + basic_features
    ),

    "sf_morphology": sf_morphology,

    "rhythm_spectral_sf_morphology": (
        rhythm_features
        + spectral_features
        + sf_morphology
    ),

    "reduced_27": reduced_features
}


# Remove accidental duplicates

for name in FEATURE_SETS:

    FEATURE_SETS[name] = list(
        dict.fromkeys(
            FEATURE_SETS[name]
        )
    )


print("\nFeature sets:")

for name, features in FEATURE_SETS.items():

    print(
        f"  {name}: "
        f"{len(features)} features"
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


def sample_weights(labels):

    return np.array(
        [
            CLASS_WEIGHTS[str(x)]
            for x in labels
        ],
        dtype=float
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

all_results = []

all_predictions = {}

unique_folds = sorted(
    df["fold"].unique()
)


for set_name, feature_list in FEATURE_SETS.items():

    print("\n")
    print("=" * 80)
    print(
        f"FEATURE SET: {set_name.upper()}"
    )
    print("=" * 80)

    print(
        f"Number of features: "
        f"{len(feature_list)}"
    )

    X = X_all[feature_list]

    oof_pred = np.empty(
        len(df),
        dtype=object
    )

    fold_results = []

    start_total = time.time()

    # --------------------------------------------------------
    # Proper 3-fold patient-wise OOF
    # --------------------------------------------------------

    for test_fold in unique_folds:

        train_mask = folds != test_fold
        test_mask = folds == test_fold

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        print(
            f"\nFold {test_fold}: "
            f"train={len(X_train):,}, "
            f"test={len(X_test):,}"
        )

        start = time.time()

        model = create_model()

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights(
                y_train
            )
        )

        pred = model.predict(
            X_test
        )

        elapsed = time.time() - start

        oof_pred[test_mask] = pred

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

        print(
            f"  Time: {elapsed:.1f}s"
        )

        print(
            f"  Accuracy: {acc:.4f}"
        )

        print(
            f"  Balanced Accuracy: {bal:.4f}"
        )

        print(
            f"  Macro F1: {macro:.4f}"
        )

        fold_results.append({
            "fold": int(test_fold),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
            "macro_f1": float(macro)
        })

    # --------------------------------------------------------
    # FINAL OOF
    # --------------------------------------------------------

    acc = accuracy_score(
        y,
        oof_pred
    )

    bal = balanced_accuracy_score(
        y,
        oof_pred
    )

    macro = f1_score(
        y,
        oof_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )

    weighted = f1_score(
        y,
        oof_pred,
        labels=CLASS_ORDER,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y,
        oof_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0
    )

    recalls = {
        cls: float(
            report[cls]["recall"]
        )
        for cls in CLASS_ORDER
    }

    print("\nFINAL OOF:")

    print(
        f"  Accuracy:          {acc:.4f}"
    )

    print(
        f"  Balanced Accuracy: {bal:.4f}"
    )

    print(
        f"  Macro F1:          {macro:.4f}"
    )

    print(
        f"  Weighted F1:       {weighted:.4f}"
    )

    print("\nRecall:")

    for cls in CLASS_ORDER:

        print(
            f"  {cls}: "
            f"{recalls[cls]:.4f}"
        )

    all_predictions[
        set_name
    ] = oof_pred

    all_results.append({
        "feature_set": set_name,
        "n_features": len(feature_list),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "N_recall": recalls["N"],
        "S_recall": recalls["S"],
        "V_recall": recalls["V"],
        "F_recall": recalls["F"],
        "Q_recall": recalls["Q"],
        "fold_results": fold_results,
        "training_time_seconds":
            float(time.time() - start_total)
    })


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("STEP 39 FINAL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(
    all_results
)

columns = [
    "feature_set",
    "n_features",
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
    results_df[columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results_path = (
    RESULTS_DIR
    / "step39_feature_ablation_metrics.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        {
            "class_weights": CLASS_WEIGHTS,
            "feature_sets": FEATURE_SETS,
            "results": all_results
        },
        f,
        indent=2
    )


pred_output = df[
    [
        "record",
        "patient_id",
        "sample",
        "class",
        "fold"
    ]
].copy()

for name, pred in all_predictions.items():

    pred_output[
        f"prediction_{name}"
    ] = pred


pred_path = (
    RESULTS_DIR
    / "step39_feature_ablation_predictions.csv"
)

pred_output.to_csv(
    pred_path,
    index=False,
    chunksize=10000
)


print("\n")
print("=" * 80)
print("STEP 39 COMPLETED")
print("=" * 80)

print("\nSaved:")
print(results_path)
print(pred_path)

print("\nReference:")
print(
    "Step 36 Balanced HGB:"
)

print(
    "Accuracy = 85.17%"
)

print(
    "Balanced Accuracy = 57.18%"
)

print(
    "Macro F1 = 52.79%"
)

print(
    "\nIMPORTANT:"
)

print(
    "Do not select a feature set yet."
)

print(
    "Send the complete output for all feature sets."
)