"""
STEP 41 - STABLE FEATURE SUBSET EXPERIMENT
===========================================

Purpose:
- Test whether features repeatedly selected across the 3 folds
  provide a compact and robust ECG classification feature set.
- Compare:
    1. ALL 38 FEATURES
    2. FIXED STEP-39 23 FEATURES
    3. STABLE 16 FEATURES (selected in 3/3 folds)
    4. STABLE 21 FEATURES (selected in >=2/3 folds)

Evaluation:
- Patient-wise 3-fold OOF
- Train on 2 folds, test on 1 fold
- No patient leakage
- Same HGB model and class weighting as Steps 36-40

Outputs:
    results/step41_stable_feature_subset_metrics.json
    results/step41_stable_feature_subset_predictions.csv
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "enhanced_causal_ecg_features.csv",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET = "class"
FOLD_COLUMN = "fold"

CLASSES = [
    "N",
    "S",
    "V",
    "F",
    "Q",
]


# Same class weights as Step 36 / Step 39 / Step 40
CLASS_WEIGHTS = {
    "N": 1.0,
    "S": 3.0,
    "V": 1.0,
    "F": 5.0,
    "Q": 1.5,
}


# ============================================================
# STEP 39 — FIXED 23 FEATURES
# ============================================================

STEP39_23 = [
    "prev_rr",
    "prev_hr",

    "spectral_entropy",
    "dominant_frequency",
    "energy_ratio",

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


# ============================================================
# STEP 40 — STABLE FEATURES
# ============================================================

# Selected in ALL 3 folds
STABLE_16 = [
    "absolute_area",
    "dominant_frequency",
    "energy_ratio",
    "min",
    "morph_energy_asymmetry",
    "morph_max_negative_slope",
    "morph_max_positive_slope",
    "morph_mean_abs_slope",
    "morph_min",
    "morph_pre30_energy_ratio",
    "morph_zero_crossings",
    "peak_to_peak",
    "prev_hr",
    "prev_rr",
    "skewness",
    "spectral_entropy",
]


# Selected in at least 2/3 folds
STABLE_21 = [
    "absolute_area",
    "dominant_frequency",
    "energy_ratio",
    "min",
    "morph_energy_asymmetry",
    "morph_max_negative_slope",
    "morph_max_positive_slope",
    "morph_mean_abs_slope",
    "morph_min",
    "morph_pre30_energy_ratio",
    "morph_zero_crossings",
    "peak_to_peak",
    "prev_hr",
    "prev_rr",
    "skewness",
    "spectral_entropy",

    # 2/3 fold features
    "max",
    "morph_annotation_pre10_diff",
    "morph_mean",
    "morph_pre10_energy_ratio",
    "morph_pre30_mean",
]


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
        random_state=RANDOM_STATE,
    )


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

def make_sample_weights(y):

    return np.array(
        [
            CLASS_WEIGHTS[label]
            for label in y
        ],
        dtype=np.float64,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASSES,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def calculate_recall(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES,
    )

    recalls = {}

    for i, cls in enumerate(CLASSES):

        total = cm[i].sum()

        if total == 0:
            recalls[cls] = 0.0
        else:
            recalls[cls] = float(
                cm[i, i] / total
            )

    return recalls


# ============================================================
# EVALUATE ONE FEATURE SET
# ============================================================

def evaluate_feature_set(
    df,
    feature_set_name,
    features,
):

    print("\n")
    print("=" * 80)
    print(
        f"FEATURE SET: "
        f"{feature_set_name.upper()}"
    )
    print("=" * 80)

    print(
        f"Number of features: "
        f"{len(features)}"
    )

    predictions = []

    fold_results = []

    for test_fold in sorted(
        df[FOLD_COLUMN].unique()
    ):

        train_mask = (
            df[FOLD_COLUMN] != test_fold
        )

        test_mask = (
            df[FOLD_COLUMN] == test_fold
        )

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        X_train = df.loc[
            train_idx,
            features,
        ]

        y_train = df.loc[
            train_idx,
            TARGET,
        ].astype(str)

        X_test = df.loc[
            test_idx,
            features,
        ]

        y_test = df.loc[
            test_idx,
            TARGET,
        ].astype(str)

        print(
            f"\nFold {test_fold}: "
            f"train={len(train_idx):,}, "
            f"test={len(test_idx):,}"
        )

        start = time.time()

        model = create_model()

        model.fit(
            X_train,
            y_train,
            sample_weight=make_sample_weights(
                y_train
            ),
        )

        y_pred = model.predict(
            X_test
        )

        elapsed = (
            time.time() - start
        )

        metrics = calculate_metrics(
            y_test,
            y_pred,
        )

        recalls = calculate_recall(
            y_test,
            y_pred,
        )

        print(
            f"  Time: {elapsed:.1f}s"
        )

        print(
            f"  Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"  Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        for idx, true_label, pred_label in zip(
            test_idx,
            y_test,
            y_pred,
        ):

            predictions.append({
                "index": int(idx),
                "fold": int(test_fold),
                "feature_set": feature_set_name,
                "true_class": true_label,
                "prediction": pred_label,
            })

        fold_results.append({
            "fold": int(test_fold),
            "train_samples": int(
                len(train_idx)
            ),
            "test_samples": int(
                len(test_idx)
            ),
            **metrics,
            "recall": recalls,
            "time_seconds": float(
                elapsed
            ),
        })

    # --------------------------------------------------------
    # OOF METRICS
    # --------------------------------------------------------

    pred_df = pd.DataFrame(
        predictions
    )

    y_true = pred_df[
        "true_class"
    ]

    y_pred = pred_df[
        "prediction"
    ]

    final_metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    final_recall = calculate_recall(
        y_true,
        y_pred,
    )

    print("\nFINAL OOF:")

    print(
        f"  Accuracy:          "
        f"{final_metrics['accuracy']:.4f}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{final_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Macro F1:          "
        f"{final_metrics['macro_f1']:.4f}"
    )

    print(
        f"  Weighted F1:       "
        f"{final_metrics['weighted_f1']:.4f}"
    )

    print("\nRecall:")

    for cls in CLASSES:

        print(
            f"  {cls}: "
            f"{final_recall[cls]:.4f}"
        )

    return {
        "feature_set": feature_set_name,
        "n_features": len(features),
        "features": features,
        "fold_results": fold_results,
        "final_oof": {
            **final_metrics,
            "recall": final_recall,
        },
        "predictions": pred_df,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STEP 41 - STABLE FEATURE SUBSET EXPERIMENT"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print(
        f"\nDataset shape: {df.shape}"
    )

    excluded = {
        "record",
        "patient_id",
        "sample",
        "class",
        "fold",
    }

    all_features = [
        c
        for c in df.columns
        if c not in excluded
    ]

    print(
        f"Total ML features: "
        f"{len(all_features)}"
    )

    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    feature_sets = {

        "all_38": all_features,

        "fixed_step39_23": STEP39_23,

        "stable_16": STABLE_16,

        "stable_21": STABLE_21,
    }

    for name, features in feature_sets.items():

        missing = [
            f
            for f in features
            if f not in df.columns
        ]

        if missing:

            raise ValueError(
                f"\n{name} contains missing "
                f"features:\n"
                + "\n".join(missing)
            )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    if (
        df[all_features]
        .isnull()
        .any()
        .any()
    ):

        raise ValueError(
            "Missing values detected."
        )

    if np.isinf(
        df[all_features]
        .to_numpy()
    ).any():

        raise ValueError(
            "Infinite values detected."
        )

    print(
        "Data quality: OK"
    )

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    all_results = {}
    all_predictions = []

    for name, features in feature_sets.items():

        result = evaluate_feature_set(
            df,
            name,
            features,
        )

        all_results[name] = {
            "feature_set": result[
                "feature_set"
            ],
            "n_features": result[
                "n_features"
            ],
            "features": result[
                "features"
            ],
            "fold_results": result[
                "fold_results"
            ],
            "final_oof": result[
                "final_oof"
            ],
        }

        all_predictions.append(
            result["predictions"]
        )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "STEP 41 FINAL COMPARISON"
    )
    print("=" * 80)

    comparison_rows = []

    for name, result in all_results.items():

        m = result[
            "final_oof"
        ]

        recall = m[
            "recall"
        ]

        comparison_rows.append({

            "feature_set":
                name,

            "n_features":
                result[
                    "n_features"
                ],

            "accuracy":
                m[
                    "accuracy"
                ],

            "balanced_accuracy":
                m[
                    "balanced_accuracy"
                ],

            "macro_f1":
                m[
                    "macro_f1"
                ],

            "weighted_f1":
                m[
                    "weighted_f1"
                ],

            "N_recall":
                recall["N"],

            "S_recall":
                recall["S"],

            "V_recall":
                recall["V"],

            "F_recall":
                recall["F"],

            "Q_recall":
                recall["Q"],
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # FEATURE REDUCTION
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "FEATURE REDUCTION ANALYSIS"
    )
    print("=" * 80)

    baseline_balanced = all_results[
        "all_38"
    ][
        "final_oof"
    ][
        "balanced_accuracy"
    ]

    baseline_macro = all_results[
        "all_38"
    ][
        "final_oof"
    ][
        "macro_f1"
    ]

    for name in [
        "fixed_step39_23",
        "stable_16",
        "stable_21",
    ]:

        m = all_results[
            name
        ][
            "final_oof"
        ]

        bal_change = (
            m["balanced_accuracy"]
            - baseline_balanced
        )

        macro_change = (
            m["macro_f1"]
            - baseline_macro
        )

        print(
            f"\n{name}:"
        )

        print(
            f"  Balanced Accuracy change: "
            f"{bal_change:+.4f}"
        )

        print(
            f"  Macro F1 change: "
            f"{macro_change:+.4f}"
        )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        "step41_stable_feature_subset_predictions.csv",
    )

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    output = {

        "step": 41,

        "dataset_shape":
            list(df.shape),

        "total_ml_features":
            len(all_features),

        "classes":
            CLASSES,

        "class_weights":
            CLASS_WEIGHTS,

        "evaluation":
            "Patient-wise 3-fold OOF",

        "model":
            "HistGradientBoostingClassifier",

        "feature_sets": {

            "all_38":
                all_features,

            "fixed_step39_23":
                STEP39_23,

            "stable_16":
                STABLE_16,

            "stable_21":
                STABLE_21,
        },

        "results":
            all_results,
    }

    metrics_path = os.path.join(
        RESULTS_DIR,
        "step41_stable_feature_subset_metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "STEP 41 COMPLETED"
    )
    print("=" * 80)

    print("\nSaved:")

    print(
        metrics_path
    )

    print(
        predictions_path
    )

    print("\nIMPORTANT:")

    print(
        "Do NOT select the final feature set yet."
    )

    print(
        "Send the COMPLETE Step 41 output "
        "for analysis."
    )


if __name__ == "__main__":
    main()