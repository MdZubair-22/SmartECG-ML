"""
STEP 42 - HGB HYPERPARAMETER OPTIMIZATION
==========================================

Goal:
- Optimize the current best 23-feature ECG model.
- Keep the Step-39 23-feature set fixed.
- Patient-wise 3-fold OOF evaluation.
- Use the same class weights as Steps 36-41.
- Avoid a huge brute-force search.

Current reference:
    Accuracy          = 86.96%
    Balanced Accuracy = 60.22%
    Macro F1          = 55.53%

Primary optimization metric:
    Balanced Accuracy

Secondary metric:
    Macro F1

Outputs:
    results/step42_hgb_hyperparameter_metrics.json
    results/step42_hgb_hyperparameter_predictions.csv
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


# ============================================================
# CLASS WEIGHTS
# ============================================================

CLASS_WEIGHTS = {
    "N": 1.0,
    "S": 3.0,
    "V": 1.0,
    "F": 5.0,
    "Q": 1.5,
}


# ============================================================
# CURRENT BEST 23 FEATURES
# ============================================================

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


# ============================================================
# HYPERPARAMETER CANDIDATES
# ============================================================

# Current Step-39/41 baseline:
#
# learning_rate = 0.08
# max_iter = 300
# max_leaf_nodes = 31
# min_samples_leaf = 20
# l2_regularization = 1.0

PARAMETER_CONFIGS = [

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    {
        "name": "baseline",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    # --------------------------------------------------------
    # LEARNING RATE / ITERATIONS
    # --------------------------------------------------------

    {
        "name": "lr_005_iter400",
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    {
        "name": "lr_010_iter250",
        "learning_rate": 0.10,
        "max_iter": 250,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    {
        "name": "lr_015_iter200",
        "learning_rate": 0.15,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    # --------------------------------------------------------
    # TREE COMPLEXITY
    # --------------------------------------------------------

    {
        "name": "leaf15",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    {
        "name": "leaf63",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    {
        "name": "leaf127",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 127,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    # --------------------------------------------------------
    # MINIMUM LEAF SIZE
    # --------------------------------------------------------

    {
        "name": "minleaf10",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 1.0,
    },

    {
        "name": "minleaf40",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
    },

    {
        "name": "minleaf80",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 80,
        "l2_regularization": 1.0,
    },

    # --------------------------------------------------------
    # REGULARIZATION
    # --------------------------------------------------------

    {
        "name": "l2_01",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },

    {
        "name": "l2_3",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 3.0,
    },

    {
        "name": "l2_10",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 10.0,
    },

    # --------------------------------------------------------
    # COMBINED CONFIGURATIONS
    # --------------------------------------------------------

    {
        "name": "balanced_complex",
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    {
        "name": "balanced_regularized",
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 3.0,
    },

    {
        "name": "deep_regularized",
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 40,
        "l2_regularization": 3.0,
    },

    {
        "name": "compact_regularized",
        "learning_rate": 0.10,
        "max_iter": 250,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
        "l2_regularization": 3.0,
    },
]


# ============================================================
# MODEL
# ============================================================

def create_model(params):

    return HistGradientBoostingClassifier(

        learning_rate=params[
            "learning_rate"
        ],

        max_iter=params[
            "max_iter"
        ],

        max_leaf_nodes=params[
            "max_leaf_nodes"
        ],

        min_samples_leaf=params[
            "min_samples_leaf"
        ],

        l2_regularization=params[
            "l2_regularization"
        ],

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

def calculate_metrics(
    y_true,
    y_pred,
):

    return {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=CLASSES,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=CLASSES,
                    average="weighted",
                    zero_division=0,
                )
            ),
    }


def calculate_recall(
    y_true,
    y_pred,
):

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
# EVALUATE ONE CONFIGURATION
# ============================================================

def evaluate_configuration(
    df,
    params,
):

    name = params["name"]

    print("\n")
    print("-" * 80)
    print(
        f"CONFIGURATION: {name}"
    )
    print("-" * 80)

    print(
        f"learning_rate = "
        f"{params['learning_rate']}"
    )

    print(
        f"max_iter = "
        f"{params['max_iter']}"
    )

    print(
        f"max_leaf_nodes = "
        f"{params['max_leaf_nodes']}"
    )

    print(
        f"min_samples_leaf = "
        f"{params['min_samples_leaf']}"
    )

    print(
        f"l2_regularization = "
        f"{params['l2_regularization']}"
    )

    oof_true = []
    oof_pred = []

    fold_results = []

    total_start = time.time()

    for test_fold in sorted(
        df[FOLD_COLUMN].unique()
    ):

        train_mask = (
            df[FOLD_COLUMN]
            != test_fold
        )

        test_mask = (
            df[FOLD_COLUMN]
            == test_fold
        )

        train_idx = df.index[
            train_mask
        ]

        test_idx = df.index[
            test_mask
        ]

        X_train = df.loc[
            train_idx,
            FEATURES,
        ]

        y_train = df.loc[
            train_idx,
            TARGET,
        ].astype(str)

        X_test = df.loc[
            test_idx,
            FEATURES,
        ]

        y_test = df.loc[
            test_idx,
            TARGET,
        ].astype(str)

        start = time.time()

        model = create_model(
            params
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=
                make_sample_weights(
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

        print(
            f"\nFold {test_fold}: "
            f"train={len(train_idx):,}, "
            f"test={len(test_idx):,}"
        )

        print(
            f"  Time: "
            f"{elapsed:.1f}s"
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

        oof_true.extend(
            y_test.tolist()
        )

        oof_pred.extend(
            y_pred.tolist()
        )

        fold_results.append({
            "fold":
                int(test_fold),

            "train_samples":
                int(len(train_idx)),

            "test_samples":
                int(len(test_idx)),

            **metrics,

            "time_seconds":
                float(elapsed),
        })

    # --------------------------------------------------------
    # FINAL OOF
    # --------------------------------------------------------

    final_metrics = calculate_metrics(
        oof_true,
        oof_pred,
    )

    recalls = calculate_recall(
        oof_true,
        oof_pred,
    )

    total_time = (
        time.time() - total_start
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
            f"{recalls[cls]:.4f}"
        )

    return {

        "name":
            name,

        "parameters":
            params,

        "fold_results":
            fold_results,

        "final_oof": {
            **final_metrics,
            "recall":
                recalls,
        },

        "total_time_seconds":
            float(total_time),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STEP 42 - HGB HYPERPARAMETER OPTIMIZATION"
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
        f"\nDataset shape: "
        f"{df.shape}"
    )

    print(
        f"Using features: "
        f"{len(FEATURES)}"
    )

    print(
        f"Configurations: "
        f"{len(PARAMETER_CONFIGS)}"
    )

    # --------------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------------

    missing = [
        f
        for f in FEATURES
        if f not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing features:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    if (
        df[FEATURES]
        .isnull()
        .any()
        .any()
    ):

        raise ValueError(
            "Missing values detected."
        )

    if np.isinf(
        df[FEATURES]
        .to_numpy()
    ).any():

        raise ValueError(
            "Infinite values detected."
        )

    print(
        "Data quality: OK"
    )

    # --------------------------------------------------------
    # RUN ALL CONFIGURATIONS
    # --------------------------------------------------------

    all_results = []

    for i, params in enumerate(
        PARAMETER_CONFIGS,
        start=1,
    ):

        print("\n")
        print("=" * 80)

        print(
            f"CONFIGURATION "
            f"{i}/{len(PARAMETER_CONFIGS)}"
        )

        print("=" * 80)

        result = evaluate_configuration(
            df,
            params,
        )

        all_results.append(
            result
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "STEP 42 FINAL COMPARISON"
    )
    print("=" * 80)

    rows = []

    for result in all_results:

        m = result[
            "final_oof"
        ]

        p = result[
            "parameters"
        ]

        r = m[
            "recall"
        ]

        rows.append({

            "configuration":
                result["name"],

            "learning_rate":
                p["learning_rate"],

            "max_iter":
                p["max_iter"],

            "max_leaf_nodes":
                p["max_leaf_nodes"],

            "min_samples_leaf":
                p["min_samples_leaf"],

            "l2_regularization":
                p["l2_regularization"],

            "accuracy":
                m["accuracy"],

            "balanced_accuracy":
                m["balanced_accuracy"],

            "macro_f1":
                m["macro_f1"],

            "weighted_f1":
                m["weighted_f1"],

            "N_recall":
                r["N"],

            "S_recall":
                r["S"],

            "V_recall":
                r["V"],

            "F_recall":
                r["F"],

            "Q_recall":
                r["Q"],
        })

    comparison_df = pd.DataFrame(
        rows
    )

    # Sort primarily by balanced accuracy,
    # secondarily by macro F1.

    comparison_df = comparison_df.sort_values(
        [
            "balanced_accuracy",
            "macro_f1",
        ],
        ascending=False,
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST CONFIGURATION
    # --------------------------------------------------------

    best_row = (
        comparison_df.iloc[0]
    )

    best_name = (
        best_row[
            "configuration"
        ]
    )

    print("\n")
    print("=" * 80)
    print(
        "BEST CONFIGURATION "
        "(BALANCED ACCURACY)"
    )
    print("=" * 80)

    print(
        f"Name: "
        f"{best_name}"
    )

    print(
        f"Accuracy: "
        f"{best_row['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_row['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{best_row['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{best_row['weighted_f1']:.4f}"
    )

    print("\nRecall:")

    for cls in CLASSES:

        print(
            f"  {cls}: "
            f"{best_row[f'{cls}_recall']:.4f}"
        )

    # --------------------------------------------------------
    # COMPARE WITH STEP 39
    # --------------------------------------------------------

    reference_balanced = 0.602211
    reference_macro = 0.555344
    reference_accuracy = 0.869583

    print("\n")
    print("=" * 80)
    print(
        "COMPARISON WITH CURRENT STEP-39 BEST"
    )
    print("=" * 80)

    print(
        f"Reference Accuracy: "
        f"{reference_accuracy:.4f}"
    )

    print(
        f"Optimized Accuracy: "
        f"{best_row['accuracy']:.4f}"
    )

    print(
        f"Change: "
        f"{best_row['accuracy'] - reference_accuracy:+.4f}"
    )

    print()

    print(
        f"Reference Balanced Accuracy: "
        f"{reference_balanced:.4f}"
    )

    print(
        f"Optimized Balanced Accuracy: "
        f"{best_row['balanced_accuracy']:.4f}"
    )

    print(
        f"Change: "
        f"{best_row['balanced_accuracy'] - reference_balanced:+.4f}"
    )

    print()

    print(
        f"Reference Macro F1: "
        f"{reference_macro:.4f}"
    )

    print(
        f"Optimized Macro F1: "
        f"{best_row['macro_f1']:.4f}"
    )

    print(
        f"Change: "
        f"{best_row['macro_f1'] - reference_macro:+.4f}"
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    output = {

        "step": 42,

        "dataset_shape":
            list(df.shape),

        "features":
            FEATURES,

        "n_features":
            len(FEATURES),

        "classes":
            CLASSES,

        "class_weights":
            CLASS_WEIGHTS,

        "evaluation":
            "Patient-wise 3-fold OOF",

        "optimization_metric":
            "Balanced Accuracy",

        "secondary_metric":
            "Macro F1",

        "reference_step39": {

            "accuracy":
                reference_accuracy,

            "balanced_accuracy":
                reference_balanced,

            "macro_f1":
                reference_macro,
        },

        "configurations":
            all_results,

        "sorted_comparison":
            rows,
    }

    metrics_path = os.path.join(
        RESULTS_DIR,
        "step42_hgb_hyperparameter_metrics.json",
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
    # SAVE CSV
    # --------------------------------------------------------

    predictions_path = os.path.join(
        RESULTS_DIR,
        "step42_hgb_hyperparameter_predictions.csv",
    )

    # Save comparison table as CSV.
    comparison_df.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "STEP 42 COMPLETED"
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
        "Do NOT change the class weights or "
        "feature set yet."
    )

    print(
        "Send the COMPLETE Step 42 output "
        "for analysis."
    )


if __name__ == "__main__":
    main()