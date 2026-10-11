"""
STEP 40 - NESTED / FOLD-SPECIFIC FEATURE SELECTION
===================================================

Purpose:
- Validate whether the Step 39 23-feature improvement is robust.
- Perform feature selection using TRAINING data only.
- Never use the held-out test fold for feature selection.
- Patient-wise 3-fold OOF evaluation.
- Compare:
    1. All 38 features
    2. Fixed Step-39 23 features
    3. Fold-specific selected features

Model:
    HistGradientBoostingClassifier
    Same balanced class weights as Step 36/39

Output:
    results/step40_nested_feature_selection_metrics.json
    results/step40_nested_feature_selection_predictions.csv
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
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "enhanced_causal_ecg_features.csv",
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET = "class"
FOLD_COLUMN = "fold"

CLASSES = ["N", "S", "V", "F", "Q"]

# Same balanced weights used in Step 36 and Step 39
CLASS_WEIGHTS = {
    "N": 1.0,
    "S": 3.0,
    "V": 1.0,
    "F": 5.0,
    "Q": 1.5,
}


# ============================================================
# STEP 39 FIXED 23 FEATURES
# ============================================================

STEP39_23 = [
    # Rhythm
    "prev_rr",
    "prev_hr",

    # Spectral
    "spectral_entropy",
    "dominant_frequency",
    "energy_ratio",

    # S/F morphology
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
# MODEL
# ============================================================

def create_model():
    """
    Same HGB configuration used in Step 36 / Step 39.
    """

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
        [CLASS_WEIGHTS[label] for label in y],
        dtype=np.float64,
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features_from_training_data(
    X_train,
    y_train,
    candidate_features,
):
    """
    Select features using ONLY training data.

    Uses permutation importance on the training subset.

    To reduce instability from a single fitted model:
    - fit HGB on training data
    - calculate permutation importance
    - rank candidate features
    - retain the strongest features

    IMPORTANT:
    No test-fold information is used here.
    """

    print("\n  Performing TRAINING-ONLY feature selection...")

    selector_model = create_model()

    sample_weights = make_sample_weights(y_train)

    selector_model.fit(
        X_train[candidate_features],
        y_train,
        sample_weight=sample_weights,
    )

    # Permutation importance on training data.
    # This is used only for ranking within the training fold.
    result = permutation_importance(
        selector_model,
        X_train[candidate_features],
        y_train,
        scoring="balanced_accuracy",
        n_repeats=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": candidate_features,
        "importance": result.importances_mean,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    # Keep top 23 features.
    selected = importance.head(23)["feature"].tolist()

    print("  Selected features:")
    for i, feature in enumerate(selected, 1):
        score = importance.loc[
            importance["feature"] == feature,
            "importance"
        ].iloc[0]

        print(f"    {i:2d}. {feature:<40} {score:.6f}")

    return selected, importance


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(y_true, y_pred):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
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


def recall_by_class(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES,
    )

    recalls = {}

    for i, cls in enumerate(CLASSES):
        denominator = cm[i].sum()

        if denominator == 0:
            recalls[cls] = 0.0
        else:
            recalls[cls] = float(
                cm[i, i] / denominator
            )

    return recalls


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STEP 40 - NESTED / FOLD-SPECIFIC FEATURE SELECTION")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------------
    # IDENTIFY FEATURES
    # --------------------------------------------------------

    excluded = {
        "record",
        "patient_id",
        "sample",
        "class",
        "fold",
    }

    all_features = [
        c for c in df.columns
        if c not in excluded
    ]

    print(f"Total ML features: {len(all_features)}")

    missing_step39 = [
        f for f in STEP39_23
        if f not in df.columns
    ]

    if missing_step39:
        raise ValueError(
            "Missing Step-39 features:\n"
            + "\n".join(missing_step39)
        )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    X_all = df[all_features].copy()

    if X_all.isnull().any().any():
        raise ValueError("Missing values detected.")

    if np.isinf(X_all.to_numpy()).any():
        raise ValueError("Infinite values detected.")

    y_all = df[TARGET].astype(str)

    folds = sorted(
        df[FOLD_COLUMN].dropna().unique()
    )

    print(f"Folds: {folds}")
    print(f"Classes: {CLASSES}")

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    predictions = []

    fold_results = []

    selected_features_by_fold = {}

    # --------------------------------------------------------
    # PATIENT-WISE 3-FOLD OOF
    # --------------------------------------------------------

    for test_fold in folds:

        print("\n" + "=" * 80)
        print(f"TEST FOLD {test_fold}")
        print("=" * 80)

        train_mask = df[FOLD_COLUMN] != test_fold
        test_mask = df[FOLD_COLUMN] == test_fold

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        X_train = df.loc[
            train_idx,
            all_features
        ].copy()

        y_train = df.loc[
            train_idx,
            TARGET
        ].astype(str)

        X_test = df.loc[
            test_idx,
            all_features
        ].copy()

        y_test = df.loc[
            test_idx,
            TARGET
        ].astype(str)

        print(
            f"Train={len(train_idx):,}, "
            f"Test={len(test_idx):,}"
        )

        # ----------------------------------------------------
        # MODEL A: ALL 38
        # ----------------------------------------------------

        print("\n[MODEL A] ALL 38 FEATURES")

        start = time.time()

        model_all = create_model()

        model_all.fit(
            X_train[all_features],
            y_train,
            sample_weight=make_sample_weights(y_train),
        )

        pred_all = model_all.predict(
            X_test[all_features]
        )

        time_all = time.time() - start

        metrics_all = evaluate_predictions(
            y_test,
            pred_all,
        )

        print(
            f"Accuracy:          "
            f"{metrics_all['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics_all['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1:          "
            f"{metrics_all['macro_f1']:.4f}"
        )

        # ----------------------------------------------------
        # MODEL B: FIXED STEP-39 23 FEATURES
        # ----------------------------------------------------

        print("\n[MODEL B] FIXED STEP-39 23 FEATURES")

        start = time.time()

        model_fixed = create_model()

        model_fixed.fit(
            X_train[STEP39_23],
            y_train,
            sample_weight=make_sample_weights(y_train),
        )

        pred_fixed = model_fixed.predict(
            X_test[STEP39_23]
        )

        time_fixed = time.time() - start

        metrics_fixed = evaluate_predictions(
            y_test,
            pred_fixed,
        )

        print(
            f"Accuracy:          "
            f"{metrics_fixed['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics_fixed['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1:          "
            f"{metrics_fixed['macro_f1']:.4f}"
        )

        # ----------------------------------------------------
        # MODEL C: FOLD-SPECIFIC FEATURE SELECTION
        # ----------------------------------------------------

        print(
            "\n[MODEL C] FOLD-SPECIFIC "
            "TRAINING-ONLY FEATURE SELECTION"
        )

        selected_features, importance = (
            select_features_from_training_data(
                X_train,
                y_train,
                all_features,
            )
        )

        selected_features_by_fold[str(test_fold)] = (
            selected_features
        )

        start = time.time()

        model_selected = create_model()

        model_selected.fit(
            X_train[selected_features],
            y_train,
            sample_weight=make_sample_weights(y_train),
        )

        pred_selected = model_selected.predict(
            X_test[selected_features]
        )

        time_selected = time.time() - start

        metrics_selected = evaluate_predictions(
            y_test,
            pred_selected,
        )

        recalls_selected = recall_by_class(
            y_test,
            pred_selected,
        )

        print(
            f"\nSelected feature count: "
            f"{len(selected_features)}"
        )

        print(
            f"Accuracy:          "
            f"{metrics_selected['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics_selected['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1:          "
            f"{metrics_selected['macro_f1']:.4f}"
        )

        print("Recall:")

        for cls in CLASSES:
            print(
                f"  {cls}: "
                f"{recalls_selected[cls]:.4f}"
            )

        # ----------------------------------------------------
        # SAVE OOF PREDICTIONS
        # ----------------------------------------------------

        for idx, true_label, pa, pf, ps in zip(
            test_idx,
            y_test,
            pred_all,
            pred_fixed,
            pred_selected,
        ):

            predictions.append({
                "index": int(idx),
                "fold": int(test_fold),
                "true_class": true_label,
                "all_38_prediction": pa,
                "fixed_23_prediction": pf,
                "nested_prediction": ps,
            })

        # ----------------------------------------------------
        # FOLD RESULT
        # ----------------------------------------------------

        fold_results.append({
            "test_fold": int(test_fold),

            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),

            "all_38": {
                **metrics_all,
                "time_seconds": float(time_all),
            },

            "fixed_step39_23": {
                **metrics_fixed,
                "time_seconds": float(time_fixed),
            },

            "nested_selected": {
                **metrics_selected,
                "recall": recalls_selected,
                "n_features": len(selected_features),
                "selected_features": selected_features,
                "time_seconds": float(time_selected),
            },
        })

    # ========================================================
    # FINAL OOF RESULTS
    # ========================================================

    pred_df = pd.DataFrame(predictions)

    print("\n")
    print("=" * 80)
    print("STEP 40 FINAL OOF COMPARISON")
    print("=" * 80)

    results = {}

    model_columns = {
        "all_38": "all_38_prediction",
        "fixed_step39_23": "fixed_23_prediction",
        "nested_selected": "nested_prediction",
    }

    y_true = pred_df["true_class"]

    for model_name, prediction_column in model_columns.items():

        y_pred = pred_df[prediction_column]

        metrics = evaluate_predictions(
            y_true,
            y_pred,
        )

        recalls = recall_by_class(
            y_true,
            y_pred,
        )

        results[model_name] = {
            **metrics,
            "recall": recalls,
        }

        print(f"\n{model_name.upper()}")

        print(
            f"Accuracy:          "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1:          "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1:       "
            f"{metrics['weighted_f1']:.4f}"
        )

        print("\nRecall:")

        for cls in CLASSES:
            print(
                f"  {cls}: "
                f"{recalls[cls]:.4f}"
            )

    # ========================================================
    # FEATURE STABILITY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FEATURE SELECTION STABILITY")
    print("=" * 80)

    feature_counts = {}

    for fold, features in selected_features_by_fold.items():

        print(f"\nFold {fold}:")
        print("  " + ", ".join(features))

        for feature in features:
            feature_counts[feature] = (
                feature_counts.get(feature, 0) + 1
            )

    stable_features = sorted(
        feature_counts.items(),
        key=lambda x: (-x[1], x[0]),
    )

    print("\nFeature selection frequency:")

    for feature, count in stable_features:
        print(
            f"  {feature:<40} "
            f"{count}/3 folds"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    metrics_output = {
        "step": 40,

        "dataset_shape": list(df.shape),

        "total_features": len(all_features),

        "classes": CLASSES,

        "class_weights": CLASS_WEIGHTS,

        "folds": [int(f) for f in folds],

        "step39_fixed_features": STEP39_23,

        "fold_results": fold_results,

        "final_oof": results,

        "selected_features_by_fold":
            selected_features_by_fold,

        "feature_selection_frequency":
            {
                feature: int(count)
                for feature, count
                in feature_counts.items()
            },

        "methodology": {
            "evaluation":
                "Patient-wise 3-fold OOF",

            "feature_selection":
                "Performed separately inside each training fold",

            "test_fold_used_for_selection":
                False,

            "model":
                "HistGradientBoostingClassifier",

            "class_weighting":
                "Sample weights N=1, S=3, V=1, F=5, Q=1.5",

            "purpose":
                "Validate robustness of Step-39 feature reduction",
        },
    }

    metrics_path = os.path.join(
        RESULTS_DIR,
        "step40_nested_feature_selection_metrics.json",
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        "step40_nested_feature_selection_predictions.csv",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_output,
            f,
            indent=2,
        )

    pred_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("STEP 40 COMPLETED")
    print("=" * 80)

    print("\nSaved:")
    print(metrics_path)
    print(predictions_path)

    print("\nIMPORTANT:")
    print(
        "Send the COMPLETE Step 40 output "
        "before selecting the final feature set."
    )


if __name__ == "__main__":
    main()