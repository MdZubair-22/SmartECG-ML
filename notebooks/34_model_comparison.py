"""
STEP 34 - CLASS IMBALANCE MODEL COMPARISON
===========================================

Compare ECG classification strategies using the SAME
patient-wise 3-fold splits.

Models:
    1. Standard Random Forest
    2. Class-Weighted Random Forest
    3. Balanced Random Forest

Metrics:
    Accuracy
    Balanced Accuracy
    Macro F1
    Weighted F1
    Per-class precision/recall/F1
    Confusion matrix

IMPORTANT:
    No patient appears in more than one fold.
"""

import os
import json
import warnings

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

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "enhanced_causal_ecg_features.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "step34_model_comparison.json"
)

PREDICTION_FILE = os.path.join(
    RESULTS_DIR,
    "step34_predictions.csv"
)

CLASSES = [
    "N",
    "S",
    "V",
    "F",
    "Q"
]

CLASS_TO_INT = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4
}

INT_TO_CLASS = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q"
}

N_FOLDS = 3

RANDOM_STATE = 42

N_ESTIMATORS = 300


# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODELS = {

    "standard_random_forest": {
        "type": "RandomForestClassifier",
        "class_weight": None
    },

    "class_weighted_random_forest": {
        "type": "RandomForestClassifier",
        "class_weight": {
            0: 1.0,
            1: 2.5,
            2: 1.0,
            3: 8.0,
            4: 1.2
        }
    }
}


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )
    }


# ============================================================
# MODEL CREATION
# ============================================================

def create_rf(class_weight):

    return RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features="sqrt",

        class_weight=class_weight,

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "STEP 34 - CLASS IMBALANCE MODEL COMPARISON"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print()
    print("Loading enhanced dataset...")

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "record",
        "patient_id",
        "sample",
        "class",
        "fold"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
    )

    df["class"] = (
        df["class"]
        .astype(str)
        .str.strip()
    )

    df["fold"] = pd.to_numeric(
        df["fold"],
        errors="coerce"
    )

    if df["fold"].isna().any():

        raise ValueError(
            "NaN found in fold column."
        )

    df["fold"] = df["fold"].astype(int)

    # --------------------------------------------------------
    # CHECK PATIENT LEAKAGE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PATIENT LEAKAGE CHECK")
    print("=" * 70)

    patient_fold_counts = (
        df.groupby("patient_id")["fold"]
        .nunique()
    )

    leakage = patient_fold_counts[
        patient_fold_counts > 1
    ]

    print(
        f"Patients appearing in multiple folds: "
        f"{len(leakage)}"
    )

    if len(leakage) > 0:

        raise ValueError(
            "PATIENT LEAKAGE DETECTED."
        )

    print(
        "SUCCESS: No patient appears in "
        "multiple folds."
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    excluded = {
        "record",
        "patient_id",
        "sample",
        "class",
        "fold",
        "time_seconds",
        "annotation"
    }

    feature_columns = [
        c for c in df.columns
        if c not in excluded
    ]

    print()
    print(
        f"Number of ML features: "
        f"{len(feature_columns)}"
    )

    X = (
        df[feature_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    if np.isnan(X).any():

        raise ValueError(
            "NaN values detected."
        )

    if np.isinf(X).any():

        raise ValueError(
            "Infinite values detected."
        )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    unknown_classes = (
        set(df["class"].unique())
        - set(CLASSES)
    )

    if unknown_classes:

        raise ValueError(
            f"Unknown classes: "
            f"{unknown_classes}"
        )

    y = (
        df["class"]
        .map(CLASS_TO_INT)
        .to_numpy(
            dtype=np.int8
        )
    )

    folds = df["fold"].to_numpy(
        dtype=np.int8
    )

    # ========================================================
    # RESULTS STORAGE
    # ========================================================

    all_results = {}

    all_prediction_rows = []

    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model_name, config in MODELS.items():

        print()
        print("=" * 70)
        print(
            f"MODEL: {model_name.upper()}"
        )
        print("=" * 70)

        print()
        print(
            f"Class weight: "
            f"{config['class_weight']}"
        )

        model_predictions = np.full(
            len(df),
            -1,
            dtype=np.int8
        )

        model_probabilities = np.zeros(
            (
                len(df),
                len(CLASSES)
            ),
            dtype=np.float32
        )

        fold_results = {}

        # ----------------------------------------------------
        # 3 FOLDS
        # ----------------------------------------------------

        for test_fold in range(N_FOLDS):

            print()
            print(
                "-" * 70
            )
            print(
                f"TEST FOLD: {test_fold}"
            )
            print(
                "-" * 70
            )

            train_mask = (
                folds != test_fold
            )

            test_mask = (
                folds == test_fold
            )

            train_idx = np.where(
                train_mask
            )[0]

            test_idx = np.where(
                test_mask
            )[0]

            print(
                f"Training samples: "
                f"{len(train_idx)}"
            )

            print(
                f"Testing samples: "
                f"{len(test_idx)}"
            )

            # ------------------------------------------------
            # DISTRIBUTION
            # ------------------------------------------------

            print()
            print(
                "Training distribution:"
            )

            print(
                pd.Series(
                    [
                        INT_TO_CLASS[v]
                        for v in y[train_idx]
                    ]
                ).value_counts()
            )

            print()
            print(
                "Testing distribution:"
            )

            print(
                pd.Series(
                    [
                        INT_TO_CLASS[v]
                        for v in y[test_idx]
                    ]
                ).value_counts()
            )

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            print()
            print(
                "Training model..."
            )

            model = create_rf(
                config["class_weight"]
            )

            model.fit(
                X[train_idx],
                y[train_idx]
            )

            print(
                "Training complete."
            )

            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            probabilities = (
                model.predict_proba(
                    X[test_idx]
                )
            )

            full_probabilities = np.zeros(
                (
                    len(test_idx),
                    len(CLASSES)
                ),
                dtype=np.float32
            )

            for column, class_id in enumerate(
                model.classes_
            ):

                full_probabilities[
                    :,
                    int(class_id)
                ] = probabilities[
                    :,
                    column
                ]

            predictions = np.argmax(
                full_probabilities,
                axis=1
            )

            model_predictions[
                test_idx
            ] = predictions

            model_probabilities[
                test_idx
            ] = full_probabilities

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            metrics = calculate_metrics(
                y[test_idx],
                predictions
            )

            fold_results[
                str(test_fold)
            ] = metrics

            print()
            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Balanced Accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )

            print(
                f"Macro F1: "
                f"{metrics['macro_f1']:.4f}"
            )

            print(
                f"Weighted F1: "
                f"{metrics['weighted_f1']:.4f}"
            )

        # ====================================================
        # FINAL OOF METRICS
        # ====================================================

        print()
        print("=" * 70)
        print(
            f"FINAL OOF RESULTS: "
            f"{model_name}"
        )
        print("=" * 70)

        missing_predictions = np.sum(
            model_predictions < 0
        )

        if missing_predictions > 0:

            raise ValueError(
                f"Missing predictions: "
                f"{missing_predictions}"
            )

        metrics = calculate_metrics(
            y,
            model_predictions
        )

        print()
        print(
            f"Accuracy:           "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy:  "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1:           "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1:        "
            f"{metrics['weighted_f1']:.4f}"
        )

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        print()
        print(
            "Classification Report:"
        )

        print(
            classification_report(
                y,
                model_predictions,
                labels=list(
                    range(len(CLASSES))
                ),
                target_names=CLASSES,
                digits=4,
                zero_division=0
            )
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        cm = confusion_matrix(
            y,
            model_predictions,
            labels=list(
                range(len(CLASSES))
            )
        )

        print(
            "Confusion Matrix:"
        )

        print(
            pd.DataFrame(
                cm,
                index=CLASSES,
                columns=CLASSES
            )
        )

        # ----------------------------------------------------
        # PER CLASS METRICS
        # ----------------------------------------------------

        report = classification_report(
            y,
            model_predictions,
            labels=list(
                range(len(CLASSES))
            ),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0
        )

        per_class = {}

        for class_name in CLASSES:

            per_class[class_name] = {

                "precision": float(
                    report[class_name][
                        "precision"
                    ]
                ),

                "recall": float(
                    report[class_name][
                        "recall"
                    ]
                ),

                "f1": float(
                    report[class_name][
                        "f1-score"
                    ]
                ),

                "support": int(
                    report[class_name][
                        "support"
                    ]
                )
            }

        all_results[
            model_name
        ] = {

            "model_config": config,

            "final_oof_metrics": metrics,

            "per_class_metrics": per_class,

            "confusion_matrix": cm.tolist(),

            "fold_metrics": fold_results
        }

        # ----------------------------------------------------
        # SAVE PREDICTIONS FOR THIS MODEL
        # ----------------------------------------------------

        for i in range(len(df)):

            all_prediction_rows.append({

                "sample": df.iloc[i]["sample"],

                "record": df.iloc[i]["record"],

                "patient_id": df.iloc[i][
                    "patient_id"
                ],

                "fold": int(
                    df.iloc[i]["fold"]
                ),

                "true_class": df.iloc[i][
                    "class"
                ],

                "model": model_name,

                "predicted_class":
                    INT_TO_CLASS[
                        int(
                            model_predictions[i]
                        )
                    ],

                "prob_N":
                    float(
                        model_probabilities[
                            i, 0
                        ]
                    ),

                "prob_S":
                    float(
                        model_probabilities[
                            i, 1
                        ]
                    ),

                "prob_V":
                    float(
                        model_probabilities[
                            i, 2
                        ]
                    ),

                "prob_F":
                    float(
                        model_probabilities[
                            i, 3
                        ]
                    ),

                "prob_Q":
                    float(
                        model_probabilities[
                            i, 4
                        ]
                    )
            })

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_rows = []

    for model_name, result in all_results.items():

        metrics = result[
            "final_oof_metrics"
        ]

        comparison_rows.append({

            "model": model_name,

            "accuracy":
                metrics["accuracy"],

            "balanced_accuracy":
                metrics["balanced_accuracy"],

            "macro_f1":
                metrics["macro_f1"],

            "weighted_f1":
                metrics["weighted_f1"]
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print()

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # PER-CLASS COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("PER-CLASS RECALL COMPARISON")
    print("=" * 70)

    recall_rows = []

    for class_name in CLASSES:

        row = {
            "class": class_name
        }

        for model_name, result in (
            all_results.items()
        ):

            row[
                model_name
            ] = result[
                "per_class_metrics"
            ][
                class_name
            ][
                "recall"
            ]

        recall_rows.append(row)

    recall_df = pd.DataFrame(
        recall_rows
    )

    print(
        recall_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    print()
    print(
        "Saving model comparison..."
    )

    final_output = {

        "step": 34,

        "dataset": os.path.basename(
            FEATURE_FILE
        ),

        "samples": int(
            len(df)
        ),

        "features": int(
            len(feature_columns)
        ),

        "classes": CLASSES,

        "folds": N_FOLDS,

        "patient_leakage": 0,

        "models": all_results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_output,
            f,
            indent=4
        )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    # ========================================================
    # SAVE PREDICTIONS IN CHUNKS
    # ========================================================

    print()
    print(
        "Saving predictions..."
    )

    prediction_df = pd.DataFrame(
        all_prediction_rows
    )

    temp_file = (
        PREDICTION_FILE
        + ".tmp"
    )

    if os.path.exists(temp_file):
        os.remove(temp_file)

    chunk_size = 10000

    for start in range(
        0,
        len(prediction_df),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(prediction_df)
        )

        prediction_df.iloc[
            start:end
        ].to_csv(
            temp_file,
            mode=(
                "w"
                if start == 0
                else "a"
            ),
            header=(
                start == 0
            ),
            index=False
        )

        print(
            f"Saved rows "
            f"{start:,} - {end:,}"
        )

    if os.path.exists(
        PREDICTION_FILE
    ):
        os.remove(
            PREDICTION_FILE
        )

    os.replace(
        temp_file,
        PREDICTION_FILE
    )

    print(
        f"Saved: {PREDICTION_FILE}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 34 COMPLETE")
    print("=" * 70)

    print()
    print(
        "Model comparison:"
    )

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print()
    print(
        "Next step: Step 35 - Feature importance "
        "and error analysis."
    )