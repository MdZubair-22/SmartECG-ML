"""
STEP 47B — NESTED PATIENT-WISE CROSS-VALIDATION

Purpose:
    Rigorous evaluation of SmartECG-ML using nested patient-wise CV.

Outer CV:
    Existing patient-wise folds 0, 1, 2.

Inner CV:
    Patient-wise CV performed only on the outer training patients.

Inside each outer fold:
    1. Inner CV compares a small set of HGB configurations.
    2. Best configuration is selected using INNER balanced accuracy.
    3. Best model is retrained on ALL outer-training data.
    4. Completely untouched outer-test patients are evaluated.

This prevents the outer test patients from influencing
hyperparameter selection.

IMPORTANT:
    The fixed 23-feature set comes from the previous project
    feature-selection process. This experiment evaluates the
    frozen feature/model pipeline under nested validation.

No dataset files are modified.
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
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enhanced_causal_ecg_features.csv"
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

CLASSES = [
    "N",
    "S",
    "V",
    "F",
    "Q",
]


# ============================================================
# 3. FROZEN 23 FEATURES
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
# 4. CLASS WEIGHTS
# ============================================================

CLASS_WEIGHTS = {
    "N": 1.0,
    "S": 3.0,
    "V": 1.0,
    "F": 5.0,
    "Q": 1.5,
}


# ============================================================
# 5. INNER-CV HYPERPARAMETER SEARCH SPACE
# ============================================================

PARAMETER_CONFIGS = {

    "baseline": {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    "l2_01": {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },

    "lr_010": {
        "learning_rate": 0.10,
        "max_iter": 250,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    "lr_005": {
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    "minleaf40": {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
    },

    "leaf15": {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    "leaf63": {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    "l2_3": {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 3.0,
    },
}


# ============================================================
# 6. HELPER FUNCTION
# ============================================================

def train_model(X_train, y_train, params):

    model = HistGradientBoostingClassifier(
        **params,
        random_state=RANDOM_STATE,
    )

    sample_weights = (
        y_train
        .map(CLASS_WEIGHTS)
        .to_numpy()
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    return model


# ============================================================
# 7. LOAD DATA
# ============================================================

print("=" * 80)
print("STEP 47B — NESTED PATIENT-WISE CROSS-VALIDATION")
print("=" * 80)

start_time = time.time()

print("\n[1/8] Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Patients: "
    f"{df['patient_id'].nunique()}"
)

print(
    f"Samples: "
    f"{len(df):,}"
)


# ============================================================
# 8. VERIFY DATA
# ============================================================

print("\n[2/8] Verifying dataset...")

required = [
    "patient_id",
    "fold",
    "class",
] + FEATURES

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


X = df[FEATURES].copy()

y = df["class"].astype(str)

folds = df["fold"].astype(int)

if X.isna().sum().sum() > 0:

    raise ValueError(
        "Missing values found."
    )

if np.isinf(X.to_numpy()).sum() > 0:

    raise ValueError(
        "Infinite values found."
    )


# ============================================================
# 9. VERIFY PATIENT/FOLD INTEGRITY
# ============================================================

patient_fold_counts = (
    df.groupby("patient_id")["fold"]
    .nunique()
)

leaking_patients = (
    patient_fold_counts[
        patient_fold_counts > 1
    ]
)

if len(leaking_patients) > 0:

    raise RuntimeError(
        "Patient leakage detected!"
    )

print(
    "Patient-fold integrity: PASS"
)

print(
    f"Unique patients: "
    f"{df['patient_id'].nunique()}"
)


# ============================================================
# 10. OUTER CV
# ============================================================

print("\n[3/8] Starting nested patient-wise CV...")

outer_folds = sorted(
    folds.unique()
)

oof_pred = np.empty(
    len(df),
    dtype=object,
)

oof_proba = np.zeros(
    (
        len(df),
        len(CLASSES),
    ),
    dtype=np.float32,
)

outer_results = []

inner_results = []


# ============================================================
# OUTER FOLD LOOP
# ============================================================

for outer_test_fold in outer_folds:

    print("\n")
    print("=" * 80)
    print(
        f"OUTER TEST FOLD "
        f"{outer_test_fold}"
    )
    print("=" * 80)

    outer_train_mask = (
        folds != outer_test_fold
    )

    outer_test_mask = (
        folds == outer_test_fold
    )

    outer_train_idx = np.where(
        outer_train_mask
    )[0]

    outer_test_idx = np.where(
        outer_test_mask
    )[0]

    X_outer_train = X.iloc[
        outer_train_idx
    ]

    y_outer_train = y.iloc[
        outer_train_idx
    ]

    X_outer_test = X.iloc[
        outer_test_idx
    ]

    y_outer_test = y.iloc[
        outer_test_idx
    ]

    outer_train_patients = set(
        df.iloc[
            outer_train_idx
        ]["patient_id"]
    )

    outer_test_patients = set(
        df.iloc[
            outer_test_idx
        ]["patient_id"]
    )

    overlap = (
        outer_train_patients
        & outer_test_patients
    )

    if overlap:

        raise RuntimeError(
            f"Outer patient leakage: "
            f"{overlap}"
        )

    print(
        f"Outer training samples: "
        f"{len(outer_train_idx):,}"
    )

    print(
        f"Outer testing samples: "
        f"{len(outer_test_idx):,}"
    )

    print(
        f"Outer training patients: "
        f"{len(outer_train_patients)}"
    )

    print(
        f"Outer testing patients: "
        f"{len(outer_test_patients)}"
    )


    # ========================================================
    # INNER CV
    # ========================================================

    print(
        "\n[4/8] Running inner patient-wise CV..."
    )

    # Use the existing fold labels to create
    # inner splits from the outer training patients.
    inner_folds = sorted(
        set(
            folds.iloc[
                outer_train_idx
            ]
        )
    )

    config_scores = []


    for config_name, params in (
        PARAMETER_CONFIGS.items()
    ):

        print(
            f"\nTesting configuration: "
            f"{config_name}"
        )

        inner_balanced_scores = []
        inner_accuracy_scores = []
        inner_macro_f1_scores = []

        for inner_test_fold in inner_folds:

            # Inner test = one of the folds contained
            # inside the outer training data.
            inner_test_mask = (
                (folds == inner_test_fold)
                & outer_train_mask
            )

            inner_train_mask = (
                (folds != inner_test_fold)
                & outer_train_mask
            )

            inner_train_idx = np.where(
                inner_train_mask
            )[0]

            inner_test_idx = np.where(
                inner_test_mask
            )[0]

            X_inner_train = X.iloc[
                inner_train_idx
            ]

            y_inner_train = y.iloc[
                inner_train_idx
            ]

            X_inner_test = X.iloc[
                inner_test_idx
            ]

            y_inner_test = y.iloc[
                inner_test_idx
            ]

            # Verify patient separation
            inner_train_patients = set(
                df.iloc[
                    inner_train_idx
                ]["patient_id"]
            )

            inner_test_patients = set(
                df.iloc[
                    inner_test_idx
                ]["patient_id"]
            )

            inner_overlap = (
                inner_train_patients
                & inner_test_patients
            )

            if inner_overlap:

                raise RuntimeError(
                    "Inner patient leakage detected."
                )

            model = train_model(
                X_inner_train,
                y_inner_train,
                params,
            )

            inner_pred = model.predict(
                X_inner_test
            )

            inner_acc = accuracy_score(
                y_inner_test,
                inner_pred,
            )

            inner_bal = (
                balanced_accuracy_score(
                    y_inner_test,
                    inner_pred,
                )
            )

            inner_macro_f1 = (
                f1_score(
                    y_inner_test,
                    inner_pred,
                    labels=CLASSES,
                    average="macro",
                    zero_division=0,
                )
            )

            inner_accuracy_scores.append(
                inner_acc
            )

            inner_balanced_scores.append(
                inner_bal
            )

            inner_macro_f1_scores.append(
                inner_macro_f1
            )

        mean_acc = np.mean(
            inner_accuracy_scores
        )

        mean_bal = np.mean(
            inner_balanced_scores
        )

        mean_macro_f1 = np.mean(
            inner_macro_f1_scores
        )

        config_scores.append({
            "outer_test_fold":
                int(outer_test_fold),

            "configuration":
                config_name,

            "inner_accuracy":
                float(mean_acc),

            "inner_balanced_accuracy":
                float(mean_bal),

            "inner_macro_f1":
                float(mean_macro_f1),
        })

        print(
            f"Inner Accuracy: "
            f"{mean_acc:.4f}"
        )

        print(
            f"Inner Balanced Accuracy: "
            f"{mean_bal:.4f}"
        )

        print(
            f"Inner Macro F1: "
            f"{mean_macro_f1:.4f}"
        )


    # ========================================================
    # SELECT BEST CONFIGURATION
    # ========================================================

    config_df = pd.DataFrame(
        config_scores
    )

    best_row = (
        config_df
        .sort_values(
            [
                "inner_balanced_accuracy",
                "inner_macro_f1",
                "inner_accuracy",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    best_config_name = (
        best_row["configuration"]
    )

    best_params = (
        PARAMETER_CONFIGS[
            best_config_name
        ]
    )

    print("\n")
    print(
        "BEST INNER CONFIGURATION:"
    )

    print(
        best_config_name
    )

    print(
        f"Inner Balanced Accuracy: "
        f"{best_row['inner_balanced_accuracy']:.4f}"
    )

    inner_results.extend(
        config_scores
    )


    # ========================================================
    # TRAIN ON ALL OUTER TRAINING DATA
    # ========================================================

    print(
        "\n[5/8] Retraining selected model "
        "on complete outer training data..."
    )

    final_outer_model = train_model(
        X_outer_train,
        y_outer_train,
        best_params,
    )


    # ========================================================
    # OUTER TEST PREDICTION
    # ========================================================

    print(
        "Evaluating on untouched outer test patients..."
    )

    outer_pred = (
        final_outer_model.predict(
            X_outer_test
        )
    )

    outer_prob = (
        final_outer_model.predict_proba(
            X_outer_test
        )
    )

    oof_pred[
        outer_test_idx
    ] = outer_pred


    # Correctly map probability columns
    model_class_index = {
        cls: i
        for i, cls
        in enumerate(
            final_outer_model.classes_
        )
    }

    for local_i, global_i in enumerate(
        outer_test_idx
    ):

        for cls in CLASSES:

            oof_proba[
                global_i,
                CLASSES.index(cls)
            ] = outer_prob[
                local_i,
                model_class_index[cls]
            ]


    # ========================================================
    # OUTER METRICS
    # ========================================================

    outer_acc = accuracy_score(
        y_outer_test,
        outer_pred,
    )

    outer_bal = (
        balanced_accuracy_score(
            y_outer_test,
            outer_pred,
        )
    )

    outer_macro_precision = (
        precision_score(
            y_outer_test,
            outer_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )
    )

    outer_macro_recall = (
        recall_score(
            y_outer_test,
            outer_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )
    )

    outer_macro_f1 = (
        f1_score(
            y_outer_test,
            outer_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )
    )

    outer_weighted_f1 = (
        f1_score(
            y_outer_test,
            outer_pred,
            labels=CLASSES,
            average="weighted",
            zero_division=0,
        )
    )


    print("\nOUTER TEST RESULTS:")

    print(
        f"Accuracy: "
        f"{outer_acc:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{outer_bal:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{outer_macro_precision:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{outer_macro_recall:.4f}"
    )

    print(
        f"Macro F1: "
        f"{outer_macro_f1:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{outer_weighted_f1:.4f}"
    )


    outer_results.append({

        "outer_test_fold":
            int(outer_test_fold),

        "outer_train_samples":
            int(len(outer_train_idx)),

        "outer_test_samples":
            int(len(outer_test_idx)),

        "outer_train_patients":
            int(len(outer_train_patients)),

        "outer_test_patients":
            int(len(outer_test_patients)),

        "selected_configuration":
            best_config_name,

        "inner_balanced_accuracy":
            float(
                best_row[
                    "inner_balanced_accuracy"
                ]
            ),

        "outer_accuracy":
            float(outer_acc),

        "outer_balanced_accuracy":
            float(outer_bal),

        "outer_macro_precision":
            float(
                outer_macro_precision
            ),

        "outer_macro_recall":
            float(
                outer_macro_recall
            ),

        "outer_macro_f1":
            float(
                outer_macro_f1
            ),

        "outer_weighted_f1":
            float(
                outer_weighted_f1
            ),
    })


# ============================================================
# 11. FINAL NESTED OOF METRICS
# ============================================================

print("\n[6/8] Calculating final nested OOF metrics...")

nested_accuracy = accuracy_score(
    y,
    oof_pred,
)

nested_balanced_accuracy = (
    balanced_accuracy_score(
        y,
        oof_pred,
    )
)

nested_macro_precision = (
    precision_score(
        y,
        oof_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )
)

nested_macro_recall = (
    recall_score(
        y,
        oof_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )
)

nested_macro_f1 = (
    f1_score(
        y,
        oof_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )
)

nested_weighted_f1 = (
    f1_score(
        y,
        oof_pred,
        labels=CLASSES,
        average="weighted",
        zero_division=0,
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

nested_cm = confusion_matrix(
    y,
    oof_pred,
    labels=CLASSES,
)


# ============================================================
# 13. PER-CLASS METRICS
# ============================================================

class_precision = precision_score(
    y,
    oof_pred,
    labels=CLASSES,
    average=None,
    zero_division=0,
)

class_recall = recall_score(
    y,
    oof_pred,
    labels=CLASSES,
    average=None,
    zero_division=0,
)

class_f1 = f1_score(
    y,
    oof_pred,
    labels=CLASSES,
    average=None,
    zero_division=0,
)

class_support = (
    pd.Series(y)
    .value_counts()
    .reindex(CLASSES)
    .fillna(0)
)


class_metrics_df = pd.DataFrame({

    "class": CLASSES,

    "precision":
        class_precision,

    "recall":
        class_recall,

    "f1":
        class_f1,

    "support":
        class_support.values,
})


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

print("\n[7/8] Saving nested CV results...")

prediction_df = df[
    [
        "record",
        "patient_id",
        "sample",
        "class",
        "fold",
    ]
].copy()

prediction_df[
    "prediction"
] = oof_pred

for i, cls in enumerate(
    CLASSES
):

    prediction_df[
        f"prob_{cls}"
    ] = oof_proba[:, i]


PREDICTION_PATH = (
    RESULTS_DIR
    / "step47B_nested_oof_predictions.csv"
)

prediction_df.to_csv(
    PREDICTION_PATH,
    index=False,
)


# ============================================================
# 15. SAVE OUTER RESULTS
# ============================================================

outer_df = pd.DataFrame(
    outer_results
)

OUTER_PATH = (
    RESULTS_DIR
    / "step47B_outer_fold_metrics.csv"
)

outer_df.to_csv(
    OUTER_PATH,
    index=False,
)


# ============================================================
# 16. SAVE INNER SEARCH RESULTS
# ============================================================

inner_df = pd.DataFrame(
    inner_results
)

INNER_PATH = (
    RESULTS_DIR
    / "step47B_inner_search_results.csv"
)

inner_df.to_csv(
    INNER_PATH,
    index=False,
)


# ============================================================
# 17. SAVE CLASS METRICS
# ============================================================

CLASS_PATH = (
    RESULTS_DIR
    / "step47B_nested_class_metrics.csv"
)

class_metrics_df.to_csv(
    CLASS_PATH,
    index=False,
)


# ============================================================
# 18. SAVE CONFUSION MATRIX
# ============================================================

CM_PATH = (
    RESULTS_DIR
    / "step47B_nested_confusion_matrix.csv"
)

cm_df = pd.DataFrame(
    nested_cm,
    index=[
        f"Actual_{c}"
        for c in CLASSES
    ],
    columns=[
        f"Predicted_{c}"
        for c in CLASSES
    ],
)

cm_df.to_csv(
    CM_PATH
)


# ============================================================
# 19. SAVE FINAL JSON
# ============================================================

print(
    "\n[8/8] Saving final JSON..."
)

final_results = {

    "step": "47B",

    "evaluation": (
        "Nested patient-wise 3-fold "
        "cross-validation"
    ),

    "dataset": {

        "samples": int(
            len(df)
        ),

        "patients": int(
            df["patient_id"]
            .nunique()
        ),

        "features_used": len(
            FEATURES
        ),

        "folds": [
            int(x)
            for x in outer_folds
        ],
    },

    "features": FEATURES,

    "class_weights":
        CLASS_WEIGHTS,

    "inner_search_configurations":
        PARAMETER_CONFIGS,

    "outer_fold_results":
        outer_results,

    "final_nested_metrics": {

        "accuracy":
            float(nested_accuracy),

        "balanced_accuracy":
            float(
                nested_balanced_accuracy
            ),

        "macro_precision":
            float(
                nested_macro_precision
            ),

        "macro_recall":
            float(
                nested_macro_recall
            ),

        "macro_f1":
            float(
                nested_macro_f1
            ),

        "weighted_f1":
            float(
                nested_weighted_f1
            ),
    },

    "class_metrics":
        class_metrics_df.to_dict(
            orient="records"
        ),

    "confusion_matrix":
        nested_cm.tolist(),

    "outputs": {

        "predictions":
            str(PREDICTION_PATH),

        "outer_metrics":
            str(OUTER_PATH),

        "inner_search":
            str(INNER_PATH),

        "class_metrics":
            str(CLASS_PATH),

        "confusion_matrix":
            str(CM_PATH),
    },

    "methodological_note": (
        "Outer test patients were not used for "
        "inner hyperparameter selection. "
        "Each outer test fold was evaluated only "
        "after model selection within the outer "
        "training data."
    ),
}


JSON_PATH = (
    RESULTS_DIR
    / "step47B_nested_cv_metrics.json"
)

with open(
    JSON_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_results,
        f,
        indent=4,
    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

elapsed = time.time() - start_time

print("\n")
print("=" * 80)
print("STEP 47B COMPLETE")
print("=" * 80)

print("\nFINAL NESTED OOF PERFORMANCE")

print(
    f"Accuracy: "
    f"{nested_accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{nested_balanced_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision: "
    f"{nested_macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall: "
    f"{nested_macro_recall * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{nested_macro_f1 * 100:.2f}%"
)

print(
    f"Weighted F1: "
    f"{nested_weighted_f1 * 100:.2f}%"
)


print("\nPer-class metrics:")

print(
    class_metrics_df.round(4)
    .to_string(index=False)
)


print("\nOuter fold results:")

print(
    outer_df.round(4)
    .to_string(index=False)
)


print("\nSaved files:")

print(
    PREDICTION_PATH
)

print(
    OUTER_PATH
)

print(
    INNER_PATH
)

print(
    CLASS_PATH
)

print(
    CM_PATH
)

print(
    JSON_PATH
)

print(
    f"\nExecution time: "
    f"{elapsed / 60:.2f} minutes"
)

print(
    "\nNested patient-wise evaluation "
    "completed successfully."
)