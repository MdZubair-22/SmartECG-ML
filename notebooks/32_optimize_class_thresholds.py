"""
STEP 32 - FAST CLASS-SPECIFIC THRESHOLD OPTIMIZATION

Purpose:
    Improve minority-class recognition using probability thresholds
    without changing patient-wise folds or the underlying Random Forest.

    This version keeps the original threshold strategy and objective,
    but significantly reduces the computational cost of the grid search.

Input:
    data/processed/enhanced_causal_ecg_features.csv

Outputs:
    results/step32_oof_probabilities.csv
    results/step32_threshold_results.csv
    results/step32_confusion_matrix.csv
    results/step32_classification_report.csv
    results/step32_best_thresholds.csv
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/enhanced_causal_ecg_features.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET = "class"
FOLDS = [0, 1, 2]

CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_INDEX = {
    class_name: i
    for i, class_name in enumerate(CLASSES)
}


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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
    "morph_range",
    "morph_mean",
    "morph_std",
    "morph_rms",
    "morph_pre10_mean",
    "morph_pre20_mean",
    "morph_pre30_mean",
    "morph_annotation_pre10_diff",
    "morph_annotation_pre20_diff",
    "morph_annotation_pre30_diff",
    "morph_slope10",
    "morph_slope20",
    "morph_slope30",
    "morph_max_positive_slope",
    "morph_max_negative_slope",
    "morph_mean_abs_slope",
    "morph_pre10_energy_ratio",
    "morph_pre20_energy_ratio",
    "morph_pre30_energy_ratio",
    "morph_zero_crossings",
    "morph_distance_to_peak",
    "morph_energy_asymmetry",
]


# ============================================================
# RANDOM FOREST SETTINGS
# ============================================================

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": None,
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
    }


def print_metrics(title, y_true, y_pred):

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

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

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    )

    print(cm_df)

    return metrics, cm_df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 32 - FAST CLASS-SPECIFIC THRESHOLD OPTIMIZATION")
print("=" * 70)

print("\nLoading enhanced dataset...")

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# NORMALIZE PATIENT IDS
# ============================================================

df["patient_id"] = (
    df["patient_id"]
    .astype(str)
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\nFeature validation...")

missing_features = [
    f
    for f in FEATURES
    if f not in df.columns
]

if missing_features:

    raise ValueError(
        f"Missing required features: "
        f"{missing_features}"
    )


nan_count = (
    df[FEATURES]
    .isna()
    .sum()
    .sum()
)

if nan_count > 0:

    raise ValueError(
        f"NaN values detected: {nan_count}"
    )


inf_count = np.isinf(
    df[FEATURES].to_numpy()
).sum()

if inf_count > 0:

    raise ValueError(
        f"Infinite values detected: {inf_count}"
    )


print("NaN values: 0")
print("Infinite values: 0")


# ============================================================
# PATIENT LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

patient_fold_counts = (
    df.groupby("patient_id")["fold"]
    .nunique()
)

leaking_patients = (
    patient_fold_counts[
        patient_fold_counts > 1
    ]
)

print(
    "Patients appearing in multiple folds: "
    f"{len(leaking_patients)}"
)

if len(leaking_patients) > 0:

    print(leaking_patients)

    raise ValueError(
        "PATIENT LEAKAGE DETECTED."
    )

print(
    "SUCCESS: No patient appears in multiple folds."
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    df[TARGET].value_counts()
)


# ============================================================
# GENERATE OOF PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING OUT-OF-FOLD PROBABILITIES")
print("=" * 70)

oof_probability_list = []


for fold in FOLDS:

    print("\n" + "-" * 70)
    print(f"TEST FOLD: {fold}")
    print("-" * 70)

    train_df = df[
        df["fold"] != fold
    ]

    test_df = df[
        df["fold"] == fold
    ]

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Testing samples:  {len(test_df)}"
    )

    print("\nTraining distribution:")

    print(
        train_df[TARGET]
        .value_counts()
    )

    print("\nTesting distribution:")

    print(
        test_df[TARGET]
        .value_counts()
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(
        "\nTraining Enhanced Random Forest..."
    )

    model = RandomForestClassifier(
        **RF_PARAMS
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    probabilities = (
        model.predict_proba(X_test)
    )

    model_classes = list(
        model.classes_
    )

    probability_df = pd.DataFrame(
        0.0,
        index=np.arange(
            len(test_df)
        ),
        columns=[
            f"prob_{c}"
            for c in CLASSES
        ]
    )

    for i, class_name in enumerate(
        model_classes
    ):

        probability_df[
            f"prob_{class_name}"
        ] = probabilities[:, i]

    fold_result = pd.DataFrame({

        "record":
            test_df["record"].values,

        "patient_id":
            test_df["patient_id"].values,

        "sample":
            test_df["sample"].values,

        "fold":
            test_df["fold"].values,

        "true_class":
            y_test.values,
    })

    fold_result = pd.concat(
        [
            fold_result.reset_index(
                drop=True
            ),
            probability_df.reset_index(
                drop=True
            ),
        ],
        axis=1
    )

    oof_probability_list.append(
        fold_result
    )

    # --------------------------------------------------------
    # Fold baseline
    # --------------------------------------------------------

    prob_matrix = (
        fold_result[
            [
                f"prob_{c}"
                for c in CLASSES
            ]
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    baseline_predictions = np.array(
        CLASSES
    )[
        np.argmax(
            prob_matrix,
            axis=1
        )
    ]

    fold_metrics = calculate_metrics(
        y_test,
        baseline_predictions
    )

    print("\nFold baseline results:")

    print(
        f"Accuracy:          "
        f"{fold_metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{fold_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{fold_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:        "
        f"{fold_metrics['weighted_f1']:.4f}"
    )


# ============================================================
# COMBINE OOF
# ============================================================

oof_df = pd.concat(
    oof_probability_list,
    ignore_index=True
)

print("\n" + "=" * 70)
print("OOF DATASET")
print("=" * 70)

print(
    f"Rows: {len(oof_df)}"
)

expected_rows = len(df)

if len(oof_df) != expected_rows:

    raise ValueError(
        f"OOF row mismatch: "
        f"{len(oof_df)} != "
        f"{expected_rows}"
    )

print(
    "SUCCESS: Every sample has an OOF prediction."
)


# ============================================================
# PROBABILITY MATRIX
# ============================================================

prob_columns = [
    f"prob_{c}"
    for c in CLASSES
]

probabilities = (
    oof_df[
        prob_columns
    ]
    .to_numpy(
        dtype=np.float32
    )
)


# ============================================================
# BASELINE PREDICTIONS
# ============================================================

baseline_code = np.argmax(
    probabilities,
    axis=1
)

baseline_predictions = np.array(
    CLASSES
)[baseline_code]

oof_df[
    "baseline_prediction"
] = baseline_predictions


baseline_metrics, baseline_cm = (
    print_metrics(
        "BASELINE OOF PERFORMANCE",
        oof_df["true_class"],
        oof_df["baseline_prediction"]
    )
)


# ============================================================
# CONVERT TRUE LABELS TO INTEGER CODES
# ============================================================

y_true_codes = (
    oof_df["true_class"]
    .map(CLASS_TO_INDEX)
    .to_numpy(
        dtype=np.int8
    )
)


# ============================================================
# BASELINE RECALLS
# ============================================================

baseline_confusion = (
    np.zeros(
        (
            len(CLASSES),
            len(CLASSES)
        ),
        dtype=np.int64
    )
)

np.add.at(
    baseline_confusion,
    (
        y_true_codes,
        baseline_code
    ),
    1
)

baseline_support = (
    baseline_confusion.sum(
        axis=1
    )
)

baseline_tp = np.diag(
    baseline_confusion
)

baseline_recall = np.divide(
    baseline_tp,
    baseline_support,
    out=np.zeros(
        len(CLASSES),
        dtype=float
    ),
    where=baseline_support > 0
)

baseline_n_recall = (
    baseline_recall[
        CLASS_TO_INDEX["N"]
    ]
)

baseline_v_recall = (
    baseline_recall[
        CLASS_TO_INDEX["V"]
    ]
)


# ============================================================
# THRESHOLD STRATEGY
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

print(
    """
Optimization objective:
    0.60 * Macro F1
    + 0.40 * Balanced Accuracy

Protection:
    N recall should not collapse.
    V recall should remain strong.

Prediction rule:
    adjusted_score[class] =
        probability[class] / threshold[class]
"""
)


# ============================================================
# BASELINE THRESHOLDS
# ============================================================

baseline_thresholds = {

    "N": 1.00,
    "S": 1.00,
    "V": 1.00,
    "F": 1.00,
    "Q": 1.00,
}


# ============================================================
# SEARCH SPACE
# ============================================================

threshold_values = {

    "N": np.array([
        0.95,
        1.00,
        1.05,
        1.10,
        1.15,
    ]),

    "S": np.array([
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]),

    "V": np.array([
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
    ]),

    "F": np.array([
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]),

    "Q": np.array([
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]),
}


# ============================================================
# FAST THRESHOLD PREDICTION
# ============================================================

def threshold_predict_codes(
    probabilities,
    thresholds
):
    """
    Fast integer prediction.

    Avoids creating the complete adjusted
    matrix for every threshold combination.
    """

    best_scores = (
        probabilities[:, 0]
        / thresholds[0]
    )

    best_codes = np.zeros(
        len(probabilities),
        dtype=np.int8
    )

    for class_idx in range(
        1,
        len(CLASSES)
    ):

        scores = (
            probabilities[:, class_idx]
            / thresholds[class_idx]
        )

        mask = (
            scores > best_scores
        )

        best_scores[mask] = (
            scores[mask]
        )

        best_codes[mask] = (
            class_idx
        )

    return best_codes


# ============================================================
# FAST METRIC CALCULATION
# ============================================================

def evaluate_codes(
    pred_codes,
    y_true_codes,
    baseline_n_recall,
    baseline_v_recall
):

    confusion = np.zeros(
        (
            len(CLASSES),
            len(CLASSES)
        ),
        dtype=np.int64
    )

    np.add.at(
        confusion,
        (
            y_true_codes,
            pred_codes
        ),
        1
    )

    support = (
        confusion.sum(
            axis=1
        )
    )

    tp = np.diag(
        confusion
    )

    predicted_count = (
        confusion.sum(
            axis=0
        )
    )

    recall = np.divide(
        tp,
        support,
        out=np.zeros(
            len(CLASSES),
            dtype=float
        ),
        where=support > 0
    )

    precision = np.divide(
        tp,
        predicted_count,
        out=np.zeros(
            len(CLASSES),
            dtype=float
        ),
        where=predicted_count > 0
    )

    f1 = np.divide(
        2.0
        * precision
        * recall,

        precision
        + recall,

        out=np.zeros(
            len(CLASSES),
            dtype=float
        ),

        where=(
            precision
            + recall
        ) > 0
    )

    total = (
        confusion.sum()
    )

    accuracy = (
        tp.sum() / total
        if total > 0
        else 0.0
    )

    balanced_accuracy = (
        recall.mean()
    )

    macro_f1 = (
        f1.mean()
    )

    weighted_f1 = (
        np.sum(
            (support / total)
            * f1
        )
        if total > 0
        else 0.0
    )

    n_recall = recall[
        CLASS_TO_INDEX["N"]
    ]

    s_recall = recall[
        CLASS_TO_INDEX["S"]
    ]

    v_recall = recall[
        CLASS_TO_INDEX["V"]
    ]

    f_recall = recall[
        CLASS_TO_INDEX["F"]
    ]

    q_recall = recall[
        CLASS_TO_INDEX["Q"]
    ]

    # --------------------------------------------------------
    # Safety constraints
    # --------------------------------------------------------

    minimum_n_recall = max(
        0.85,
        baseline_n_recall - 0.05
    )

    minimum_v_recall = max(
        0.80,
        baseline_v_recall - 0.05
    )

    if (
        n_recall
        < minimum_n_recall
    ):
        return None

    if (
        v_recall
        < minimum_v_recall
    ):
        return None

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective = (
        0.60 * macro_f1
        + 0.40 * balanced_accuracy
    )

    return {

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "N_recall":
            n_recall,

        "S_recall":
            s_recall,

        "V_recall":
            v_recall,

        "F_recall":
            f_recall,

        "Q_recall":
            q_recall,

        "objective":
            objective,
    }


# ============================================================
# GRID SEARCH
# ============================================================

print(
    "\nSearching threshold combinations..."
)

results = []

combination_count = (
    len(threshold_values["N"])
    * len(threshold_values["S"])
    * len(threshold_values["V"])
    * len(threshold_values["F"])
    * len(threshold_values["Q"])
)

print(
    f"Total combinations: "
    f"{combination_count:,}"
)

print(
    "\nUsing optimized vectorized search..."
)


counter = 0

best_objective = -np.inf
best_result = None


# ------------------------------------------------------------
# Pre-create threshold arrays
# ------------------------------------------------------------

N_values = threshold_values["N"]
S_values = threshold_values["S"]
V_values = threshold_values["V"]
F_values = threshold_values["F"]
Q_values = threshold_values["Q"]


# ------------------------------------------------------------
# Search
# ------------------------------------------------------------

for n_threshold in N_values:

    for s_threshold in S_values:

        for v_threshold in V_values:

            for f_threshold in F_values:

                for q_threshold in Q_values:

                    counter += 1

                    thresholds = np.array(
                        [
                            n_threshold,
                            s_threshold,
                            v_threshold,
                            f_threshold,
                            q_threshold,
                        ],
                        dtype=np.float32
                    )

                    pred_codes = (
                        threshold_predict_codes(
                            probabilities,
                            thresholds
                        )
                    )

                    evaluation = (
                        evaluate_codes(
                            pred_codes,
                            y_true_codes,
                            baseline_n_recall,
                            baseline_v_recall
                        )
                    )

                    if evaluation is None:
                        continue

                    row = {

                        "N_threshold":
                            n_threshold,

                        "S_threshold":
                            s_threshold,

                        "V_threshold":
                            v_threshold,

                        "F_threshold":
                            f_threshold,

                        "Q_threshold":
                            q_threshold,
                    }

                    row.update(
                        evaluation
                    )

                    results.append(
                        row
                    )

                    if (
                        evaluation["objective"]
                        > best_objective
                    ):

                        best_objective = (
                            evaluation["objective"]
                        )

                        best_result = row

                    # Progress every 1000 combinations

                    if (
                        counter % 1000 == 0
                    ):

                        print(
                            f"Progress: "
                            f"{counter:,}/"
                            f"{combination_count:,} "
                            f"({100.0 * counter / combination_count:.1f}%) "
                            f"| Valid: "
                            f"{len(results):,} "
                            f"| Best objective: "
                            f"{best_objective:.5f}"
                        )


print(
    f"\nValid threshold combinations: "
    f"{len(results):,}"
)


if len(results) == 0:

    raise RuntimeError(
        "No threshold combination "
        "satisfied the safety constraints."
    )


# ============================================================
# RESULTS TABLE
# ============================================================

threshold_results = pd.DataFrame(
    results
)

threshold_results = (
    threshold_results
    .sort_values(
        by=[
            "macro_f1",
            "balanced_accuracy",
            "weighted_f1",
        ],
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# BEST THRESHOLDS
# ============================================================

best = (
    threshold_results.iloc[0]
)

best_thresholds = {

    "N":
        best["N_threshold"],

    "S":
        best["S_threshold"],

    "V":
        best["V_threshold"],

    "F":
        best["F_threshold"],

    "Q":
        best["Q_threshold"],
}


print("\n" + "=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)


for class_name in CLASSES:

    print(
        f"{class_name}: "
        f"{best_thresholds[class_name]:.3f}"
    )


# ============================================================
# BEST PREDICTIONS
# ============================================================

optimized_threshold_array = np.array(
    [
        best_thresholds[c]
        for c in CLASSES
    ],
    dtype=np.float32
)

optimized_codes = (
    threshold_predict_codes(
        probabilities,
        optimized_threshold_array
    )
)

optimized_predictions = np.array(
    CLASSES
)[optimized_codes]

oof_df[
    "optimized_prediction"
] = optimized_predictions


# ============================================================
# FINAL PERFORMANCE
# ============================================================

optimized_metrics, optimized_cm = (
    print_metrics(
        "OPTIMIZED THRESHOLD PERFORMANCE",
        oof_df["true_class"],
        oof_df["optimized_prediction"]
    )
)


# ============================================================
# BASELINE VS OPTIMIZED
# ============================================================

print("\n" + "=" * 70)
print("BASELINE vs OPTIMIZED")
print("=" * 70)


comparison = pd.DataFrame({

    "metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Weighted F1",
    ],

    "baseline": [

        baseline_metrics[
            "accuracy"
        ],

        baseline_metrics[
            "balanced_accuracy"
        ],

        baseline_metrics[
            "macro_f1"
        ],

        baseline_metrics[
            "weighted_f1"
        ],
    ],

    "optimized": [

        optimized_metrics[
            "accuracy"
        ],

        optimized_metrics[
            "balanced_accuracy"
        ],

        optimized_metrics[
            "macro_f1"
        ],

        optimized_metrics[
            "weighted_f1"
        ],
    ],
})


comparison[
    "improvement"
] = (
    comparison["optimized"]
    - comparison["baseline"]
)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# CLASS RECALL COMPARISON
# ============================================================

baseline_report = classification_report(
    oof_df["true_class"],
    oof_df["baseline_prediction"],
    labels=CLASSES,
    output_dict=True,
    zero_division=0
)

optimized_report = classification_report(
    oof_df["true_class"],
    oof_df["optimized_prediction"],
    labels=CLASSES,
    output_dict=True,
    zero_division=0
)


recall_comparison = pd.DataFrame({

    "class":
        CLASSES,

    "baseline_recall": [

        baseline_report[c][
            "recall"
        ]

        for c in CLASSES
    ],

    "optimized_recall": [

        optimized_report[c][
            "recall"
        ]

        for c in CLASSES
    ],
})


recall_comparison[
    "recall_improvement"
] = (
    recall_comparison[
        "optimized_recall"
    ]
    -
    recall_comparison[
        "baseline_recall"
    ]
)


print("\n" + "=" * 70)
print("CLASS RECALL COMPARISON")
print("=" * 70)


print(
    recall_comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE OOF PROBABILITIES
# ============================================================

oof_output = oof_df[
    [
        "record",
        "patient_id",
        "sample",
        "fold",
        "true_class",

        "prob_N",
        "prob_S",
        "prob_V",
        "prob_F",
        "prob_Q",

        "baseline_prediction",
        "optimized_prediction",
    ]
]


oof_output.to_csv(
    "results/step32_oof_probabilities.csv",
    index=False
)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_results.to_csv(
    "results/step32_threshold_results.csv",
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_combined = pd.DataFrame(

    optimized_cm,

    index=[
        f"true_{c}"
        for c in CLASSES
    ],

    columns=[
        f"pred_{c}"
        for c in CLASSES
    ],
)


cm_combined.to_csv(
    "results/step32_confusion_matrix.csv"
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(

    oof_df["true_class"],

    oof_df["optimized_prediction"],

    labels=CLASSES,

    output_dict=True,

    zero_division=0,
)


report_df = pd.DataFrame(
    report_dict
).transpose()


report_df.to_csv(
    "results/step32_classification_report.csv"
)


# ============================================================
# SAVE BEST THRESHOLDS
# ============================================================

best_threshold_df = pd.DataFrame({

    "class":
        CLASSES,

    "baseline_threshold": [

        baseline_thresholds[c]

        for c in CLASSES
    ],

    "optimized_threshold": [

        best_thresholds[c]

        for c in CLASSES
    ],
})


best_threshold_df.to_csv(
    "results/step32_best_thresholds.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 32 COMPLETE")
print("=" * 70)


print("\nBASELINE")

print(
    f"Accuracy:           "
    f"{baseline_metrics['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy:  "
    f"{baseline_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Macro F1:           "
    f"{baseline_metrics['macro_f1']:.4f}"
)

print(
    f"Weighted F1:        "
    f"{baseline_metrics['weighted_f1']:.4f}"
)


print("\nOPTIMIZED")

print(
    f"Accuracy:           "
    f"{optimized_metrics['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy:  "
    f"{optimized_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Macro F1:           "
    f"{optimized_metrics['macro_f1']:.4f}"
)

print(
    f"Weighted F1:        "
    f"{optimized_metrics['weighted_f1']:.4f}"
)


print("\nIMPROVEMENT")

print(
    f"Balanced Accuracy:  "
    f"{optimized_metrics['balanced_accuracy'] - baseline_metrics['balanced_accuracy']:+.4f}"
)

print(
    f"Macro F1:           "
    f"{optimized_metrics['macro_f1'] - baseline_metrics['macro_f1']:+.4f}"
)


print("\nBEST THRESHOLDS")

for class_name in CLASSES:

    print(
        f"{class_name}: "
        f"{best_thresholds[class_name]:.3f}"
    )


print("\nGenerated files:")

print(
    " - results/step32_oof_probabilities.csv"
)

print(
    " - results/step32_threshold_results.csv"
)

print(
    " - results/step32_confusion_matrix.csv"
)

print(
    " - results/step32_classification_report.csv"
)

print(
    " - results/step32_best_thresholds.csv"
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)