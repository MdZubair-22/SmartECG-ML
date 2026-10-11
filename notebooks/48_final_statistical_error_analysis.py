"""
STEP 48 - FINAL STATISTICAL & ERROR ANALYSIS
SmartECG-ML

Purpose:
1. Analyze nested patient-wise CV predictions from Step 47B
2. Generate confusion matrices
3. Calculate per-class metrics
4. Calculate ROC-AUC and Average Precision (PR-AUC)
5. Calculate fold-wise Mean +/- SD
6. Analyze major classification errors
7. Analyze class imbalance
8. Compare previous model results
9. Generate report-ready CSV/JSON files
10. Generate visualization plots

IMPORTANT:
- This script does NOT retrain any model.
- Step 47B used nested patient-wise hyperparameter validation
  with the 23-feature set frozen from Step 39.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = (
    RESULTS_DIR / "step47B_nested_oof_predictions.csv"
)

FOLD_METRICS_FILE = (
    RESULTS_DIR / "step47B_outer_fold_metrics.csv"
)

NESTED_METRICS_FILE = (
    RESULTS_DIR / "step47B_nested_cv_metrics.json"
)

CLASS_METRICS_FILE = (
    RESULTS_DIR / "step47B_nested_class_metrics.csv"
)

CONFUSION_FILE = (
    RESULTS_DIR / "step47B_nested_confusion_matrix.csv"
)

# Previous model result files
STEP34_FILE = RESULTS_DIR / "step34_model_comparison.json"
STEP35_FILE = RESULTS_DIR / "step35_gradient_boosting_metrics.json"
STEP36_FILE = RESULTS_DIR / "step36_class_balanced_hgb_metrics.json"
STEP39_FILE = RESULTS_DIR / "step39_feature_ablation_metrics.json"
STEP42_FILE = RESULTS_DIR / "step42_hgb_hyperparameter_metrics.json"


# ============================================================
# CONSTANTS
# ============================================================

CLASS_ORDER = ["N", "S", "V", "F", "Q"]

print("=" * 75)
print("STEP 48 - FINAL STATISTICAL & ERROR ANALYSIS")
print("SmartECG-ML")
print("=" * 75)


# ============================================================
# CHECK FILES
# ============================================================

print("\n[1] Checking required files...")

required_files = [
    PREDICTIONS_FILE,
    FOLD_METRICS_FILE,
    NESTED_METRICS_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )
    print(f"  OK: {file_path.name}")


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2] Loading nested CV results...")

pred_df = pd.read_csv(PREDICTIONS_FILE)
fold_df = pd.read_csv(FOLD_METRICS_FILE)

with open(NESTED_METRICS_FILE, "r", encoding="utf-8") as f:
    nested_metrics = json.load(f)

print(f"  Prediction rows: {len(pred_df):,}")
print(f"  Prediction columns: {len(pred_df.columns)}")
print(f"  Outer folds: {fold_df.shape[0]}")


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print("\nPrediction columns:")

for col in pred_df.columns:
    print(f"  - {col}")


# ============================================================
# IDENTIFY TRUE/PREDICTED LABEL COLUMNS
# ============================================================

def find_column(df, candidates):
    """
    Find first matching column from candidate names.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


TRUE_COL = find_column(
    pred_df,
    [
        "true_class",
        "true_label",
        "y_true",
        "actual",
        "class",
        "label",
    ],
)

PRED_COL = find_column(
    pred_df,
    [
        "predicted_class",
        "predicted_label",
        "y_pred",
        "prediction",
        "pred",
    ],
)

FOLD_COL = find_column(
    pred_df,
    [
        "fold",
        "test_fold",
        "outer_fold",
    ],
)

if TRUE_COL is None:
    raise ValueError(
        "Could not find true-label column in prediction CSV."
    )

if PRED_COL is None:
    raise ValueError(
        "Could not find predicted-label column in prediction CSV."
    )

print(f"\nTrue label column: {TRUE_COL}")
print(f"Predicted label column: {PRED_COL}")

if FOLD_COL:
    print(f"Fold column: {FOLD_COL}")
else:
    print("Fold column: not found")


# ============================================================
# BASIC DATA CHECK
# ============================================================

y_true = pred_df[TRUE_COL].astype(str)
y_pred = pred_df[PRED_COL].astype(str)

print("\n[3] Basic prediction validation")

print(f"  True labels: {len(y_true):,}")
print(f"  Pred labels: {len(y_pred):,}")
print(f"  Labels match length: {len(y_true) == len(y_pred)}")

unknown_true = sorted(set(y_true) - set(CLASS_ORDER))
unknown_pred = sorted(set(y_pred) - set(CLASS_ORDER))

if unknown_true:
    print(f"WARNING: Unknown true classes: {unknown_true}")

if unknown_pred:
    print(f"WARNING: Unknown predicted classes: {unknown_pred}")


# ============================================================
# OVERALL METRICS
# ============================================================

print("\n[4] Overall nested OOF metrics")

accuracy = accuracy_score(y_true, y_pred)

balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

macro_precision = precision_score(
    y_true,
    y_pred,
    labels=CLASS_ORDER,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_true,
    y_pred,
    labels=CLASS_ORDER,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASS_ORDER,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASS_ORDER,
    average="weighted",
    zero_division=0,
)

overall_metrics = {
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_acc),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
}

print(f"  Accuracy           : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Balanced Accuracy  : {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
print(f"  Macro Precision    : {macro_precision:.4f}")
print(f"  Macro Recall       : {macro_recall:.4f}")
print(f"  Macro F1           : {macro_f1:.4f}")
print(f"  Weighted F1        : {weighted_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n[5] Per-class classification metrics")

report_dict = classification_report(
    y_true,
    y_pred,
    labels=CLASS_ORDER,
    output_dict=True,
    zero_division=0,
)

class_rows = []

for cls in CLASS_ORDER:
    row = {
        "class": cls,
        "precision": report_dict[cls]["precision"],
        "recall": report_dict[cls]["recall"],
        "f1": report_dict[cls]["f1-score"],
        "support": int(report_dict[cls]["support"]),
    }

    class_rows.append(row)

    print(
        f"  {cls}: "
        f"Precision={row['precision']:.4f}, "
        f"Recall={row['recall']:.4f}, "
        f"F1={row['f1']:.4f}, "
        f"Support={row['support']:,}"
    )

class_metrics_df = pd.DataFrame(class_rows)

class_metrics_df.to_csv(
    RESULTS_DIR / "step48_nested_per_class_metrics.csv",
    index=False,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n[6] Confusion matrix")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_ORDER,
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_ORDER,
    columns=CLASS_ORDER,
)

print("\nRows = Actual")
print("Columns = Predicted\n")
print(cm_df)

cm_df.to_csv(
    RESULTS_DIR / "step48_nested_confusion_matrix.csv"
)


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

row_sums = cm.sum(axis=1, keepdims=True)

cm_normalized = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0,
)

cm_norm_df = pd.DataFrame(
    cm_normalized,
    index=CLASS_ORDER,
    columns=CLASS_ORDER,
)

cm_norm_df.to_csv(
    RESULTS_DIR / "step48_nested_normalized_confusion_matrix.csv"
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

print("\n[7] Generating confusion matrix plots...")

fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(cm)

ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_yticks(range(len(CLASS_ORDER)))

ax.set_xticklabels(CLASS_ORDER)
ax.set_yticklabels(CLASS_ORDER)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_title(
    "Nested Patient-Wise CV Confusion Matrix"
)

for i in range(len(CLASS_ORDER)):
    for j in range(len(CLASS_ORDER)):
        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
        )

fig.colorbar(im, ax=ax)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "step48_nested_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# NORMALIZED CONFUSION MATRIX PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(cm_normalized)

ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_yticks(range(len(CLASS_ORDER)))

ax.set_xticklabels(CLASS_ORDER)
ax.set_yticklabels(CLASS_ORDER)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_title(
    "Normalized Nested Patient-Wise CV Confusion Matrix"
)

for i in range(len(CLASS_ORDER)):
    for j in range(len(CLASS_ORDER)):
        ax.text(
            j,
            i,
            f"{cm_normalized[i, j]*100:.1f}%",
            ha="center",
            va="center",
        )

fig.colorbar(im, ax=ax)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "step48_nested_normalized_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# ERROR ANALYSIS
# ============================================================

print("\n[8] Error analysis")

error_df = pred_df.loc[
    y_true.values != y_pred.values
].copy()

print(
    f"  Total incorrect predictions: "
    f"{len(error_df):,}"
)

print(
    f"  Total correct predictions: "
    f"{len(pred_df) - len(error_df):,}"
)


# ============================================================
# ERROR PAIRS
# ============================================================

error_pairs = (
    pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
    })
    .query("actual != predicted")
    .groupby(["actual", "predicted"])
    .size()
    .reset_index(name="errors")
    .sort_values("errors", ascending=False)
)

total_errors = len(error_pairs)

if total_errors > 0:
    error_pairs["percentage_of_errors"] = (
        error_pairs["errors"] /
        len(error_df)
        * 100
    )

    error_pairs["percentage_of_all_samples"] = (
        error_pairs["errors"] /
        len(pred_df)
        * 100
    )

print("\nMajor error pairs:")

for _, row in error_pairs.head(15).iterrows():

    print(
        f"  Actual {row['actual']} -> "
        f"Predicted {row['predicted']}: "
        f"{int(row['errors']):,} "
        f"({row['percentage_of_errors']:.2f}% of errors)"
    )

error_pairs.to_csv(
    RESULTS_DIR / "step48_error_pairs.csv",
    index=False,
)


# ============================================================
# PER-CLASS ERROR RATE
# ============================================================

print("\n[9] Per-class error analysis")

per_class_error = []

for cls in CLASS_ORDER:

    actual_count = int(
        (y_true == cls).sum()
    )

    correct_count = int(
        ((y_true == cls) &
         (y_pred == cls)).sum()
    )

    incorrect_count = (
        actual_count -
        correct_count
    )

    error_rate = (
        incorrect_count / actual_count
        if actual_count > 0
        else 0
    )

    per_class_error.append({
        "class": cls,
        "actual_samples": actual_count,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "error_rate": error_rate,
        "error_rate_percent": error_rate * 100,
    })

    print(
        f"  {cls}: "
        f"{incorrect_count:,}/{actual_count:,} "
        f"errors "
        f"({error_rate*100:.2f}%)"
    )

per_class_error_df = pd.DataFrame(
    per_class_error
)

per_class_error_df.to_csv(
    RESULTS_DIR / "step48_per_class_error_analysis.csv",
    index=False,
)


# ============================================================
# CLASS IMBALANCE ANALYSIS
# ============================================================

print("\n[10] Class imbalance analysis")

class_counts = (
    y_true.value_counts()
    .reindex(CLASS_ORDER)
    .fillna(0)
    .astype(int)
)

total_samples = class_counts.sum()

imbalance_rows = []

for cls in CLASS_ORDER:

    count = int(class_counts[cls])

    percentage = (
        count / total_samples * 100
        if total_samples > 0
        else 0
    )

    imbalance_rows.append({
        "class": cls,
        "samples": count,
        "percentage": percentage,
    })

    print(
        f"  {cls}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )

imbalance_df = pd.DataFrame(
    imbalance_rows
)

imbalance_df.to_csv(
    RESULTS_DIR / "step48_class_imbalance.csv",
    index=False,
)


# ============================================================
# CLASS IMBALANCE RATIO
# ============================================================

majority_count = class_counts.max()
minority_count = class_counts.min()

if minority_count > 0:
    imbalance_ratio = (
        majority_count / minority_count
    )
else:
    imbalance_ratio = None

print(
    f"\n  Majority/minority ratio: "
    f"{imbalance_ratio:.2f}:1"
    if imbalance_ratio is not None
    else "\n  Majority/minority ratio: undefined"
)


# ============================================================
# FOLD-WISE STATISTICS
# ============================================================

print("\n[11] Fold-wise Mean +/- SD")

# Handle different possible column names
fold_column_map = {}

for col in fold_df.columns:

    lower = col.lower()

    if "accuracy" in lower and "balanced" not in lower:
        fold_column_map["accuracy"] = col

    elif "balanced" in lower and "accuracy" in lower:
        fold_column_map["balanced_accuracy"] = col

    elif "macro" in lower and "f1" in lower:
        fold_column_map["macro_f1"] = col

    elif "weighted" in lower and "f1" in lower:
        fold_column_map["weighted_f1"] = col

    elif "macro" in lower and "precision" in lower:
        fold_column_map["macro_precision"] = col

    elif "macro" in lower and "recall" in lower:
        fold_column_map["macro_recall"] = col


fold_statistics = []

metric_names = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]

for metric in metric_names:

    if metric not in fold_column_map:
        print(
            f"  WARNING: {metric} not found "
            f"in fold metrics."
        )
        continue

    col = fold_column_map[metric]

    values = pd.to_numeric(
        fold_df[col],
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        continue

    mean_value = values.mean()

    std_value = values.std(
        ddof=1
    ) if len(values) > 1 else 0

    fold_statistics.append({
        "metric": metric,
        "mean": mean_value,
        "std": std_value,
        "mean_percent":
            mean_value * 100,
        "std_percent":
            std_value * 100,
    })

    print(
        f"  {metric:20s}: "
        f"{mean_value:.4f} +/- {std_value:.4f}"
    )

fold_stats_df = pd.DataFrame(
    fold_statistics
)

fold_stats_df.to_csv(
    RESULTS_DIR / "step48_fold_mean_sd.csv",
    index=False,
)


# ============================================================
# FOLD COMPARISON PLOT
# ============================================================

if not fold_df.empty:

    print("\n[12] Generating fold comparison plot...")

    available_metrics = []

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]:

        if metric in fold_column_map:
            available_metrics.append(metric)

    if available_metrics:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        x = np.arange(
            len(fold_df)
        )

        width = 0.8 / len(
            available_metrics
        )

        for idx, metric in enumerate(
            available_metrics
        ):

            values = pd.to_numeric(
                fold_df[
                    fold_column_map[metric]
                ],
                errors="coerce",
            )

            ax.bar(
                x + idx * width,
                values,
                width,
                label=metric,
            )

        ax.set_xticks(
            x + width *
            (len(available_metrics)-1) / 2
        )

        ax.set_xticklabels(
            [
                f"Fold {i}"
                for i in range(len(fold_df))
            ]
        )

        ax.set_ylabel("Score")
        ax.set_title(
            "Nested Patient-Wise CV Fold Performance"
        )

        ax.legend()

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR /
            "step48_fold_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# ROC / PR DATA
# ============================================================

print("\n[13] ROC-AUC / PR-AUC analysis")

# Identify probability columns.
# Expected possibilities:
# prob_N, probability_N, N_probability, etc.

probability_columns = {}

for cls in CLASS_ORDER:

    candidates = [
        f"prob_{cls}",
        f"probability_{cls}",
        f"{cls}_prob",
        f"{cls}_probability",
        f"p_{cls}",
    ]

    found = None

    for candidate in candidates:
        if candidate in pred_df.columns:
            found = candidate
            break

    if found:
        probability_columns[cls] = found


print("\nProbability columns detected:")

for cls, col in probability_columns.items():
    print(f"  {cls}: {col}")


roc_pr_rows = []

if len(probability_columns) == len(CLASS_ORDER):

    for cls in CLASS_ORDER:

        y_binary = (
            y_true == cls
        ).astype(int)

        probabilities = pd.to_numeric(
            pred_df[
                probability_columns[cls]
            ],
            errors="coerce",
        )

        valid = probabilities.notna()

        y_binary_valid = y_binary[
            valid
        ]

        probabilities_valid = probabilities[
            valid
        ]

        try:
            roc_auc = roc_auc_score(
                y_binary_valid,
                probabilities_valid,
            )
        except Exception:
            roc_auc = np.nan

        try:
            ap = average_precision_score(
                y_binary_valid,
                probabilities_valid,
            )
        except Exception:
            ap = np.nan

        roc_pr_rows.append({
            "class": cls,
            "roc_auc": roc_auc,
            "average_precision": ap,
        })

        print(
            f"  {cls}: "
            f"ROC-AUC={roc_auc:.4f}, "
            f"Average Precision={ap:.4f}"
        )

else:

    print(
        "\nWARNING: Probability columns were "
        "not all found."
    )

    print(
        "ROC-AUC / PR-AUC calculation skipped."
    )


roc_pr_df = pd.DataFrame(
    roc_pr_rows
)

roc_pr_df.to_csv(
    RESULTS_DIR /
    "step48_nested_roc_pr_metrics.csv",
    index=False,
)


# ============================================================
# ROC CURVES
# ============================================================

if len(probability_columns) == len(CLASS_ORDER):

    print("\n[14] Generating ROC curves...")

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    for cls in CLASS_ORDER:

        y_binary = (
            y_true == cls
        ).astype(int)

        probabilities = pd.to_numeric(
            pred_df[
                probability_columns[cls]
            ],
            errors="coerce",
        )

        valid = probabilities.notna()

        y_binary = y_binary[valid]
        probabilities = probabilities[valid]

        # Manual ROC calculation
        thresholds = np.sort(
            probabilities.unique()
        )[::-1]

        tpr_values = []
        fpr_values = []

        positives = (
            y_binary == 1
        ).sum()

        negatives = (
            y_binary == 0
        ).sum()

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            tp = (
                (predictions == 1) &
                (y_binary == 1)
            ).sum()

            fp = (
                (predictions == 1) &
                (y_binary == 0)
            ).sum()

            tpr = (
                tp / positives
                if positives > 0
                else 0
            )

            fpr = (
                fp / negatives
                if negatives > 0
                else 0
            )

            tpr_values.append(tpr)
            fpr_values.append(fpr)

        ax.plot(
            fpr_values,
            tpr_values,
            label=cls,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Nested Patient-Wise CV ROC Curves"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "step48_nested_roc_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n[15] Building model comparison table")


def safe_load_json(path):

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"  Could not load {path.name}: {e}"
        )

        return None


comparison_rows = []


def add_comparison(
    model_name,
    accuracy_value,
    balanced_value,
    macro_f1_value,
    weighted_f1_value,
):

    comparison_rows.append({
        "model": model_name,
        "accuracy": accuracy_value,
        "balanced_accuracy": balanced_value,
        "macro_f1": macro_f1_value,
        "weighted_f1": weighted_f1_value,
    })


# ------------------------------------------------------------
# Nested HGB
# ------------------------------------------------------------

add_comparison(
    "Nested Patient-Wise HGB",
    accuracy,
    balanced_acc,
    macro_f1,
    weighted_f1,
)


# ------------------------------------------------------------
# Step 42 optimized HGB
# ------------------------------------------------------------

step42 = safe_load_json(
    STEP42_FILE
)

if step42:

    # Try multiple JSON structures
    candidate = step42

    if isinstance(step42, dict):

        if "best_model" in step42:
            candidate = step42["best_model"]

        elif "best_result" in step42:
            candidate = step42["best_result"]

        elif "best" in step42:
            candidate = step42["best"]

    if isinstance(candidate, dict):

        acc_v = candidate.get(
            "accuracy",
            candidate.get("Accuracy")
        )

        bal_v = candidate.get(
            "balanced_accuracy",
            candidate.get("balanced_acc")
        )

        macro_v = candidate.get(
            "macro_f1",
            candidate.get("Macro F1")
        )

        weighted_v = candidate.get(
            "weighted_f1",
            candidate.get("Weighted F1")
        )

        if all(
            v is not None
            for v in [
                acc_v,
                bal_v,
                macro_v,
                weighted_v,
            ]
        ):

            add_comparison(
                "Optimized HGB",
                acc_v,
                bal_v,
                macro_v,
                weighted_v,
            )


# ------------------------------------------------------------
# Step 35 HGB
# ------------------------------------------------------------

step35 = safe_load_json(
    STEP35_FILE
)

if step35:

    # Search recursively for a likely HGB metric dictionary
    def recursive_find(obj, required_keys):

        if isinstance(obj, dict):

            keys_lower = {
                str(k).lower()
                for k in obj.keys()
            }

            if all(
                key.lower() in keys_lower
                for key in required_keys
            ):
                return obj

            for value in obj.values():

                result = recursive_find(
                    value,
                    required_keys,
                )

                if result is not None:
                    return result

        elif isinstance(obj, list):

            for item in obj:

                result = recursive_find(
                    item,
                    required_keys,
                )

                if result is not None:
                    return result

        return None

    candidate = recursive_find(
        step35,
        [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ],
    )

    if candidate:

        def get_case_insensitive(
            dictionary,
            key,
        ):

            for k, v in dictionary.items():

                if str(k).lower() == key.lower():
                    return v

            return None

        add_comparison(
            "Standard HGB",
            get_case_insensitive(
                candidate,
                "accuracy",
            ),
            get_case_insensitive(
                candidate,
                "balanced_accuracy",
            ),
            get_case_insensitive(
                candidate,
                "macro_f1",
            ),
            get_case_insensitive(
                candidate,
                "weighted_f1",
            ),
        )


# ------------------------------------------------------------
# Step 36 Balanced HGB
# ------------------------------------------------------------

step36 = safe_load_json(
    STEP36_FILE
)

if step36:

    candidate = recursive_find(
        step36,
        [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ],
    )

    if candidate:

        def get_ci(
            dictionary,
            key,
        ):

            for k, v in dictionary.items():

                if str(k).lower() == key.lower():
                    return v

            return None

        add_comparison(
            "Balanced HGB",
            get_ci(
                candidate,
                "accuracy",
            ),
            get_ci(
                candidate,
                "balanced_accuracy",
            ),
            get_ci(
                candidate,
                "macro_f1",
            ),
            get_ci(
                candidate,
                "weighted_f1",
            ),
        )


# ------------------------------------------------------------
# Save comparison
# ------------------------------------------------------------

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    RESULTS_DIR /
    "step48_model_comparison.csv",
    index=False,
)

print(
    "\nModel comparison:"
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# MODEL COMPARISON PLOT
# ============================================================

if not comparison_df.empty:

    print(
        "\n[16] Generating model comparison plot..."
    )

    plot_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    available = [
        m
        for m in plot_metrics
        if m in comparison_df.columns
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    x = np.arange(
        len(comparison_df)
    )

    width = 0.8 / len(available)

    for i, metric in enumerate(available):

        values = pd.to_numeric(
            comparison_df[metric],
            errors="coerce",
        )

        ax.bar(
            x + i * width,
            values,
            width,
            label=metric,
        )

    ax.set_xticks(
        x + width *
        (len(available)-1) / 2
    )

    ax.set_xticklabels(
        comparison_df["model"],
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("Score")

    ax.set_ylim(
        0,
        1,
    )

    ax.set_title(
        "SmartECG-ML Model Performance Comparison"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "step48_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FINAL REPORT SUMMARY
# ============================================================

print("\n[17] Creating final report summary...")


summary = {
    "project": "SmartECG-ML",
    "analysis": "Step 48 Final Statistical and Error Analysis",

    "validation_method":
        "Nested patient-wise hyperparameter validation "
        "using a frozen 23-feature set",

    "number_of_samples":
        int(len(pred_df)),

    "classes":
        CLASS_ORDER,

    "overall_metrics":
        overall_metrics,

    "class_metrics":
        class_rows,

    "class_distribution":
        imbalance_rows,

    "majority_minority_ratio":
        imbalance_ratio,

    "top_error_pairs":
        error_pairs.head(15).to_dict(
            orient="records"
        ),

    "fold_statistics":
        fold_statistics,

    "roc_pr_metrics":
        roc_pr_rows,

    "outputs": [
        "step48_nested_per_class_metrics.csv",
        "step48_nested_confusion_matrix.csv",
        "step48_nested_normalized_confusion_matrix.csv",
        "step48_nested_confusion_matrix.png",
        "step48_nested_normalized_confusion_matrix.png",
        "step48_error_pairs.csv",
        "step48_per_class_error_analysis.csv",
        "step48_class_imbalance.csv",
        "step48_fold_mean_sd.csv",
        "step48_fold_comparison.png",
        "step48_nested_roc_pr_metrics.csv",
        "step48_nested_roc_curves.png",
        "step48_model_comparison.csv",
        "step48_model_comparison.png",
    ],
}


summary_file = (
    RESULTS_DIR /
    "step48_final_statistical_analysis.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        default=float,
    )


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 48 COMPLETE")
print("=" * 75)

print("\nFINAL NESTED PATIENT-WISE RESULTS")
print("-" * 45)

print(
    f"Accuracy          : "
    f"{accuracy*100:.2f}%"
)

print(
    f"Balanced Accuracy : "
    f"{balanced_acc*100:.2f}%"
)

print(
    f"Macro Precision   : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall      : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1          : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1       : "
    f"{weighted_f1:.4f}"
)


print("\nPER-CLASS RESULTS")
print("-" * 45)

for row in class_rows:

    print(
        f"{row['class']}: "
        f"Precision={row['precision']:.4f}, "
        f"Recall={row['recall']:.4f}, "
        f"F1={row['f1']:.4f}"
    )


print("\nTOP CLASSIFICATION ERRORS")
print("-" * 45)

for _, row in error_pairs.head(10).iterrows():

    print(
        f"{row['actual']} -> "
        f"{row['predicted']}: "
        f"{int(row['errors']):,}"
    )


print("\nFOLD MEAN +/- SD")
print("-" * 45)

for row in fold_statistics:

    print(
        f"{row['metric']}: "
        f"{row['mean']:.4f} +/- "
        f"{row['std']:.4f}"
    )


print("\nOUTPUT DIRECTORY:")
print(RESULTS_DIR)

print("\nMain summary:")
print(summary_file)

print("\n" + "=" * 75)
print("READY FOR FINAL REPORT ANALYSIS")
print("=" * 75)