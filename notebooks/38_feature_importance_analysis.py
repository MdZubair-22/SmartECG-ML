"""
STEP 38 - FEATURE IMPORTANCE + CLASS SEPARATION ANALYSIS

Purpose:
1. Train the current Step 36 Balanced HGB model.
2. Calculate permutation importance.
3. Calculate class-wise feature differences.
4. Identify potentially weak/redundant features.
5. Analyze S/F confusion.

IMPORTANT:
- Uses patient-wise folds.
- Feature analysis is performed using training/validation data only.
- No test-fold information is used to select features.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


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
# LOAD DATA
# ============================================================

print("=" * 80)
print("STEP 38 - FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(f"\nDataset: {df.shape}")

feature_columns = [
    c for c in df.columns
    if c not in META_COLUMNS
]

X = df[feature_columns].copy()
y = df["class"].copy()

folds = df["fold"].astype(int).values

print(f"Features: {len(feature_columns)}")
print(f"Patients: {df['patient_id'].nunique()}")

print("\nClass distribution:")
print(
    y.value_counts()
    .reindex(CLASS_ORDER)
)


# ============================================================
# CHECK DATA QUALITY
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY")
print("=" * 80)

print("\nMissing values:")

missing = X.isna().sum()

print(
    missing[missing > 0]
)

print(
    f"\nTotal missing values: "
    f"{missing.sum()}"
)

print("\nInfinite values:")

numeric_array = X.to_numpy(dtype=float)

infinite_count = np.isinf(
    numeric_array
).sum()

print(
    f"Total infinite values: "
    f"{infinite_count}"
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE STATISTICS")
print("=" * 80)

feature_stats = []

for feature in feature_columns:

    values = X[feature].values

    feature_stats.append({
        "feature": feature,
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "unique_values": int(
            X[feature].nunique()
        )
    })

feature_stats_df = pd.DataFrame(
    feature_stats
)

print(
    feature_stats_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("HIGHLY CORRELATED FEATURE PAIRS")
print("=" * 80)

correlation = X.corr()

high_corr_pairs = []

for i in range(len(feature_columns)):

    for j in range(i + 1, len(feature_columns)):

        f1 = feature_columns[i]
        f2 = feature_columns[j]

        corr_value = correlation.loc[
            f1,
            f2
        ]

        if abs(corr_value) >= 0.90:

            high_corr_pairs.append({
                "feature_1": f1,
                "feature_2": f2,
                "correlation": float(corr_value)
            })

high_corr_df = pd.DataFrame(
    high_corr_pairs
)

if len(high_corr_df) > 0:

    print(
        high_corr_df.sort_values(
            "correlation",
            key=lambda x: abs(x),
            ascending=False
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

else:

    print(
        "No feature pairs with "
        "|correlation| >= 0.90"
    )


# ============================================================
# TRAIN/VALIDATION/TEST SETUP
# ============================================================

# We use:
#
# Fold 0 = final held-out test
# Fold 1 = validation
# Fold 2 = training
#
# This allows feature analysis without using
# the final test fold.

TRAIN_FOLD = 2
VALIDATION_FOLD = 1
TEST_FOLD = 0

train_mask = folds == TRAIN_FOLD
validation_mask = folds == VALIDATION_FOLD
test_mask = folds == TEST_FOLD

X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()

X_validation = X.loc[validation_mask].copy()
y_validation = y.loc[validation_mask].copy()

X_test = X.loc[test_mask].copy()
y_test = y.loc[test_mask].copy()

print("\n" + "=" * 80)
print("DATA SPLIT")
print("=" * 80)

print(
    f"Training fold:   {TRAIN_FOLD} "
    f"({len(X_train):,})"
)

print(
    f"Validation fold: {VALIDATION_FOLD} "
    f"({len(X_validation):,})"
)

print(
    f"Test fold:       {TEST_FOLD} "
    f"({len(X_test):,})"
)


# ============================================================
# TRAIN HGB
# ============================================================

print("\n" + "=" * 80)
print("TRAINING BALANCED HGB")
print("=" * 80)

sample_weights = np.array(
    [
        CLASS_WEIGHTS[str(label)]
        for label in y_train
    ],
    dtype=float
)

start = time.time()

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=RANDOM_STATE
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


# ============================================================
# VALIDATION PERFORMANCE
# ============================================================

validation_pred = model.predict(
    X_validation
)

validation_acc = accuracy_score(
    y_validation,
    validation_pred
)

validation_bal = balanced_accuracy_score(
    y_validation,
    validation_pred
)

validation_macro = f1_score(
    y_validation,
    validation_pred,
    labels=CLASS_ORDER,
    average="macro",
    zero_division=0
)

print("\nValidation performance:")

print(
    f"Accuracy:          {validation_acc:.4f}"
)

print(
    f"Balanced Accuracy: {validation_bal:.4f}"
)

print(
    f"Macro F1:          {validation_macro:.4f}"
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("PERMUTATION IMPORTANCE")
print("=" * 80)

print(
    "\nCalculating permutation importance "
    "on validation fold..."
)

start = time.time()

# Balanced accuracy is used because the dataset is highly
# imbalanced.

perm = permutation_importance(
    model,
    X_validation,
    y_validation,
    scoring="balanced_accuracy",
    n_repeats=3,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print(
    f"Completed in "
    f"{time.time() - start:.1f} sec"
)

importance_df = pd.DataFrame({

    "feature": feature_columns,

    "importance_mean":
        perm.importances_mean,

    "importance_std":
        perm.importances_std

})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False
).reset_index(drop=True)

importance_df["rank"] = (
    np.arange(len(importance_df)) + 1
)

print("\nFeature importance ranking:")

print(
    importance_df[
        [
            "rank",
            "feature",
            "importance_mean",
            "importance_std"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# CLASS-WISE FEATURE SEPARATION
# ============================================================

print("\n" + "=" * 80)
print("CLASS-WISE FEATURE SEPARATION")
print("=" * 80)

# Standardized mean difference:
#
# |mean(class) - mean(N)| / pooled_std
#
# This is NOT used for model selection.
# It is only exploratory analysis.

class_separation = []

reference_class = "N"

for feature in feature_columns:

    n_values = X_train.loc[
        y_train == reference_class,
        feature
    ].values

    n_mean = np.mean(n_values)
    n_std = np.std(n_values) + 1e-12

    row = {
        "feature": feature
    }

    for cls in ["S", "V", "F", "Q"]:

        cls_values = X_train.loc[
            y_train == cls,
            feature
        ].values

        cls_mean = np.mean(cls_values)
        cls_std = np.std(cls_values) + 1e-12

        pooled_std = np.sqrt(
            (
                n_std ** 2
                +
                cls_std ** 2
            ) / 2
        )

        effect = abs(
            cls_mean - n_mean
        ) / (
            pooled_std + 1e-12
        )

        row[
            f"{cls}_vs_N_effect"
        ] = float(effect)

    class_separation.append(row)

separation_df = pd.DataFrame(
    class_separation
)

for cls in ["S", "V", "F", "Q"]:

    print(
        f"\nTop features separating "
        f"{cls} from N:"
    )

    top = separation_df.sort_values(
        f"{cls}_vs_N_effect",
        ascending=False
    ).head(10)

    print(
        top[
            [
                "feature",
                f"{cls}_vs_N_effect"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# S vs F SEPARATION
# ============================================================

print("\n" + "=" * 80)
print("S VS F FEATURE SEPARATION")
print("=" * 80)

sf_rows = []

for feature in feature_columns:

    s_values = X_train.loc[
        y_train == "S",
        feature
    ].values

    f_values = X_train.loc[
        y_train == "F",
        feature
    ].values

    s_mean = np.mean(s_values)
    f_mean = np.mean(f_values)

    s_std = np.std(s_values) + 1e-12
    f_std = np.std(f_values) + 1e-12

    pooled_std = np.sqrt(
        (
            s_std ** 2
            +
            f_std ** 2
        ) / 2
    )

    effect = abs(
        s_mean - f_mean
    ) / (
        pooled_std + 1e-12
    )

    sf_rows.append({
        "feature": feature,
        "S_mean": float(s_mean),
        "F_mean": float(f_mean),
        "S_vs_F_effect": float(effect)
    })

sf_df = pd.DataFrame(
    sf_rows
).sort_values(
    "S_vs_F_effect",
    ascending=False
)

print(
    sf_df.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# FEATURE GROUP ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE GROUP IMPORTANCE")
print("=" * 80)

def get_group(feature):

    if feature.startswith("morph_"):
        return "morphology"

    if feature in [
        "prev_rr",
        "prev_hr"
    ]:
        return "rhythm"

    if feature in [
        "dominant_frequency",
        "spectral_entropy",
        "energy_ratio"
    ]:
        return "spectral"

    return "basic_signal"


importance_df["group"] = (
    importance_df["feature"]
    .apply(get_group)
)

group_importance = (
    importance_df
    .groupby("group")
    .agg(
        total_importance=(
            "importance_mean",
            "sum"
        ),
        mean_importance=(
            "importance_mean",
            "mean"
        ),
        feature_count=(
            "feature",
            "count"
        )
    )
    .reset_index()
)

print(
    group_importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# TOP / BOTTOM FEATURES
# ============================================================

print("\n" + "=" * 80)
print("TOP 15 FEATURES")
print("=" * 80)

print(
    importance_df.head(15)[
        [
            "rank",
            "feature",
            "importance_mean"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\n" + "=" * 80)
print("BOTTOM 15 FEATURES")
print("=" * 80)

print(
    importance_df.tail(15)[
        [
            "rank",
            "feature",
            "importance_mean"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

importance_path = (
    RESULTS_DIR
    / "step38_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


separation_path = (
    RESULTS_DIR
    / "step38_class_separation.csv"
)

separation_df.to_csv(
    separation_path,
    index=False
)


sf_path = (
    RESULTS_DIR
    / "step38_S_vs_F_separation.csv"
)

sf_df.to_csv(
    sf_path,
    index=False
)


group_path = (
    RESULTS_DIR
    / "step38_feature_group_importance.csv"
)

group_importance.to_csv(
    group_path,
    index=False
)


metrics_path = (
    RESULTS_DIR
    / "step38_analysis.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        {
            "dataset": str(DATA_PATH),
            "n_samples": int(len(df)),
            "n_features": int(len(feature_columns)),
            "features": feature_columns,
            "class_weights": CLASS_WEIGHTS,
            "train_fold": TRAIN_FOLD,
            "validation_fold": VALIDATION_FOLD,
            "test_fold": TEST_FOLD,
            "validation_metrics": {
                "accuracy":
                    float(validation_acc),
                "balanced_accuracy":
                    float(validation_bal),
                "macro_f1":
                    float(validation_macro)
            },
            "top_features":
                importance_df.head(15).to_dict(
                    orient="records"
                ),
            "feature_groups":
                group_importance.to_dict(
                    orient="records"
                )
        },
        f,
        indent=2
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("STEP 38 COMPLETED")
print("=" * 80)

print("\nSaved:")

print(importance_path)
print(separation_path)
print(sf_path)
print(group_path)
print(metrics_path)

print("\nNext:")
print(
    "Send the complete terminal output so we can identify "
    "which features are driving S/F/V/Q classification."
)