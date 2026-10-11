import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\enhanced_causal_ecg_features.csv"

RESULT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\results\enhanced_random_forest_results.csv"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("ENHANCED CAUSAL + MORPHOLOGY RANDOM FOREST")
print("=" * 70)


df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "record": str,
        "patient_id": str
    }
)


print(
    "\nDataset shape:",
    df.shape
)


# ============================================================
# FEATURES
# ============================================================

feature_columns = [

    # --------------------------------------------------------
    # ORIGINAL CAUSAL FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MORPHOLOGY FEATURES
    # --------------------------------------------------------

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

    "morph_energy_asymmetry"
]


print(
    "\nNumber of ML features:",
    len(feature_columns)
)


X = df[
    feature_columns
]

y = df[
    "class"
]

fold = df[
    "fold"
]


# ============================================================
# VALIDATION
# ============================================================

print(
    "\nChecking feature validity..."
)


nan_count = (
    X.isna()
    .sum()
    .sum()
)


inf_count = np.isinf(
    X
    .select_dtypes(
        include=np.number
    )
).sum().sum()


print(
    "NaN values:",
    nan_count
)

print(
    "Infinite values:",
    inf_count
)


# ============================================================
# STORAGE
# ============================================================

all_true = []

all_pred = []

results = []

# Store out-of-fold predictions for Step 27
oof_predictions = []


# ============================================================
# PATIENT-WISE 3-FOLD
# ============================================================

for test_fold in [0, 1, 2]:

    print(
        "\n" + "=" * 70
    )

    print(
        f"TEST FOLD: {test_fold}"
    )

    print(
        "=" * 70
    )


    train_mask = (
        fold != test_fold
    )

    test_mask = (
        fold == test_fold
    )


    X_train = X.loc[
        train_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    y_test = y.loc[
        test_mask
    ]


    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print(
        "\nTraining distribution:"
    )

    print(
        y_train.value_counts()
    )


    print(
        "\nTesting distribution:"
    )

    print(
        y_test.value_counts()
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = RandomForestClassifier(

        n_estimators=200,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1

    )


    print(
        "\nTraining enhanced Random Forest..."
    )


    model.fit(
        X_train,
        y_train
    )


    print(
        "Training complete."
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = model.predict(
        X_test
    )


    # ========================================================
    # SAVE OUT-OF-FOLD PREDICTIONS
    # ========================================================

    fold_predictions = df.loc[
        test_mask,
        [
            "record",
            "patient_id",
            "sample",
            "fold"
        ]
    ].copy()

    fold_predictions["true_class"] = y_test.values
    fold_predictions["predicted_class"] = y_pred

    oof_predictions.append(
        fold_predictions
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred
        )
    )


    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )


    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    print(
        "\nResults:"
    )

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy,
            4
        )
    )

    print(
        "Macro F1:",
        round(
            macro_f1,
            4
        )
    )

    print(
        "Weighted F1:",
        round(
            weighted_f1,
            4
        )
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[
                "N",
                "S",
                "V",
                "F",
                "Q"
            ],
            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        y_test,

        y_pred,

        labels=[
            "N",
            "S",
            "V",
            "F",
            "Q"
        ]

    )


    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )


    # ========================================================
    # STORE
    # ========================================================

    all_true.extend(
        y_test.tolist()
    )

    all_pred.extend(
        y_pred.tolist()
    )


    results.append({

        "fold":
            test_fold,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1

    })


# ============================================================
# OVERALL RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "OVERALL ENHANCED MODEL"
)

print(
    "=" * 70
)


overall_accuracy = accuracy_score(
    all_true,
    all_pred
)


overall_balanced_accuracy = (
    balanced_accuracy_score(
        all_true,
        all_pred
    )
)


overall_macro_f1 = f1_score(
    all_true,
    all_pred,
    average="macro",
    zero_division=0
)


overall_weighted_f1 = f1_score(
    all_true,
    all_pred,
    average="weighted",
    zero_division=0
)


print(
    "\nOverall Accuracy:",
    round(
        overall_accuracy,
        4
    )
)


print(
    "Overall Balanced Accuracy:",
    round(
        overall_balanced_accuracy,
        4
    )
)


print(
    "Overall Macro F1:",
    round(
        overall_macro_f1,
        4
    )
)


print(
    "Overall Weighted F1:",
    round(
        overall_weighted_f1,
        4
    )
)


# ============================================================
# OVERALL REPORT
# ============================================================

print(
    "\nOverall Classification Report:"
)


print(
    classification_report(
        all_true,
        all_pred,
        labels=[
            "N",
            "S",
            "V",
            "F",
            "Q"
        ],
        zero_division=0
    )
)


# ============================================================
# OVERALL CONFUSION MATRIX
# ============================================================

print(
    "\nOverall Confusion Matrix:"
)


overall_cm = confusion_matrix(

    all_true,

    all_pred,

    labels=[
        "N",
        "S",
        "V",
        "F",
        "Q"
    ]

)


print(
    overall_cm
)


# ============================================================
# FOLD SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\nFold Summary:"
)


print(
    results_df
)


mean_macro_f1 = (
    results_df[
        "macro_f1"
    ].mean()
)


std_macro_f1 = (
    results_df[
        "macro_f1"
    ].std()
)


print(
    "\nMean Macro F1:",
    round(
        mean_macro_f1,
        4
    )
)


print(
    "Std Macro F1:",
    round(
        std_macro_f1,
        4
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(
    "\nCalculating feature importance..."
)


final_model = RandomForestClassifier(

    n_estimators=200,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


final_model.fit(
    X,
    y
)


importance = pd.DataFrame({

    "feature":
        feature_columns,

    "importance":
        final_model.feature_importances_

})


importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    "\nTop 20 Features:"
)


print(
    importance
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    RESULT_PATH,
    index=False
)


IMPORTANCE_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\results\enhanced_feature_importance.csv"


importance.to_csv(
    IMPORTANCE_PATH,
    index=False
)


# ============================================================
# SAVE OUT-OF-FOLD PREDICTIONS
# ============================================================

oof_df = pd.concat(
    oof_predictions,
    ignore_index=True
)

OOF_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\results\enhanced_oof_predictions.csv"

oof_df.to_csv(
    OOF_PATH,
    index=False
)

print(
    "\nOut-of-fold predictions saved to:"
)

print(
    OOF_PATH
)


print(
    "\nResults saved to:"
)


print(
    RESULT_PATH
)


print(
    IMPORTANCE_PATH
)