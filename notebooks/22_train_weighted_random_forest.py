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
# PATH
# ============================================================

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\causal_ecg_features.csv"

RESULT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\results\weighted_random_forest_results.csv"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("CLASS-WEIGHTED RANDOM FOREST")
print("=" * 70)

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "record": str,
        "patient_id": str
    }
)


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
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
    "prev_hr"
]


X = df[
    feature_columns
]

y = df["class"]

fold = df["fold"]


# ============================================================
# STORAGE
# ============================================================

all_true = []

all_pred = []

results = []


# ============================================================
# 3-FOLD EVALUATION
# ============================================================

for test_fold in [0, 1, 2]:

    print("\n" + "=" * 70)

    print(
        f"TEST FOLD: {test_fold}"
    )

    print("=" * 70)


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


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=200,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1

    )


    print(
        "\nTraining class-weighted Random Forest..."
    )


    model.fit(
        X_train,
        y_train
    )


    print(
        "Training complete."
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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


    print("\nResults:")

    print(
        "Accuracy:",
        round(accuracy, 4)
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


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

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

    print(cm)


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

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
# OVERALL
# ============================================================

print("\n" + "=" * 70)

print("OVERALL WEIGHTED RANDOM FOREST")

print("=" * 70)


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


print(
    "\nMean Macro F1:",
    round(
        results_df[
            "macro_f1"
        ].mean(),
        4
    )
)

print(
    "Std Macro F1:",
    round(
        results_df[
            "macro_f1"
        ].std(),
        4
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    RESULT_PATH,
    index=False
)


print(
    "\nSaved to:"
)

print(
    RESULT_PATH
)