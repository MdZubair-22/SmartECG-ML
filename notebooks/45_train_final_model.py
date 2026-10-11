"""
STEP 45 — FINAL ECG HGB MODEL TRAINING

Purpose:
    Train and save the final frozen ECG classification model.

Model:
    HistGradientBoostingClassifier

Features:
    Fixed 23-feature subset selected in Step 39

Class weights:
    N = 1.0
    S = 3.0
    V = 1.0
    F = 5.0
    Q = 1.5

IMPORTANT:
    This model is intended as the final project model.
    Step 43/44 provided the post-selection patient-wise OOF
    evaluation. The OOF score should not be described as an
    independently unbiased test estimate.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enhanced_causal_ecg_features.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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
# FIXED 23 FEATURES
# ============================================================

FEATURES = [

    # -------------------------
    # Rhythm
    # -------------------------
    "prev_rr",
    "prev_hr",

    # -------------------------
    # Spectral
    # -------------------------
    "spectral_entropy",
    "dominant_frequency",
    "energy_ratio",

    # -------------------------
    # Morphology
    # -------------------------
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
# FINAL HGB PARAMETERS
# ============================================================

HGB_PARAMS = {

    "learning_rate": 0.08,

    "max_iter": 300,

    "max_leaf_nodes": 31,

    "min_samples_leaf": 20,

    "l2_regularization": 0.1,

    "random_state": RANDOM_STATE,
}


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
# START
# ============================================================

print("=" * 80)
print("STEP 45 — FINAL ECG HGB MODEL TRAINING")
print("=" * 80)

start_time = time.time()


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n[1/6] Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

print(f"Dataset shape: {df.shape}")

print(
    f"Number of samples: "
    f"{len(df):,}"
)

print(
    f"Number of patients: "
    f"{df['patient_id'].nunique()}"
)

print(
    f"Number of features used: "
    f"{len(FEATURES)}"
)


# ============================================================
# 4. VERIFY REQUIRED COLUMNS
# ============================================================

print("\n[2/6] Verifying dataset...")

required_columns = [
    "patient_id",
    "fold",
    "class",
] + FEATURES

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 5. PREPARE X AND y
# ============================================================

X = df[FEATURES].copy()

y = df["class"].astype(str)


# ============================================================
# VERIFY CLASSES
# ============================================================

print("\nClass distribution:")

class_distribution = (
    y.value_counts()
    .reindex(CLASSES)
)

print(class_distribution)


unexpected_classes = sorted(
    set(y.unique()) - set(CLASSES)
)

if unexpected_classes:

    raise ValueError(
        f"Unexpected classes found: "
        f"{unexpected_classes}"
    )


# ============================================================
# 6. DATA QUALITY
# ============================================================

print("\nChecking feature quality...")

missing_values = int(
    X.isna().sum().sum()
)

infinite_values = int(
    np.isinf(
        X.to_numpy()
    ).sum()
)

print(
    f"Missing values: "
    f"{missing_values}"
)

print(
    f"Infinite values: "
    f"{infinite_values}"
)

if missing_values > 0:

    raise ValueError(
        "Missing feature values detected."
    )

if infinite_values > 0:

    raise ValueError(
        "Infinite feature values detected."
    )


# ============================================================
# 7. CREATE SAMPLE WEIGHTS
# ============================================================

print("\n[3/6] Creating class-balanced sample weights...")

sample_weights = (
    y.map(CLASS_WEIGHTS)
    .to_numpy()
)

print("\nWeight distribution:")

for cls in CLASSES:

    count = int(
        (y == cls).sum()
    )

    print(
        f"{cls}: "
        f"{count:,} samples "
        f"× weight {CLASS_WEIGHTS[cls]}"
    )


# ============================================================
# 8. TRAIN FINAL MODEL
# ============================================================

print("\n[4/6] Training final HGB model...")

print("\nModel parameters:")

for key, value in HGB_PARAMS.items():

    print(
        f"{key}: {value}"
    )


model = HistGradientBoostingClassifier(
    **HGB_PARAMS
)

model.fit(
    X,
    y,
    sample_weight=sample_weights,
)

print("\nModel training completed.")


# ============================================================
# 9. VERIFY MODEL
# ============================================================

print("\n[5/6] Verifying trained model...")

print(
    f"Model classes: "
    f"{list(model.classes_)}"
)

if list(model.classes_) != CLASSES:

    print(
        "\nWARNING:"
        "\nModel class ordering differs from expected order."
    )


# Test prediction on a small sample

sample_X = X.iloc[:10]

sample_predictions = model.predict(
    sample_X
)

sample_probabilities = (
    model.predict_proba(sample_X)
)

print("\nSample predictions:")

for i, prediction in enumerate(
    sample_predictions
):

    print(
        f"Sample {i}: "
        f"{prediction}"
    )

print(
    "\nProbability matrix shape: "
    f"{sample_probabilities.shape}"
)


# ============================================================
# 10. SAVE MODEL
# ============================================================

print("\n[6/6] Saving final model...")

MODEL_PATH = (
    MODEL_DIR
    / "final_ecg_hgb_model.joblib"
)

joblib.dump(
    model,
    MODEL_PATH,
)

print(
    f"\nModel saved to:\n"
    f"{MODEL_PATH}"
)


# ============================================================
# 11. SAVE FEATURE LIST
# ============================================================

FEATURE_LIST_PATH = (
    MODEL_DIR
    / "final_feature_list.json"
)

feature_metadata = {

    "number_of_features": len(FEATURES),

    "features": FEATURES,

    "feature_groups": {

        "rhythm": [
            "prev_rr",
            "prev_hr",
        ],

        "spectral": [
            "spectral_entropy",
            "dominant_frequency",
            "energy_ratio",
        ],

        "morphology": [
            feature
            for feature in FEATURES
            if feature not in [
                "prev_rr",
                "prev_hr",
                "spectral_entropy",
                "dominant_frequency",
                "energy_ratio",
            ]
        ],
    },
}

with open(
    FEATURE_LIST_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        feature_metadata,
        f,
        indent=4,
    )


# ============================================================
# 12. SAVE MODEL CONFIGURATION
# ============================================================

CONFIG_PATH = (
    MODEL_DIR
    / "final_model_config.json"
)

model_config = {

    "project": "SmartECG-ML",

    "model_name": (
        "Final HistGradientBoosting "
        "ECG Beat Classifier"
    ),

    "model_type": (
        "HistGradientBoostingClassifier"
    ),

    "classes": CLASSES,

    "number_of_training_samples": int(
        len(df)
    ),

    "number_of_patients": int(
        df["patient_id"].nunique()
    ),

    "number_of_features": len(FEATURES),

    "features": FEATURES,

    "hyperparameters": HGB_PARAMS,

    "class_weights": CLASS_WEIGHTS,

    "training_method": (
        "Full available training dataset"
    ),

    "evaluation_reference": (
        "Step 43 patient-wise 3-fold OOF"
    ),

    "evaluation_note": (
        "Step 43 was a frozen post-selection OOF "
        "evaluation. Step 42 hyperparameters were "
        "selected using OOF results."
    ),
}

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        model_config,
        f,
        indent=4,
    )


# ============================================================
# 13. SAVE TRAINING SUMMARY
# ============================================================

SUMMARY_PATH = (
    MODEL_DIR
    / "final_model_training_summary.json"
)

summary = {

    "dataset": {
        "samples": int(len(df)),
        "patients": int(
            df["patient_id"].nunique()
        ),
        "columns": int(len(df.columns)),
    },

    "classes": {
        cls: int(
            (y == cls).sum()
        )
        for cls in CLASSES
    },

    "features_used": len(FEATURES),

    "model": HGB_PARAMS,

    "class_weights": CLASS_WEIGHTS,

    "saved_files": {
        "model": str(MODEL_PATH),
        "feature_list": str(
            FEATURE_LIST_PATH
        ),
        "configuration": str(
            CONFIG_PATH
        ),
    },

    "reference_metrics": {

        "accuracy": 0.8681749622926094,

        "balanced_accuracy": (
            0.6057902261952295
        ),

        "macro_f1": (
            0.5561225479068235
        ),

        "weighted_f1": (
            0.880464111638375
        ),
    },
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

elapsed = time.time() - start_time

print("\n")
print("=" * 80)
print("STEP 45 COMPLETE")
print("=" * 80)

print(
    f"\nFinal model: "
    f"HistGradientBoostingClassifier"
)

print(
    f"Features: "
    f"{len(FEATURES)}"
)

print(
    f"Training samples: "
    f"{len(df):,}"
)

print(
    f"Patients: "
    f"{df['patient_id'].nunique()}"
)

print("\nSaved files:")

print(
    f"1. {MODEL_PATH}"
)

print(
    f"2. {FEATURE_LIST_PATH}"
)

print(
    f"3. {CONFIG_PATH}"
)

print(
    f"4. {SUMMARY_PATH}"
)

print(
    f"\nReference Step 43 performance:"
)

print(
    "Accuracy          : 86.82%"
)

print(
    "Balanced Accuracy : 60.58%"
)

print(
    "Macro F1          : 55.61%"
)

print(
    "Weighted F1       : 88.05%"
)

print(
    f"\nExecution time: "
    f"{elapsed / 60:.2f} minutes"
)

print(
    "\nFinal model training completed successfully."
)