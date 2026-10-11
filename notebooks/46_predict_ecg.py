"""
STEP 46 — FINAL ECG PREDICTION SYSTEM

Loads the final trained HGB model and predicts the ECG beat class
from the 23 frozen features.

IMPORTANT:
    Never assume probability column order.
    The model's actual class ordering is read from model.classes_.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "final_ecg_hgb_model.joblib"
)

FEATURE_PATH = (
    ROOT
    / "models"
    / "final_feature_list.json"
)


# ============================================================
# 2. EXPECTED CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "N",
    "S",
    "V",
    "F",
    "Q",
]


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("=" * 80)
print("STEP 46 — FINAL ECG PREDICTION SYSTEM")
print("=" * 80)

print("\n[1/5] Loading final model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print(
    f"Model loaded successfully:\n"
    f"{MODEL_PATH}"
)

print(
    f"\nModel classes: "
    f"{list(model.classes_)}"
)


# ============================================================
# 4. LOAD FEATURE LIST
# ============================================================

print("\n[2/5] Loading feature configuration...")

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Feature file not found:\n{FEATURE_PATH}"
    )

with open(
    FEATURE_PATH,
    "r",
    encoding="utf-8",
) as f:
    feature_config = json.load(f)


FEATURES = feature_config["features"]

print(
    f"Number of required features: "
    f"{len(FEATURES)}"
)

print("\nRequired features:")

for i, feature in enumerate(
    FEATURES,
    start=1,
):
    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# 5. VERIFY CLASSES
# ============================================================

print("\n[3/5] Verifying model classes...")

model_classes = list(model.classes_)

if set(model_classes) != set(
    EXPECTED_CLASSES
):

    raise ValueError(
        "Model classes do not match expected ECG classes.\n"
        f"Expected: {EXPECTED_CLASSES}\n"
        f"Found: {model_classes}"
    )

print(
    "Class verification successful."
)

print(
    "\nActual model class order:"
)

print(model_classes)

print(
    "\nThis order will be used to correctly map "
    "prediction probabilities."
)


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict_ecg(features):
    """
    Predict one ECG beat.

    Parameters
    ----------
    features : dict
        Dictionary containing all 23 required feature values.

    Returns
    -------
    dict
        Prediction and class probabilities.
    """

    # --------------------------------------------------------
    # Verify feature names
    # --------------------------------------------------------

    missing = [
        feature
        for feature in FEATURES
        if feature not in features
    ]

    if missing:

        raise ValueError(
            "Missing ECG features:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Create feature vector
    # --------------------------------------------------------

    feature_vector = np.array(
        [
            features[feature]
            for feature in FEATURES
        ],
        dtype=float,
    ).reshape(1, -1)

    # --------------------------------------------------------
    # Check invalid values
    # --------------------------------------------------------

    if not np.isfinite(
        feature_vector
    ).all():

        raise ValueError(
            "Feature vector contains "
            "NaN or infinite values."
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        feature_vector
    )[0]

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            feature_vector
        )[0]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Map probabilities using model.classes_
    # --------------------------------------------------------

    probability_dict = {}

    for class_name, probability in zip(
        model.classes_,
        probabilities,
    ):

        probability_dict[
            class_name
        ] = float(probability)

    # --------------------------------------------------------
    # Reorder only for display
    # --------------------------------------------------------

    ordered_probabilities = {

        class_name:
            probability_dict[class_name]

        for class_name in EXPECTED_CLASSES
    }

    return {

        "prediction": str(
            prediction
        ),

        "probabilities":
            ordered_probabilities,
    }


# ============================================================
# 7. DEMO PREDICTION USING FIRST DATASET SAMPLE
# ============================================================

print("\n[4/5] Running demonstration prediction...")

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enhanced_causal_ecg_features.csv"
)

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH,
    nrows=1,
    low_memory=False,
)

sample = {
    feature:
        float(df.iloc[0][feature])
    for feature in FEATURES
}

result = predict_ecg(
    sample
)


# ============================================================
# 8. DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 80)
print("ECG PREDICTION RESULT")
print("=" * 80)

print(
    f"\nPredicted ECG Class: "
    f"{result['prediction']}"
)

print("\nClass probabilities:")

for class_name, probability in (
    result["probabilities"].items()
):

    print(
        f"{class_name}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# 9. VERIFY PROBABILITY SUM
# ============================================================

probability_sum = sum(
    result["probabilities"].values()
)

print(
    f"\nProbability sum: "
    f"{probability_sum:.6f}"
)

if not np.isclose(
    probability_sum,
    1.0,
    atol=1e-5,
):

    raise RuntimeError(
        "Probability values do not sum to 1."
    )

print(
    "Probability verification: PASS"
)


# ============================================================
# 10. FINAL STATUS
# ============================================================

print("\n[5/5] Prediction system verification complete.")

print("\n" + "=" * 80)
print("STEP 46 COMPLETE")
print("=" * 80)

print(
    "\nThe final model can now be used for ECG prediction."
)

print(
    "\nModel:"
    "\n  final_ecg_hgb_model.joblib"
)

print(
    "\nFeatures:"
    f"\n  {len(FEATURES)} frozen features"
)

print(
    "\nClasses:"
    "\n  N, S, V, F, Q"
)

print(
    "\nProbability mapping:"
    "\n  Verified using model.classes_"
)

print(
    "\nNext step:"
    "\n  Build a user-facing ECG prediction application."
)