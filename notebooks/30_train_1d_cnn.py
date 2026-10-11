"""
STEP 30 - 1D CNN ECG CLASSIFICATION

Uses normalized ECG beats and patient-wise 3-fold splits.

Important:
- Uses record + sample to match beats with folds.
- Does NOT assume all 109494 annotations have extracted beats.
- Uses the patient information already stored in normalized_ecg_beats.npz.
- Prevents patient leakage.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    ReLU,
    MaxPooling1D,
    Dropout,
    GlobalAveragePooling1D,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NPZ_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "normalized_ecg_beats.npz"
)

FOLD_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "patient_3folds.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 30 - 1D CNN ECG CLASSIFICATION")
print("=" * 70)


# ============================================================
# LOAD ECG DATA
# ============================================================

print("\nLoading normalized ECG beats...")

data = np.load(NPZ_PATH)

signals = data["signals"]
labels = data["labels"]
records = data["records"]
patients = data["patients"]
samples = data["samples"]

print("Signal shape:", signals.shape)
print("Labels shape:", labels.shape)
print("Records shape:", records.shape)
print("Patients shape:", patients.shape)
print("Samples shape:", samples.shape)


# ============================================================
# VALIDATION
# ============================================================

print("\nSignal validation:")

print("NaN values:", np.isnan(signals).sum())
print("Infinite values:", np.isinf(signals).sum())

assert len(signals) == len(labels)
assert len(signals) == len(records)
assert len(signals) == len(patients)
assert len(signals) == len(samples)

print("All arrays have matching length.")


# ============================================================
# CREATE BEAT METADATA
# ============================================================

print("\nCreating extracted-beat metadata...")

beat_df = pd.DataFrame({
    "record": records.astype(str),
    "sample": samples.astype(np.int64),
    "patient_id": patients.astype(str),
    "label": labels.astype(str)
})

print("Extracted beats:", len(beat_df))

print("\nUnique extracted patients:", beat_df["patient_id"].nunique())
print("Unique extracted records:", beat_df["record"].nunique())


# ============================================================
# LOAD PATIENT-WISE FOLDS
# ============================================================

print("\nLoading patient-wise fold information...")

fold_df = pd.read_csv(
    FOLD_PATH,
    dtype={
        "record": str,
        "patient_id": str,
        "annotation": str,
        "class": str
    }
)

print("Fold dataset shape:", fold_df.shape)

print("Fold columns:")
print(fold_df.columns.tolist())


# ============================================================
# NORMALIZE JOIN KEYS
# ============================================================

fold_df["record"] = fold_df["record"].astype(str)
fold_df["sample"] = fold_df["sample"].astype(np.int64)
fold_df["patient_id"] = fold_df["patient_id"].astype(str)


# ============================================================
# CREATE UNIQUE FOLD MAPPING
# ============================================================

print("\nCreating record + sample → fold mapping...")

fold_mapping = fold_df[
    ["record", "sample", "patient_id", "fold"]
].drop_duplicates(
    subset=["record", "sample"]
)

print("Unique fold mapping rows:", len(fold_mapping))


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicate_keys = fold_mapping.duplicated(
    subset=["record", "sample"]
).sum()

print("Duplicate record/sample keys:", duplicate_keys)

if duplicate_keys != 0:
    raise ValueError(
        "Duplicate record/sample keys found in fold mapping."
    )


# ============================================================
# MATCH EXTRACTED BEATS WITH FOLDS
# ============================================================

print("\nMatching extracted ECG beats with patient-wise folds...")

beat_df = beat_df.merge(
    fold_mapping,
    on=["record", "sample"],
    how="left",
    suffixes=("_npz", "_fold")
)

print("Matched rows:", len(beat_df))

missing_folds = beat_df["fold"].isna().sum()

print("Missing fold assignments:", missing_folds)

if missing_folds > 0:

    print("\nExamples of unmatched beats:")
    print(
        beat_df[
            beat_df["fold"].isna()
        ][
            ["record", "sample", "patient_id_npz", "label"]
        ].head(20)
    )

    raise ValueError(
        f"{missing_folds} extracted beats could not be matched "
        "to patient-wise folds."
    )


# ============================================================
# PATIENT CONSISTENCY CHECK
# ============================================================

print("\nChecking patient consistency...")

patient_mismatch = (
    beat_df["patient_id_npz"]
    != beat_df["patient_id_fold"]
)

print(
    "Patient ID mismatches:",
    patient_mismatch.sum()
)

if patient_mismatch.sum() > 0:

    print(
        beat_df[
            patient_mismatch
        ][
            [
                "record",
                "sample",
                "patient_id_npz",
                "patient_id_fold"
            ]
        ].head(20)
    )

    raise ValueError(
        "Patient mismatch detected between NPZ and fold dataset."
    )


# ============================================================
# FINAL PATIENT ID
# ============================================================

beat_df["patient_id"] = beat_df["patient_id_npz"]

beat_df["fold"] = beat_df["fold"].astype(int)


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

patient_fold_counts = (
    beat_df.groupby("patient_id")["fold"]
    .nunique()
)

leaking_patients = patient_fold_counts[
    patient_fold_counts > 1
]

print(
    "Patients appearing in multiple folds:",
    len(leaking_patients)
)

if len(leaking_patients) > 0:

    print(leaking_patients)

    raise ValueError(
        "PATIENT LEAKAGE DETECTED!"
    )

print(
    "SUCCESS: No patient appears in multiple folds."
)


# ============================================================
# VERIFY BEAT ORDER
# ============================================================

print("\nChecking metadata alignment...")

assert np.all(
    beat_df["label"].values == labels.astype(str)
)

print("Labels correctly aligned.")


# ============================================================
# CLASS INFORMATION
# ============================================================

classes = sorted(
    beat_df["label"].unique()
)

print("\nClasses:")
print(classes)

print("\nClass distribution:")
print(
    beat_df["label"].value_counts()
)


# ============================================================
# FOLD DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-WISE FOLD DISTRIBUTION")
print("=" * 70)

for fold in sorted(beat_df["fold"].unique()):

    subset = beat_df[
        beat_df["fold"] == fold
    ]

    print(f"\nFOLD {fold}")

    print(
        "Patients:",
        subset["patient_id"].nunique()
    )

    print(
        "Samples:",
        len(subset)
    )

    print(
        subset["label"].value_counts()
    )


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    beat_df["label"]
)

print("\nLabel mapping:")

for index, class_name in enumerate(
    label_encoder.classes_
):

    print(
        index,
        "→",
        class_name
    )


# ============================================================
# CNN INPUT SHAPE
# ============================================================

X = signals.astype(np.float32)

# Add channel dimension
X = X[..., np.newaxis]

print("\nCNN input shape:", X.shape)


# ============================================================
# MODEL FUNCTION
# ============================================================

def build_model(input_shape, num_classes):

    model = Sequential([

        Input(shape=input_shape),

        Conv1D(
            32,
            kernel_size=7,
            padding="same"
        ),
        BatchNormalization(),
        ReLU(),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            64,
            kernel_size=5,
            padding="same"
        ),
        BatchNormalization(),
        ReLU(),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            128,
            kernel_size=3,
            padding="same"
        ),
        BatchNormalization(),
        ReLU(),

        GlobalAveragePooling1D(),

        Dropout(0.30),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.30),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 3-FOLD PATIENT-WISE CNN
# ============================================================

all_predictions = []
fold_results = []

for test_fold in sorted(
    beat_df["fold"].unique()
):

    print("\n" + "=" * 70)
    print(f"TEST FOLD: {test_fold}")
    print("=" * 70)

    train_mask = (
        beat_df["fold"].values != test_fold
    )

    test_mask = (
        beat_df["fold"].values == test_fold
    )

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    print("\nTraining distribution:")

    print(
        beat_df.loc[
            train_mask,
            "label"
        ].value_counts()
    )

    print("\nTesting distribution:")

    print(
        beat_df.loc[
            test_mask,
            "label"
        ].value_counts()
    )


    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\nBuilding CNN...")

    model = build_model(
        input_shape=X_train.shape[1:],
        num_classes=len(classes)
    )

    model.summary()


    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTraining 1D CNN...")

    history = model.fit(
        X_train,
        y_train,

        validation_split=0.10,

        epochs=30,

        batch_size=256,

        callbacks=callbacks,

        verbose=1,

        shuffle=True
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nPredicting test fold...")

    probabilities = model.predict(
        X_test,
        batch_size=512,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
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
        round(balanced_accuracy, 4)
    )

    print(
        "Macro F1:",
        round(macro_f1, 4)
    )

    print(
        "Weighted F1:",
        round(weighted_f1, 4)
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(classes)),
            target_names=classes,
            zero_division=0
        )
    )


    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(classes))
    )

    print(cm)


    # --------------------------------------------------------
    # SAVE OOF PREDICTIONS
    # --------------------------------------------------------

    test_metadata = beat_df.loc[
        test_mask,
        [
            "record",
            "patient_id",
            "sample",
            "fold"
        ]
    ].copy()

    test_metadata["true_class"] = (
        label_encoder.inverse_transform(
            y_test
        )
    )

    test_metadata["predicted_class"] = (
        label_encoder.inverse_transform(
            predictions
        )
    )

    all_predictions.append(
        test_metadata
    )


    fold_results.append({

        "fold": test_fold,

        "accuracy": accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1
    })


    # --------------------------------------------------------
    # SAVE FOLD MODEL
    # --------------------------------------------------------

    model_path = os.path.join(
        RESULTS_DIR,
        f"step30_cnn_fold_{test_fold}.keras"
    )

    model.save(model_path)

    print(
        "\nSaved model:",
        model_path
    )


# ============================================================
# COMBINE OOF PREDICTIONS
# ============================================================

oof_df = pd.concat(
    all_predictions,
    ignore_index=True
)

oof_path = os.path.join(
    RESULTS_DIR,
    "step30_cnn_oof_predictions.csv"
)

oof_df.to_csv(
    oof_path,
    index=False
)


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

overall_true = label_encoder.transform(
    oof_df["true_class"]
)

overall_pred = label_encoder.transform(
    oof_df["predicted_class"]
)

overall_accuracy = accuracy_score(
    overall_true,
    overall_pred
)

overall_balanced_accuracy = (
    balanced_accuracy_score(
        overall_true,
        overall_pred
    )
)

overall_macro_f1 = f1_score(
    overall_true,
    overall_pred,
    average="macro",
    zero_division=0
)

overall_weighted_f1 = f1_score(
    overall_true,
    overall_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OVERALL 1D CNN RESULTS")
print("=" * 70)

print(
    "\nOverall Accuracy:",
    round(overall_accuracy, 4)
)

print(
    "Overall Balanced Accuracy:",
    round(overall_balanced_accuracy, 4)
)

print(
    "Overall Macro F1:",
    round(overall_macro_f1, 4)
)

print(
    "Overall Weighted F1:",
    round(overall_weighted_f1, 4)
)


print("\nOverall Classification Report:")

print(
    classification_report(
        overall_true,
        overall_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0
    )
)


print("\nOverall Confusion Matrix:")

print(
    confusion_matrix(
        overall_true,
        overall_pred,
        labels=np.arange(len(classes))
    )
)


# ============================================================
# FOLD SUMMARY
# ============================================================

fold_summary = pd.DataFrame(
    fold_results
)

print("\nFold Summary:")

print(fold_summary)

print(
    "\nMean Macro F1:",
    round(
        fold_summary["macro_f1"].mean(),
        4
    )
)

print(
    "Std Macro F1:",
    round(
        fold_summary["macro_f1"].std(),
        4
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "step30_cnn_results.csv"
)

fold_summary.to_csv(
    results_path,
    index=False
)


print("\n" + "=" * 70)
print("STEP 30 COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    "-",
    oof_path
)

print(
    "-",
    results_path
)

for fold in sorted(
    beat_df["fold"].unique()
):

    print(
        "- results/"
        f"step30_cnn_fold_{fold}.keras"
    )