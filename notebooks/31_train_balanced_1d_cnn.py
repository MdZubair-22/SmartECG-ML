"""
STEP 31 - CLASS-BALANCED / FOCAL-LOSS 1D CNN
SmartECG-ML

Purpose:
Improve minority-class ECG detection while preserving
patient-wise cross-validation.

Classes:
N = Normal
S = Supraventricular ectopic
V = Ventricular ectopic
F = Fusion
Q = Unknown / other

Important:
- Patient-wise folds are preserved.
- No patient leakage.
- ECG waveform input remains 252 samples.
- Uses focal loss.
- Uses moderate class weighting.
- Generates out-of-fold predictions.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from tensorflow.keras import layers, models, callbacks


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data/processed"
RESULTS_DIR = "results"

SIGNAL_PATH = os.path.join(
    DATA_DIR,
    "normalized_ecg_beats.npz"
)

FOLD_PATH = os.path.join(
    DATA_DIR,
    "patient_3folds.csv"
)

OOF_PATH = os.path.join(
    RESULTS_DIR,
    "balanced_cnn_oof_predictions.csv"
)

RESULT_PATH = os.path.join(
    RESULTS_DIR,
    "balanced_cnn_results.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

WINDOW_SIZE = 252
NUM_CLASSES = 5

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

CLASS_TO_INDEX = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}

# Moderate weights.
# Do NOT use extreme weights for F.
CLASS_WEIGHTS = {
    0: 1.0,   # N
    1: 1.8,   # S
    2: 1.0,   # V
    3: 3.0,   # F
    4: 1.4,   # Q
}

# Focal loss parameters
FOCAL_GAMMA = 1.5

# Training
BATCH_SIZE = 256
EPOCHS = 30

LEARNING_RATE = 1e-3

PATIENCE = 6


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("STEP 31 - CLASS-BALANCED / FOCAL-LOSS 1D CNN")
print("=" * 70)


# ================================================================
# LOAD ECG SIGNALS
# ================================================================

print()
print("Loading normalized ECG beats...")

data = np.load(SIGNAL_PATH)

signals = data["signals"]
labels = data["labels"]
records = data["records"].astype(str)
patients = data["patients"].astype(str)
samples = data["samples"]

print("Signal shape:", signals.shape)
print("Labels shape:", labels.shape)

if signals.shape[1] != WINDOW_SIZE:
    raise ValueError(
        f"Expected ECG window of {WINDOW_SIZE}, "
        f"got {signals.shape[1]}"
    )


# ================================================================
# SIGNAL VALIDATION
# ================================================================

print()
print("Signal validation:")

print("NaN values:", np.isnan(signals).sum())
print("Infinite values:", np.isinf(signals).sum())

if np.isnan(signals).any():
    raise ValueError("Signals contain NaN values.")

if np.isinf(signals).any():
    raise ValueError("Signals contain infinite values.")


# ================================================================
# LOAD PATIENT-WISE FOLDS
# ================================================================

print()
print("Loading patient-wise fold information...")

fold_df = pd.read_csv(FOLD_PATH)

print("Fold dataset shape:", fold_df.shape)
print("Fold columns:", fold_df.columns.tolist())


required_columns = [
    "record",
    "sample",
    "patient_id",
    "fold",
]

for col in required_columns:
    if col not in fold_df.columns:
        raise ValueError(
            f"Required column missing from fold dataset: {col}"
        )


# ================================================================
# NORMALIZE KEY TYPES
# ================================================================

fold_df["record"] = fold_df["record"].astype(str)
fold_df["patient_id"] = fold_df["patient_id"].astype(str)
fold_df["sample"] = fold_df["sample"].astype(np.int64)

records = records.astype(str)
patients = patients.astype(str)
samples = samples.astype(np.int64)


# ================================================================
# CREATE ECG KEY
# ================================================================

print()
print("Matching ECG beats with patient-wise folds...")

signal_keys = pd.DataFrame({
    "record": records,
    "patient_id": patients,
    "sample": samples,
})

signal_keys["key"] = (
    signal_keys["record"]
    + "_"
    + signal_keys["patient_id"]
    + "_"
    + signal_keys["sample"].astype(str)
)

fold_df["key"] = (
    fold_df["record"]
    + "_"
    + fold_df["patient_id"]
    + "_"
    + fold_df["sample"].astype(str)
)


# ================================================================
# CHECK DUPLICATE KEYS
# ================================================================

signal_duplicates = signal_keys["key"].duplicated().sum()
fold_duplicates = fold_df["key"].duplicated().sum()

print("Signal duplicate keys:", signal_duplicates)
print("Fold duplicate keys:", fold_duplicates)

if signal_duplicates > 0:
    raise ValueError("Duplicate signal keys detected.")

if fold_duplicates > 0:
    raise ValueError("Duplicate fold keys detected.")


# ================================================================
# MAP FOLDS
# ================================================================

fold_mapping = fold_df.set_index("key")["fold"]

signal_keys["fold"] = signal_keys["key"].map(fold_mapping)

missing_folds = signal_keys["fold"].isna().sum()

print("Missing fold assignments:", missing_folds)

if missing_folds > 0:
    print()
    print("Example unmatched keys:")
    print(
        signal_keys[
            signal_keys["fold"].isna()
        ].head(10)
    )

    raise ValueError(
        "Some ECG beats could not be matched to folds."
    )

folds = signal_keys["fold"].astype(int).values


# ================================================================
# VERIFY PATIENT LEAKAGE
# ================================================================

print()
print("=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

patient_fold_check = pd.DataFrame({
    "patient_id": patients,
    "fold": folds,
})

patient_fold_counts = (
    patient_fold_check
    .groupby("patient_id")["fold"]
    .nunique()
)

leaking_patients = patient_fold_counts[
    patient_fold_counts > 1
]

print(
    "Patients in multiple folds:",
    len(leaking_patients)
)

if len(leaking_patients) > 0:
    print(leaking_patients)
    raise ValueError(
        "PATIENT LEAKAGE DETECTED."
    )

print(
    "SUCCESS: No patient appears in multiple folds."
)


# ================================================================
# LABEL VALIDATION
# ================================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

label_series = pd.Series(labels)

print(label_series.value_counts())


unknown_classes = set(labels) - set(CLASS_NAMES)

if unknown_classes:
    raise ValueError(
        f"Unknown ECG classes found: {unknown_classes}"
    )


# ================================================================
# CONVERT LABELS
# ================================================================

y = np.array([
    CLASS_TO_INDEX[label]
    for label in labels
], dtype=np.int64)


# ================================================================
# PREPARE CNN INPUT
# ================================================================

# CNN expects:
# samples × time × channels

X = signals.astype(np.float32)
X = X[..., np.newaxis]

print()
print("CNN input shape:", X.shape)
print("Label shape:", y.shape)


# ================================================================
# FOCAL LOSS
# ================================================================

def focal_loss(
    gamma=1.5,
    class_weights=None
):
    """
    Multi-class focal loss.

    Reduces contribution from easy examples
    and focuses training on difficult examples.
    """

    if class_weights is not None:
        class_weights_tensor = tf.constant(
            [
                class_weights[i]
                for i in range(NUM_CLASSES)
            ],
            dtype=tf.float32
        )
    else:
        class_weights_tensor = None

    def loss_function(y_true, y_pred):

        y_true = tf.cast(
            tf.reshape(y_true, [-1]),
            tf.int32
        )

        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )

        y_true_one_hot = tf.one_hot(
            y_true,
            depth=NUM_CLASSES
        )

        true_probability = tf.reduce_sum(
            y_true_one_hot * y_pred,
            axis=1
        )

        focal_factor = tf.pow(
            1.0 - true_probability,
            gamma
        )

        cross_entropy = -tf.math.log(
            true_probability
        )

        loss = (
            focal_factor *
            cross_entropy
        )

        if class_weights_tensor is not None:

            sample_weights = tf.gather(
                class_weights_tensor,
                y_true
            )

            loss = loss * sample_weights

        return tf.reduce_mean(loss)

    return loss_function


# ================================================================
# BUILD CNN
# ================================================================

def build_model():

    model = models.Sequential([
        
        layers.Input(
            shape=(WINDOW_SIZE, 1)
        ),

        # Block 1
        layers.Conv1D(
            32,
            kernel_size=7,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(
            pool_size=2
        ),
        layers.Dropout(0.20),

        # Block 2
        layers.Conv1D(
            64,
            kernel_size=5,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(
            pool_size=2
        ),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv1D(
            128,
            kernel_size=3,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(
            pool_size=2
        ),
        layers.Dropout(0.30),

        # Block 4
        layers.Conv1D(
            256,
            kernel_size=3,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.GlobalAveragePooling1D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.40),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        ),
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(
            gamma=FOCAL_GAMMA,
            class_weights=CLASS_WEIGHTS
        ),
        metrics=[
            "accuracy"
        ]
    )

    return model


# ================================================================
# TRAINING STORAGE
# ================================================================

oof_true = []
oof_pred = []
oof_record = []
oof_patient = []
oof_sample = []
oof_fold = []

fold_results = []


# ================================================================
# 3-FOLD PATIENT-WISE TRAINING
# ================================================================

for test_fold in [0, 1, 2]:

    print()
    print("=" * 70)
    print(f"TEST FOLD: {test_fold}")
    print("=" * 70)

    train_mask = folds != test_fold
    test_mask = folds == test_fold

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print()
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print()
    print("Training distribution:")

    train_labels = [
        CLASS_NAMES[i]
        for i in y_train
    ]

    print(
        pd.Series(train_labels)
        .value_counts()
    )

    print()
    print("Testing distribution:")

    test_labels = [
        CLASS_NAMES[i]
        for i in y_test
    ]

    print(
        pd.Series(test_labels)
        .value_counts()
    )

    # ------------------------------------------------------------
    # BUILD MODEL
    # ------------------------------------------------------------

    print()
    print("Building CNN...")

    model = build_model()

    print()
    model.summary()

    # ------------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------------

    fold_model_path = os.path.join(
        RESULTS_DIR,
        f"step31_best_model_fold_{test_fold}.keras"
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        fold_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    print()
    print("Training class-balanced focal-loss CNN...")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            reduce_lr,
            checkpoint
        ],
        shuffle=True,
        verbose=1
    )

    print()
    print("Training complete.")

    # ------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # ------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------

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

    print()
    print("Results:")
    print(
        f"Accuracy: {accuracy:.4f}"
    )
    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}"
    )
    print(
        f"Macro F1: {macro_f1:.4f}"
    )
    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    # ------------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------------

    print()
    print("Classification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=range(NUM_CLASSES),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # ------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=range(NUM_CLASSES)
    )

    print("Confusion Matrix:")
    print(cm)

    # ------------------------------------------------------------
    # STORE OOF
    # ------------------------------------------------------------

    test_indices = np.where(
        test_mask
    )[0]

    for local_index, original_index in enumerate(
        test_indices
    ):

        oof_record.append(
            records[original_index]
        )

        oof_patient.append(
            patients[original_index]
        )

        oof_sample.append(
            samples[original_index]
        )

        oof_fold.append(
            test_fold
        )

        oof_true.append(
            CLASS_NAMES[y_test[local_index]]
        )

        oof_pred.append(
            CLASS_NAMES[predictions[local_index]]
        )

    fold_results.append({
        "fold": test_fold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })

    # ------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------

    del model

    tf.keras.backend.clear_session()


# ================================================================
# OVERALL OOF RESULTS
# ================================================================

print()
print("=" * 70)
print("OVERALL STEP 31 RESULTS")
print("=" * 70)

oof_true_numeric = np.array([
    CLASS_TO_INDEX[x]
    for x in oof_true
])

oof_pred_numeric = np.array([
    CLASS_TO_INDEX[x]
    for x in oof_pred
])

overall_accuracy = accuracy_score(
    oof_true_numeric,
    oof_pred_numeric
)

overall_balanced_accuracy = (
    balanced_accuracy_score(
        oof_true_numeric,
        oof_pred_numeric
    )
)

overall_macro_f1 = f1_score(
    oof_true_numeric,
    oof_pred_numeric,
    average="macro",
    zero_division=0
)

overall_weighted_f1 = f1_score(
    oof_true_numeric,
    oof_pred_numeric,
    average="weighted",
    zero_division=0
)

print()
print(
    f"Overall Accuracy: "
    f"{overall_accuracy:.4f}"
)

print(
    f"Overall Balanced Accuracy: "
    f"{overall_balanced_accuracy:.4f}"
)

print(
    f"Overall Macro F1: "
    f"{overall_macro_f1:.4f}"
)

print(
    f"Overall Weighted F1: "
    f"{overall_weighted_f1:.4f}"
)


# ================================================================
# OVERALL CLASSIFICATION REPORT
# ================================================================

print()
print("Overall Classification Report:")

print(
    classification_report(
        oof_true_numeric,
        oof_pred_numeric,
        labels=range(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ================================================================
# OVERALL CONFUSION MATRIX
# ================================================================

overall_cm = confusion_matrix(
    oof_true_numeric,
    oof_pred_numeric,
    labels=range(NUM_CLASSES)
)

print()
print("Overall Confusion Matrix:")
print(overall_cm)


# ================================================================
# FOLD SUMMARY
# ================================================================

fold_df_results = pd.DataFrame(
    fold_results
)

print()
print("Fold Summary:")
print(fold_df_results)

print()
print(
    f"Mean Macro F1: "
    f"{fold_df_results['macro_f1'].mean():.4f}"
)

print(
    f"Std Macro F1: "
    f"{fold_df_results['macro_f1'].std():.4f}"
)


# ================================================================
# SAVE OOF PREDICTIONS
# ================================================================

oof_df = pd.DataFrame({
    "record": oof_record,
    "patient_id": oof_patient,
    "sample": oof_sample,
    "fold": oof_fold,
    "true_class": oof_true,
    "predicted_class": oof_pred
})

oof_df.to_csv(
    OOF_PATH,
    index=False
)


# ================================================================
# SAVE RESULTS
# ================================================================

results_rows = []

for row in fold_results:
    results_rows.append(row)

results_rows.append({
    "fold": "overall",
    "accuracy": overall_accuracy,
    "balanced_accuracy": overall_balanced_accuracy,
    "macro_f1": overall_macro_f1,
    "weighted_f1": overall_weighted_f1
})

results_df = pd.DataFrame(
    results_rows
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ================================================================
# FINAL VALIDATION
# ================================================================

print()
print("=" * 70)
print("STEP 31 VALIDATION")
print("=" * 70)

print(
    "OOF predictions:",
    len(oof_df)
)

print(
    "Original ECG beats:",
    len(signals)
)

if len(oof_df) != len(signals):
    raise ValueError(
        "OOF prediction count does not match ECG beat count."
    )

patient_fold_check = (
    oof_df
    .groupby("patient_id")["fold"]
    .nunique()
)

leakage = patient_fold_check[
    patient_fold_check > 1
]

print(
    "Patients appearing in multiple folds:",
    len(leakage)
)

if len(leakage) > 0:
    raise ValueError(
        "Patient leakage detected in OOF predictions."
    )

print(
    "SUCCESS: OOF predictions cover all ECG beats."
)

print(
    "SUCCESS: No patient leakage."
)


# ================================================================
# COMPLETE
# ================================================================

print()
print("=" * 70)
print("STEP 31 COMPLETE")
print("=" * 70)

print()
print("Generated files:")

print(
    f" - {OOF_PATH}"
)

print(
    f" - {RESULT_PATH}"
)

print(
    " - results/step31_best_model_fold_0.keras"
)

print(
    " - results/step31_best_model_fold_1.keras"
)

print(
    " - results/step31_best_model_fold_2.keras"
)