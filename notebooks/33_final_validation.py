"""
STEP 33 - PROPER PATIENT-WISE FINAL VALIDATION
================================================

Purpose:
    Evaluate the ECG classifier using patient-wise 3-fold CV
    while selecting class-specific thresholds WITHOUT using
    the held-out test fold.

Pipeline:
    1. Load enhanced ECG features
    2. Validate patient-wise 3-fold assignments
    3. For each fold:
         - Train Random Forest on the other two folds
         - Generate validation probabilities from one training-fold
         - Select thresholds using that validation fold only
         - Apply selected thresholds to the held-out fold
    4. Combine all held-out predictions
    5. Report unbiased patient-wise metrics
    6. Save predictions, thresholds and metrics

Classes:
    N, S, V, F, Q
"""

import json
import os
import warnings
from itertools import product

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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_FILE = os.path.join(DATA_DIR, "enhanced_causal_ecg_features.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "step33_final_predictions.csv")
THRESHOLD_FILE = os.path.join(RESULTS_DIR, "step33_fold_thresholds.json")
METRICS_FILE = os.path.join(RESULTS_DIR, "step33_final_metrics.json")

CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_INT = {label: idx for idx, label in enumerate(CLASSES)}
INT_TO_CLASS = {idx: label for label, idx in CLASS_TO_INT.items()}

N_FOLDS = 3
RANDOM_STATE = 42
N_ESTIMATORS = 300
THRESHOLD_RANGES = {
    "N": np.arange(0.95, 1.151, 0.05, dtype=np.float32),
    "S": np.arange(0.55, 1.001, 0.05, dtype=np.float32),
    "V": np.arange(0.90, 1.101, 0.05, dtype=np.float32),
    "F": np.arange(0.35, 1.001, 0.05, dtype=np.float32),
    "Q": np.arange(0.60, 1.001, 0.05, dtype=np.float32),
}
MACRO_WEIGHT = 0.60
BALANCED_WEIGHT = 0.40


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def apply_thresholds(probabilities, thresholds):
    threshold_array = np.array([thresholds[c] for c in CLASSES], dtype=np.float32)
    adjusted = probabilities / threshold_array
    return np.argmax(adjusted, axis=1)


def print_report(title, y_true, y_pred):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:        {metrics['weighted_f1']:.4f}")
    print()
    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            digits=4,
            zero_division=0,
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
    return metrics


def find_best_thresholds(y_true, probabilities, baseline_recall=None):
    if baseline_recall is None:
        baseline_pred = np.argmax(probabilities, axis=1)
        baseline_recall = {}
        for class_name, class_id in CLASS_TO_INT.items():
            mask = y_true == class_id
            if np.any(mask):
                baseline_recall[class_name] = np.mean(baseline_pred[mask] == class_id)
            else:
                baseline_recall[class_name] = 0.0

    min_n_recall = max(0.85, baseline_recall["N"] - 0.05)
    min_v_recall = max(0.80, baseline_recall["V"] - 0.05)

    print()
    print("Threshold protection:")
    print(f"Minimum N recall: {min_n_recall:.4f}")
    print(f"Minimum V recall: {min_v_recall:.4f}")

    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float32)

    threshold_values = [THRESHOLD_RANGES[c] for c in CLASSES]
    threshold_grid = list(product(*threshold_values))
    total_combinations = len(threshold_grid)
    print(f"Threshold combinations: {total_combinations:,}")

    best_objective = -np.inf
    best_thresholds = None
    best_metrics = None
    valid_count = 0
    count = 0
    progress_every = 1000

    for combination in threshold_grid:
        count += 1
        thresholds = {class_name: float(value) for class_name, value in zip(CLASSES, combination)}

        adjusted = np.empty((probabilities.shape[0], len(CLASSES)), dtype=np.float32)
        for class_idx, class_name in enumerate(CLASSES):
            adjusted[:, class_idx] = probabilities[:, class_idx] / thresholds[class_name]

        pred = np.argmax(adjusted, axis=1)

        n_mask = y_true == CLASS_TO_INT["N"]
        n_recall = np.mean(pred[n_mask] == CLASS_TO_INT["N"]) if np.any(n_mask) else 0.0
        if n_recall < min_n_recall:
            continue

        v_mask = y_true == CLASS_TO_INT["V"]
        v_recall = np.mean(pred[v_mask] == CLASS_TO_INT["V"]) if np.any(v_mask) else 0.0
        if v_recall < min_v_recall:
            continue

        valid_count += 1
        metrics = calculate_metrics(y_true, pred)
        objective = MACRO_WEIGHT * metrics["macro_f1"] + BALANCED_WEIGHT * metrics["balanced_accuracy"]

        if objective > best_objective:
            best_objective = objective
            best_thresholds = thresholds.copy()
            best_metrics = metrics.copy()

        if count % progress_every == 0:
            print(
                f"Progress: {count:,}/{total_combinations:,} "
                f"({100.0 * count / total_combinations:.1f}%) | "
                f"Valid: {valid_count:,} | Best objective: {best_objective:.5f}"
            )

    print()
    print(f"Valid combinations: {valid_count:,}")

    if best_thresholds is None:
        print("WARNING: No threshold combination passed the protection constraints.")
        best_thresholds = {c: 1.0 for c in CLASSES}
        best_metrics = calculate_metrics(y_true, apply_thresholds(probabilities, best_thresholds))

    print()
    print("Best validation thresholds:")
    for c in CLASSES:
        print(f"    {c}: {best_thresholds[c]:.3f}")
    print()
    print("Best validation metrics:")
    for key, value in best_metrics.items():
        print(f"    {key}: {value:.4f}")

    return best_thresholds, best_metrics


def atomic_json_write(path, payload):
    temp_path = path + ".tmp"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    if os.path.exists(path):
        os.remove(path)
    os.replace(temp_path, path)


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 33 - PROPER PATIENT-WISE FINAL VALIDATION")
    print("=" * 70)

    print()
    print("Loading enhanced dataset...")
    df = pd.read_csv(FEATURE_FILE)
    print(f"Dataset shape: {df.shape}")

    required_columns = ["record", "patient_id", "sample", "class", "fold"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print()
    print("Using patient_id and fold directly from enhanced_causal_ecg_features.csv")
    print()
    print("Fold distribution:")
    print(df["fold"].value_counts().sort_index())

    df["record"] = df["record"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["class"] = df["class"].astype(str).str.strip()
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    if df["fold"].isna().any():
        raise ValueError("NaN values found in fold column.")
    df["fold"] = df["fold"].astype(int)
    invalid_folds = sorted(set(df["fold"].unique()) - {0, 1, 2})
    if invalid_folds:
        raise ValueError(f"Invalid fold values: {invalid_folds}")

    print()
    print("=" * 70)
    print("PATIENT-FOLD CONSISTENCY CHECK")
    print("=" * 70)
    patient_fold_counts = df.groupby("patient_id")["fold"].nunique()
    leakage = patient_fold_counts[patient_fold_counts > 1]
    print(f"Patients appearing in multiple folds: {len(leakage)}")
    if len(leakage) > 0:
        print(leakage)
        raise ValueError("PATIENT LEAKAGE DETECTED.")
    print("SUCCESS: No patient appears in multiple folds.")
    print()
    print("Patients per fold:")
    print(df.groupby("fold")["patient_id"].nunique())

    if "sample" not in df.columns:
        raise ValueError("Feature dataset does not contain 'sample' column.")
    if "fold" not in df.columns:
        raise ValueError("Feature dataset does not contain 'fold' column.")
    if "class" not in df.columns:
        raise ValueError("Feature dataset does not contain 'class' column.")
    if "patient_id" not in df.columns:
        raise ValueError("Feature dataset does not contain 'patient_id' column.")

    df["sample"] = pd.to_numeric(df["sample"], errors="coerce")
    if df["sample"].isna().any():
        raise ValueError("Feature dataset contains invalid sample IDs.")

    excluded_columns = {"sample", "record", "time_seconds", "annotation", "class", "patient_id", "fold"}
    feature_columns = [c for c in df.columns if c not in excluded_columns]
    print(f"\nNumber of ML features: {len(feature_columns)}")
    print("Features:")
    for i, feature in enumerate(feature_columns, start=1):
        print(f"{i:2d}. {feature}")

    X_all = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    nan_count = X_all.isna().sum().sum()
    inf_count = np.isinf(X_all.to_numpy(dtype=np.float64)).sum()
    print(f"\nFeature validation:\nNaN values: {nan_count}\nInfinite values: {inf_count}")
    if nan_count > 0:
        raise ValueError("NaN values detected.")
    if inf_count > 0:
        raise ValueError("Infinite values detected.")

    df["class"] = df["class"].astype(str)
    unknown_classes = sorted(set(df["class"].unique()) - set(CLASSES))
    if unknown_classes:
        raise ValueError(f"Unknown classes found: {unknown_classes}")

    y_all = df["class"].map(CLASS_TO_INT).to_numpy(dtype=np.int8)

    print()
    print("=" * 70)
    print("PATIENT LEAKAGE CHECK")
    print("=" * 70)
    patient_fold_counts = df.groupby("patient_id")["fold"].nunique()
    leakage = patient_fold_counts[patient_fold_counts > 1]
    print(f"Patients appearing in multiple folds: {len(leakage)}")
    if len(leakage) > 0:
        print(leakage)
        raise ValueError("PATIENT LEAKAGE DETECTED.")
    print("SUCCESS: No patient appears in multiple folds.")

    final_predictions = np.full(len(df), -1, dtype=np.int8)
    final_probabilities = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    fold_thresholds = {}
    fold_metrics = {}

    print()
    print("=" * 70)
    print("PATIENT-WISE 3-FOLD VALIDATION")
    print("=" * 70)

    for test_fold in range(N_FOLDS):
        print()
        print("-" * 70)
        print(f"TEST FOLD: {test_fold}")
        print("-" * 70)

        test_mask = df["fold"].to_numpy() == test_fold
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        print(f"Training samples: {len(train_idx)}")
        print(f"Held-out samples: {len(test_idx)}")

        training_folds = [f for f in range(N_FOLDS) if f != test_fold]
        validation_fold = training_folds[0]
        model_train_folds = training_folds[1:]

        validation_idx = np.where(df["fold"].to_numpy() == validation_fold)[0]
        model_train_idx = np.where(np.isin(df["fold"].to_numpy(), model_train_folds))[0]

        X_test = X_all.iloc[test_idx].to_numpy(dtype=np.float32)
        y_test = y_all[test_idx]

        print()
        print("Training distribution:")
        print(pd.Series(y_all[train_idx]).map(INT_TO_CLASS).value_counts())
        print()
        print("Held-out distribution:")
        print(pd.Series(y_test).map(INT_TO_CLASS).value_counts())

        print()
        print(f"Threshold validation fold: {validation_fold}")
        print(f"Model-training folds: {model_train_folds}")

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_all.iloc[model_train_idx].to_numpy(dtype=np.float32), y_all[model_train_idx])
        print("Training complete.")

        validation_probabilities = model.predict_proba(X_all.iloc[validation_idx].to_numpy(dtype=np.float32))
        validation_full_prob = np.zeros((len(validation_idx), len(CLASSES)), dtype=np.float32)
        for column, class_id in enumerate(model.classes_):
            validation_full_prob[:, int(class_id)] = validation_probabilities[:, column]

        validation_baseline_pred = np.argmax(validation_full_prob, axis=1)
        validation_baseline_metrics = calculate_metrics(y_all[validation_idx], validation_baseline_pred)
        print()
        print("Threshold-validation baseline:")
        print(f"Accuracy: {validation_baseline_metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {validation_baseline_metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {validation_baseline_metrics['macro_f1']:.4f}")

        print()
        print("Selecting thresholds using validation fold ONLY...")
        thresholds, validation_threshold_metrics = find_best_thresholds(
            y_all[validation_idx],
            validation_full_prob,
        )

        fold_thresholds[str(test_fold)] = {
            "test_fold": int(test_fold),
            "validation_fold": int(validation_fold),
            "model_train_folds": [int(x) for x in model_train_folds],
            "thresholds": thresholds,
            "validation_metrics": validation_threshold_metrics,
        }

        test_probabilities = model.predict_proba(X_test)
        test_full_prob = np.zeros((len(test_idx), len(CLASSES)), dtype=np.float32)
        for column, class_id in enumerate(model.classes_):
            test_full_prob[:, int(class_id)] = test_probabilities[:, column]

        baseline_test_pred = np.argmax(test_full_prob, axis=1)
        baseline_test_metrics = calculate_metrics(y_test, baseline_test_pred)
        optimized_test_pred = apply_thresholds(test_full_prob, thresholds)
        optimized_test_metrics = calculate_metrics(y_test, optimized_test_pred)

        final_predictions[test_idx] = optimized_test_pred
        final_probabilities[test_idx] = test_full_prob

        print()
        print("HELD-OUT FOLD RESULTS")
        print(f"Baseline Accuracy: {baseline_test_metrics['accuracy']:.4f}")
        print(f"Optimized Accuracy: {optimized_test_metrics['accuracy']:.4f}")
        print(f"Baseline Balanced Accuracy: {baseline_test_metrics['balanced_accuracy']:.4f}")
        print(f"Optimized Balanced Accuracy: {optimized_test_metrics['balanced_accuracy']:.4f}")
        print(f"Baseline Macro F1: {baseline_test_metrics['macro_f1']:.4f}")
        print(f"Optimized Macro F1: {optimized_test_metrics['macro_f1']:.4f}")
        print()
        print("Selected thresholds:")
        for c in CLASSES:
            print(f"    {c}: {thresholds[c]:.3f}")

        fold_metrics[str(test_fold)] = {
            "baseline": baseline_test_metrics,
            "optimized": optimized_test_metrics,
        }

    print()
    print("=" * 70)
    print("FINAL OOF VALIDATION CHECK")
    print("=" * 70)
    missing_predictions = int(np.sum(final_predictions < 0))
    print(f"Missing predictions: {missing_predictions}")
    if missing_predictions > 0:
        raise ValueError("Some samples do not have predictions.")
    print("SUCCESS: Every sample has a strictly held-out prediction.")

    final_true = y_all
    final_baseline_pred = np.argmax(final_probabilities, axis=1)
    final_optimized_pred = final_predictions

    baseline_metrics = print_report(
        "FINAL PATIENT-WISE BASELINE PERFORMANCE",
        final_true,
        final_baseline_pred,
    )
    optimized_metrics = print_report(
        "FINAL PATIENT-WISE THRESHOLD PERFORMANCE",
        final_true,
        final_optimized_pred,
    )

    print()
    print("=" * 70)
    print("FINAL BASELINE vs OPTIMIZED")
    print("=" * 70)
    comparison = pd.DataFrame(
        {
            "metric": ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1"],
            "baseline": [
                baseline_metrics["accuracy"],
                baseline_metrics["balanced_accuracy"],
                baseline_metrics["macro_f1"],
                baseline_metrics["weighted_f1"],
            ],
            "optimized": [
                optimized_metrics["accuracy"],
                optimized_metrics["balanced_accuracy"],
                optimized_metrics["macro_f1"],
                optimized_metrics["weighted_f1"],
            ],
        }
    )
    comparison["difference"] = comparison["optimized"] - comparison["baseline"]
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print()
    print("=" * 70)
    print("FINAL CLASS RECALL")
    print("=" * 70)
    recall_rows = []
    for class_name, class_id in CLASS_TO_INT.items():
        mask = final_true == class_id
        baseline_recall = np.mean(final_baseline_pred[mask] == class_id)
        optimized_recall = np.mean(final_optimized_pred[mask] == class_id)
        recall_rows.append(
            {
                "class": class_name,
                "baseline_recall": baseline_recall,
                "optimized_recall": optimized_recall,
                "difference": optimized_recall - baseline_recall,
            }
        )
    print(pd.DataFrame(recall_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output_df = df[["sample", "record", "patient_id", "fold", "class"]].copy()
    output_df["true_label"] = output_df["class"]
    output_df["predicted_label"] = [INT_TO_CLASS[x] for x in final_optimized_pred]
    output_df["prob_N"] = final_probabilities[:, 0]
    output_df["prob_S"] = final_probabilities[:, 1]
    output_df["prob_V"] = final_probabilities[:, 2]
    output_df["prob_F"] = final_probabilities[:, 3]
    output_df["prob_Q"] = final_probabilities[:, 4]

    temp_output = OUTPUT_FILE + ".tmp"
    if os.path.exists(temp_output):
        os.remove(temp_output)
    chunk_size = 10000
    for start in range(0, len(output_df), chunk_size):
        end = min(start + chunk_size, len(output_df))
        output_df.iloc[start:end].to_csv(
            temp_output,
            mode="w" if start == 0 else "a",
            header=(start == 0),
            index=False,
        )
        print(f"Saved rows {start:,} - {end:,}")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    os.replace(temp_output, OUTPUT_FILE)
    print()
    print("Predictions saved:")
    print(OUTPUT_FILE)

    atomic_json_write(THRESHOLD_FILE, fold_thresholds)
    print("Fold thresholds saved:")
    print(THRESHOLD_FILE)

    final_results = {
        "step": 33,
        "classes": CLASSES,
        "n_samples": int(len(df)),
        "n_folds": N_FOLDS,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "max_depth": None,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": RANDOM_STATE,
        },
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "fold_metrics": fold_metrics,
        "fold_thresholds": fold_thresholds,
    }
    atomic_json_write(METRICS_FILE, final_results)
    print("Metrics saved:")
    print(METRICS_FILE)

    print()
    print("=" * 70)
    print("STEP 33 COMPLETE")
    print("=" * 70)
    print()
    print("Final files:")
    print(f"1. {OUTPUT_FILE}")
    print(f"2. {THRESHOLD_FILE}")
    print(f"3. {METRICS_FILE}")
    print()
    print("Final optimized metrics:")
    print(f"Accuracy:          {optimized_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {optimized_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {optimized_metrics['macro_f1']:.4f}")
    print(f"Weighted F1:       {optimized_metrics['weighted_f1']:.4f}")
    print()
    print("IMPORTANT:")
    print("These predictions are generated from patient-wise held-out folds.")
    print("Thresholds for each test fold were selected without using that test fold.")
    print()
    print("Next step: Step 34 - Model comparison.")
