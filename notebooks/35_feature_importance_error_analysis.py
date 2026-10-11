"""
STEP 35 - FEATURE IMPORTANCE AND ERROR ANALYSIS
==============================================

Build the final RF feature-importance summary and review
model errors using the patient-wise OOF predictions from Step 34.
"""

import json
import os
from pathlib import Path

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


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_FILE = DATA_DIR / "enhanced_causal_ecg_features.csv"
PREDICTION_FILE = RESULTS_DIR / "step34_predictions.csv"
IMPORTANCE_FILE = RESULTS_DIR / "step35_feature_importance.csv"
ERROR_SUMMARY_FILE = RESULTS_DIR / "step35_error_summary.json"
ERROR_DETAILS_FILE = RESULTS_DIR / "step35_error_details.csv"

CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_INT = {label: idx for idx, label in enumerate(CLASSES)}
INT_TO_CLASS = {idx: label for label, idx in CLASS_TO_INT.items()}


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 35 - FEATURE IMPORTANCE AND ERROR ANALYSIS")
    print("=" * 70)

    print("\nLoading feature dataset...")
    df = pd.read_csv(FEATURE_FILE)
    print(f"Dataset shape: {df.shape}")

    required_columns = ["record", "patient_id", "sample", "class", "fold"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["class"] = df["class"].astype(str).str.strip()
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    if df["fold"].isna().any():
        raise ValueError("NaN values found in fold column.")
    df["fold"] = df["fold"].astype(int)

    print("\nPATIENT LEAKAGE CHECK")
    patient_fold_counts = df.groupby("patient_id")["fold"].nunique()
    leakage = patient_fold_counts[patient_fold_counts > 1]
    print(f"Patients appearing in multiple folds: {len(leakage)}")
    if len(leakage) > 0:
        print(leakage)
        raise ValueError("PATIENT LEAKAGE DETECTED.")
    print("SUCCESS: No patient appears in multiple folds.")

    excluded = {"sample", "record", "time_seconds", "annotation", "class", "patient_id", "fold"}
    feature_columns = [c for c in df.columns if c not in excluded]
    print(f"\nNumber of ML features: {len(feature_columns)}")

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(X).any():
        raise ValueError("NaN values found in feature matrix.")
    if np.isinf(X).any():
        raise ValueError("Infinite values found in feature matrix.")

    unknown_classes = sorted(set(df["class"].unique()) - set(CLASSES))
    if unknown_classes:
        raise ValueError(f"Unknown classes found: {unknown_classes}")

    y = df["class"].map(CLASS_TO_INT).to_numpy(dtype=np.int8)

    print("\nTraining final Random Forest on all data for feature importance...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\nTop feature importances:")
    print(importance_df.head(20).to_string(index=False, formatters={"importance": lambda x: f"{x:.6f}"}))

    importance_df.to_csv(IMPORTANCE_FILE, index=False)
    print(f"\nSaved feature importance: {IMPORTANCE_FILE}")

    print("\nLoading patient-wise prediction file...")
    pred_df = pd.read_csv(PREDICTION_FILE)
    required_pred_columns = ["sample", "record", "patient_id", "fold", "true_class", "model", "predicted_class"]
    missing_pred_columns = [c for c in required_pred_columns if c not in pred_df.columns]
    if missing_pred_columns:
        raise ValueError(f"Missing prediction columns: {missing_pred_columns}")

    pred_df["patient_id"] = pred_df["patient_id"].astype(str).str.strip()
    pred_df["fold"] = pd.to_numeric(pred_df["fold"], errors="coerce")
    if pred_df["fold"].isna().any():
        raise ValueError("NaN values found in prediction fold column.")
    pred_df["fold"] = pred_df["fold"].astype(int)

    print("\nPATIENT LEAKAGE CHECK ON PREDICTIONS")
    patient_fold_counts = pred_df.groupby("patient_id")["fold"].nunique()
    leakage = patient_fold_counts[patient_fold_counts > 1]
    print(f"Patients appearing in multiple folds: {len(leakage)}")
    if len(leakage) > 0:
        print(leakage)
        raise ValueError("PATIENT LEAKAGE DETECTED IN PREDICTIONS.")
    print("SUCCESS: No patient appears in multiple folds.")

    model_summaries = {}
    error_rows = []

    for model_name in sorted(pred_df["model"].unique()):
        model_rows = pred_df[pred_df["model"] == model_name].copy()
        y_true = model_rows["true_class"].astype(str).str.strip().tolist()
        y_pred = model_rows["predicted_class"].astype(str).str.strip().tolist()

        metrics = calculate_metrics(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        per_class = {}
        for class_name in CLASSES:
            per_class[class_name] = {
                "precision": float(report[class_name]["precision"]),
                "recall": float(report[class_name]["recall"]),
                "f1": float(report[class_name]["f1-score"]),
                "support": int(report[class_name]["support"]),
            }

        model_summaries[model_name] = {
            "metrics": metrics,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "n_samples": int(len(model_rows)),
        }

        errors = model_rows[model_rows["true_class"] != model_rows["predicted_class"]].copy()
        for _, row in errors.iterrows():
            error_rows.append({
                "model": model_name,
                "sample": row["sample"],
                "record": row["record"],
                "patient_id": row["patient_id"],
                "fold": int(row["fold"]),
                "true_class": row["true_class"],
                "predicted_class": row["predicted_class"],
            })

        print(f"\n=== {model_name} ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"Error count: {len(errors):,}")
        print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, digits=4, zero_division=0))

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(ERROR_DETAILS_FILE, index=False)

    summary = {
        "step": 35,
        "dataset": FEATURE_FILE.name,
        "prediction_file": PREDICTION_FILE.name,
        "patient_leakage": int(len(leakage)),
        "models": model_summaries,
        "top_feature_importance": importance_df.head(20).to_dict(orient="records"),
    }

    with open(ERROR_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved error details: {ERROR_DETAILS_FILE}")
    print(f"Saved summary: {ERROR_SUMMARY_FILE}")
    print("\nSTEP 35 COMPLETE")
