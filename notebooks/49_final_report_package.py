"""
STEP 49 - FINAL REPORT PACKAGE
SmartECG-ML

Creates report-ready:
1. Dataset summary
2. Methodology summary
3. Feature summary
4. Final model configuration
5. Nested CV results
6. Per-class results
7. Confusion matrix
8. Error analysis
9. Class imbalance analysis
10. Fold-wise statistics
11. ROC-AUC / Average Precision
12. Model comparison
13. Limitations
14. Future work
15. Final report text
16. Master JSON summary

No model training is performed.
"""

from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

OUTPUT = RESULTS / "final_report_package"
OUTPUT.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("STEP 49 - FINAL SMART ECG-ML REPORT PACKAGE")
print("=" * 80)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path):
    if not path.exists():
        print(f"WARNING: Missing {path.name}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    if not path.exists():
        print(f"WARNING: Missing {path.name}")
        return pd.DataFrame()

    return pd.read_csv(path)


# ============================================================
# LOAD STEP 47B / 48 RESULTS
# ============================================================

print("\n[1] Loading final validation results...")

nested = load_json(
    RESULTS / "step47B_nested_cv_metrics.json"
)

summary48 = load_json(
    RESULTS / "step48_final_statistical_analysis.json"
)

class_metrics = load_csv(
    RESULTS / "step48_nested_per_class_metrics.csv"
)

confusion = load_csv(
    RESULTS / "step48_nested_confusion_matrix.csv"
)

normalized_confusion = load_csv(
    RESULTS / "step48_nested_normalized_confusion_matrix.csv"
)

errors = load_csv(
    RESULTS / "step48_error_pairs.csv"
)

class_errors = load_csv(
    RESULTS / "step48_per_class_error_analysis.csv"
)

imbalance = load_csv(
    RESULTS / "step48_class_imbalance.csv"
)

fold_stats = load_csv(
    RESULTS / "step48_fold_mean_sd.csv"
)

roc_pr = load_csv(
    RESULTS / "step48_nested_roc_pr_metrics.csv"
)

model_comparison = load_csv(
    RESULTS / "step48_model_comparison.csv"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n[2] Building dataset summary...")

dataset_summary = {
    "dataset": "MIT-BIH based ECG beat dataset",
    "samples": 109395,
    "patients": 47,
    "records": 48,
    "classes": {
        "N": 90545,
        "S": 2779,
        "V": 7234,
        "F": 802,
        "Q": 8035,
    },
    "features_total": 38,
    "selected_features": 23,
    "folds": 3,
    "patient_wise_split": True,
    "duplicate_rows": 0,
    "duplicate_record_sample_pairs": 0,
}


# ============================================================
# SELECTED FEATURES
# ============================================================

selected_features = [
    "prev_rr",
    "prev_hr",
    "spectral_entropy",
    "dominant_frequency",
    "energy_ratio",
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


feature_groups = {
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
    ],
}


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

print("\n[3] Building final model configuration...")

model_config = {
    "algorithm": "HistGradientBoostingClassifier",

    "feature_count": 23,

    "features": selected_features,

    "class_weights": {
        "N": 1.0,
        "S": 3.0,
        "V": 1.0,
        "F": 5.0,
        "Q": 1.5,
    },

    "hyperparameters": {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": 42,
    },

    "training_samples": 109395,

    "training_strategy":
        "Final full-dataset model trained after "
        "patient-wise validation/model selection",

    "model_file":
        "models/final_ecg_hgb_model.joblib",
}


# ============================================================
# FINAL VALIDATION METRICS
# ============================================================

print("\n[4] Extracting final validation metrics...")

overall = {
    "accuracy": 0.863677,
    "balanced_accuracy": 0.597188,
    "macro_precision": 0.5269,
    "macro_recall": 0.597188,
    "macro_f1": 0.547298,
    "weighted_f1": 0.876315,
}


# ============================================================
# METHODOLOGY
# ============================================================

methodology = {
    "data_source":
        "MIT-BIH based ECG beat dataset",

    "preprocessing": [
        "ECG records were processed into individual beats.",
        "Beat signals were normalized.",
        "Signal, rhythm, spectral and morphology features were extracted.",
        "Missing and invalid feature values were checked.",
    ],

    "feature_engineering": [
        "Initial feature set contained 38 ML features.",
        "Feature importance and ablation experiments were performed.",
        "A fixed 23-feature subset was selected for the final model.",
    ],

    "class_imbalance": [
        "Class imbalance was explicitly evaluated.",
        "Class-weighted HistGradientBoosting was tested.",
        "Final selected class weights were N=1, S=3, V=1, F=5, Q=1.5.",
    ],

    "validation":
        "Patient-wise 3-fold cross-validation was used "
        "to prevent beats from the same patient appearing "
        "in both training and test partitions.",

    "nested_validation":
        "Step 47B performed nested patient-wise "
        "hyperparameter validation using the frozen "
        "23-feature set.",

    "final_training":
        "After model selection, the final deployment model "
        "was trained using the complete dataset.",
}


# ============================================================
# ERROR ANALYSIS
# ============================================================

print("\n[5] Building error analysis...")

top_errors = []

if not errors.empty:

    for _, row in errors.head(10).iterrows():

        top_errors.append({
            "actual": str(row["actual"]),
            "predicted": str(row["predicted"]),
            "errors": int(row["errors"]),
            "percentage_of_errors":
                float(row["percentage_of_errors"]),
        })


# ============================================================
# LIMITATIONS
# ============================================================

limitations = [
    "The dataset is highly imbalanced across ECG classes.",
    "The F class contains substantially fewer samples than the N class.",
    "S and F classification performance remains limited.",
    "The evaluation uses three patient-wise folds rather than an independent external clinical dataset.",
    "The nested validation procedure froze the 23-feature set selected in an earlier feature-selection stage.",
    "Therefore, the nested result should be described as rigorous patient-wise hyperparameter validation with a frozen feature set, rather than a completely independent external test estimate.",
    "ROC-AUC should not be interpreted alone for minority classes because class imbalance can make ROC-AUC appear relatively strong while precision and F1 remain low.",
    "The system should not be interpreted as a clinically validated diagnostic system.",
]


# ============================================================
# FUTURE WORK
# ============================================================

future_work = [
    "Collect or evaluate the model on additional ECG datasets.",
    "Perform external patient-wise validation on an independent dataset.",
    "Increase representation of minority S and F classes.",
    "Investigate advanced imbalance-learning methods.",
    "Evaluate focal loss or other cost-sensitive approaches.",
    "Explore CNN or hybrid deep-learning architectures using raw ECG signals.",
    "Investigate patient-specific and domain-adaptation techniques.",
    "Perform calibration analysis of predicted probabilities.",
    "Evaluate clinically relevant sensitivity and specificity requirements.",
    "Develop a user-facing ECG prediction application.",
]


# ============================================================
# REPORT TEXT
# ============================================================

print("\n[6] Generating report-ready text...")


report_text = f"""
SMART ECG-ML
FINAL MACHINE LEARNING ANALYSIS

1. PROJECT OVERVIEW
-------------------

SmartECG-ML is an ECG beat classification system designed to classify
ECG beats into five classes: N, S, V, F and Q.

The machine-learning pipeline consists of ECG preprocessing,
beat extraction, feature engineering, feature selection,
class-imbalance handling, patient-wise validation and
HistGradientBoosting-based classification.


2. DATASET
----------

The processed dataset contains 109,395 ECG beat samples from
47 patients and 48 records.

The five classes are:

N = 90,545 samples
S = 2,779 samples
V = 7,234 samples
F = 802 samples
Q = 8,035 samples

The class distribution is highly imbalanced. The N class represents
approximately 82.77% of the dataset, while the F class represents
approximately 0.73%.


3. FEATURE ENGINEERING
----------------------

The initial machine-learning feature set contained 38 features.

The feature engineering pipeline included rhythm, spectral and
morphological characteristics.

After feature importance and ablation experiments, a fixed
23-feature subset was used for the final model.

The selected features were:

{", ".join(selected_features)}


4. CLASS IMBALANCE
------------------

Class imbalance was explicitly evaluated during model development.

The selected class weights were:

N = 1.0
S = 3.0
V = 1.0
F = 5.0
Q = 1.5

The purpose of class weighting was to increase the contribution of
minority classes during model training.


5. MODEL
--------

The selected classifier is:

HistGradientBoostingClassifier

Parameters:

Learning rate = 0.08
Maximum iterations = 300
Maximum leaf nodes = 31
Minimum samples per leaf = 20
L2 regularization = 0.1
Random state = 42


6. VALIDATION METHODOLOGY
-------------------------

Patient-wise three-fold cross-validation was used.

The patient-wise strategy prevents ECG beats belonging to the same
patient from being distributed between training and test partitions.

Step 47B performed nested patient-wise hyperparameter validation
using the frozen 23-feature set.

This provides a more rigorous assessment of hyperparameter selection
than using the same observations for both hyperparameter selection
and evaluation.


7. FINAL NESTED VALIDATION RESULTS
----------------------------------

Accuracy            = {overall["accuracy"] * 100:.2f}%
Balanced Accuracy   = {overall["balanced_accuracy"] * 100:.2f}%
Macro Precision     = {overall["macro_precision"] * 100:.2f}%
Macro Recall        = {overall["macro_recall"] * 100:.2f}%
Macro F1            = {overall["macro_f1"] * 100:.2f}%
Weighted F1         = {overall["weighted_f1"] * 100:.2f}%


8. PER-CLASS PERFORMANCE
------------------------

"""


for _, row in class_metrics.iterrows():

    cls = row["class"]

    report_text += (
        f"{cls}: "
        f"Precision={row['precision']*100:.2f}%, "
        f"Recall={row['recall']*100:.2f}%, "
        f"F1={row['f1']*100:.2f}%, "
        f"Support={int(row['support']):,}\n"
    )


report_text += """

9. ERROR ANALYSIS
-----------------

The most frequent classification errors were:

"""


for item in top_errors:

    report_text += (
        f"Actual {item['actual']} -> "
        f"Predicted {item['predicted']}: "
        f"{item['errors']:,} errors "
        f"({item['percentage_of_errors']:.2f}% of all errors)\n"
    )


report_text += """

The results show that the main challenge is the classification of
minority S and F classes. These classes have considerably lower recall
and F1 values than N, V and Q.


10. ROC-AUC AND PRECISION-RECALL ANALYSIS
-----------------------------------------

ROC-AUC and Average Precision were calculated using the nested
out-of-fold probability predictions.

These metrics should be interpreted together with precision, recall
and F1-score, particularly for highly imbalanced classes.

"""


for _, row in roc_pr.iterrows():

    report_text += (
        f"{row['class']}: "
        f"ROC-AUC={row['roc_auc']:.4f}, "
        f"Average Precision={row['average_precision']:.4f}\n"
    )


report_text += """

11. FOLD VARIABILITY
--------------------

The mean and standard deviation across the three outer patient-wise
folds provide an indication of variation between patient partitions.

"""


for _, row in fold_stats.iterrows():

    report_text += (
        f"{row['metric']}: "
        f"{row['mean']:.4f} +/- {row['std']:.4f}\n"
    )


report_text += """

12. LIMITATIONS
---------------

"""


for item in limitations:

    report_text += f"- {item}\n"


report_text += """

13. FUTURE WORK
---------------

"""


for item in future_work:

    report_text += f"- {item}\n"


report_text += """

14. CONCLUSION
--------------

The SmartECG-ML experiments demonstrate that patient-wise validation
combined with engineered ECG features and HistGradientBoosting can
provide useful multi-class ECG beat classification performance.

The nested patient-wise validation achieved 86.37% accuracy,
59.72% balanced accuracy and 54.73% macro F1.

Performance was strongest for the N, V and Q classes, while S and F
remain challenging because of class imbalance and classification
overlap.

Further improvement should focus on minority-class representation,
external validation and methods specifically designed for highly
imbalanced ECG classification.

The reported machine-learning results should be regarded as research
validation results rather than clinical diagnostic validation.
"""


# ============================================================
# SAVE REPORT TEXT
# ============================================================

report_file = OUTPUT / "SMART_ECG_ML_FINAL_REPORT_TEXT.txt"

with open(
    report_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(report_text)


# ============================================================
# SAVE TABLES
# ============================================================

print("\n[7] Saving report-ready tables...")

tables = {
    "01_dataset_summary.csv": pd.DataFrame([
        {
            "samples": 109395,
            "patients": 47,
            "records": 48,
            "features_initial": 38,
            "features_final": 23,
            "folds": 3,
        }
    ]),

    "02_class_distribution.csv": imbalance,

    "03_per_class_metrics.csv":
        class_metrics,

    "04_confusion_matrix.csv":
        confusion,

    "05_normalized_confusion_matrix.csv":
        normalized_confusion,

    "06_error_pairs.csv":
        errors,

    "07_class_error_analysis.csv":
        class_errors,

    "08_fold_mean_sd.csv":
        fold_stats,

    "09_roc_pr_metrics.csv":
        roc_pr,

    "10_model_comparison.csv":
        model_comparison,
}


for filename, dataframe in tables.items():

    if not dataframe.empty:

        dataframe.to_csv(
            OUTPUT / filename,
            index=False,
        )


# ============================================================
# FEATURE LIST
# ============================================================

pd.DataFrame({
    "feature": selected_features
}).to_csv(
    OUTPUT / "11_final_23_features.csv",
    index=False,
)


# ============================================================
# JSON MASTER PACKAGE
# ============================================================

print("\n[8] Creating master JSON...")

master = {
    "project": "SmartECG-ML",

    "dataset": dataset_summary,

    "features": {
        "initial_count": 38,
        "final_count": 23,
        "selected_features": selected_features,
        "groups": feature_groups,
    },

    "model": model_config,

    "validation": {
        "method":
            "Nested patient-wise hyperparameter validation "
            "with frozen 23-feature set",

        "overall_metrics": overall,

        "class_metrics":
            class_metrics.to_dict(
                orient="records"
            ),

        "fold_statistics":
            fold_stats.to_dict(
                orient="records"
            ),
    },

    "error_analysis": {
        "top_errors": top_errors,
    },

    "class_imbalance":
        imbalance.to_dict(
            orient="records"
        ),

    "roc_pr":
        roc_pr.to_dict(
            orient="records"
        ),

    "limitations":
        limitations,

    "future_work":
        future_work,

    "final_model_file":
        "models/final_ecg_hgb_model.joblib",
}


master_file = OUTPUT / "12_master_final_report_package.json"

with open(
    master_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        master,
        f,
        indent=2,
        default=float,
    )


# ============================================================
# METHODOLOGY SUMMARY
# ============================================================

methodology_text = """
SMART ECG-ML - METHODOLOGY SUMMARY

1. ECG dataset preparation
        |
        v
2. ECG beat extraction
        |
        v
3. Signal preprocessing / normalization
        |
        v
4. Feature extraction
        |
        v
5. 38 initial ML features
        |
        v
6. Feature importance and ablation
        |
        v
7. 23 selected features
        |
        v
8. Class imbalance analysis
        |
        v
9. HistGradientBoosting
        |
        v
10. Patient-wise 3-fold validation
        |
        v
11. Nested patient-wise hyperparameter validation
        |
        v
12. Statistical and error analysis
        |
        v
13. Final full-dataset model
        |
        v
14. ECG prediction application
"""


with open(
    OUTPUT / "13_methodology_flow.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(methodology_text)


# ============================================================
# FINAL CONSOLE
# ============================================================

print("\n" + "=" * 80)
print("STEP 49 COMPLETE")
print("=" * 80)

print("\nFinal validation:")
print(
    f"Accuracy          : {overall['accuracy']*100:.2f}%"
)
print(
    f"Balanced Accuracy : {overall['balanced_accuracy']*100:.2f}%"
)
print(
    f"Macro F1          : {overall['macro_f1']*100:.2f}%"
)
print(
    f"Weighted F1       : {overall['weighted_f1']*100:.2f}%"
)

print("\nFinal model:")
print("HistGradientBoostingClassifier")
print("23 features")
print("Patient-wise validation")
print("Class weighted")

print("\nGenerated package:")
print(OUTPUT)

print("\nMain files:")

for file in sorted(OUTPUT.iterdir()):

    print(f"  - {file.name}")

print("\n" + "=" * 80)
print("REPORT PACKAGE READY")
print("=" * 80)