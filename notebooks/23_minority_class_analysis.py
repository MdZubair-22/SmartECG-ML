import pandas as pd
import numpy as np


INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\causal_ecg_features.csv"


df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "record": str,
        "patient_id": str
    }
)


print("=" * 70)
print("MINORITY CLASS SEPARABILITY ANALYSIS")
print("=" * 70)


# ============================================================
# FEATURES
# ============================================================

features = [

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


# ============================================================
# CLASS STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE MEDIANS")
print("=" * 70)


medians = (
    df
    .groupby("class")[features]
    .median()
)


print(
    medians.to_string()
)


# ============================================================
# F CLASS STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("F CLASS STATISTICS")
print("=" * 70)


f_stats = (
    df[
        df["class"] == "F"
    ][features]
    .describe()
    .T
)


print(
    f_stats.to_string()
)


# ============================================================
# COMPARE F AGAINST OTHER CLASSES
# ============================================================

print("\n" + "=" * 70)
print("F CLASS vs OTHER CLASSES")
print("=" * 70)


classes = [
    "N",
    "S",
    "V",
    "Q"
]


for other_class in classes:

    print("\n" + "-" * 70)

    print(
        f"F vs {other_class}"
    )

    print("-" * 70)


    f_median = (
        df[
            df["class"] == "F"
        ][features]
        .median()
    )


    other_median = (
        df[
            df["class"] == other_class
        ][features]
        .median()
    )


    difference = (
        f_median
        -
        other_median
    )


    comparison = pd.DataFrame({

        "F_median":
            f_median,

        f"{other_class}_median":
            other_median,

        "difference":
            difference,

        "absolute_difference":
            difference.abs()

    })


    print(
        comparison
        .sort_values(
            "absolute_difference",
            ascending=False
        )
        .to_string()
    )


# ============================================================
# PATIENT DISTRIBUTION OF F
# ============================================================

print("\n" + "=" * 70)
print("F CLASS BY PATIENT")
print("=" * 70)


f_patient = (
    df[
        df["class"] == "F"
    ]
    .groupby("patient_id")
    .size()
    .sort_values(
        ascending=False
    )
)


print(
    f_patient.to_string()
)


# ============================================================
# PATIENTS WITH F
# ============================================================

print("\nPatients containing F:")

print(
    f_patient.size
)


print(
    "Total F samples:",
    f_patient.sum()
)


print(
    "Median F samples per F-containing patient:",
    f_patient.median()
)


# ============================================================
# FOLD DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("F CLASS BY FOLD")
print("=" * 70)


print(
    pd.crosstab(
        df["fold"],
        df["class"]
    )["F"]
)


# ============================================================
# FEATURE VARIABILITY
# ============================================================

print("\n" + "=" * 70)
print("F CLASS FEATURE VARIABILITY")
print("=" * 70)


f_std = (
    df[
        df["class"] == "F"
    ][features]
    .std()
    .sort_values(
        ascending=False
    )
)


print(
    f_std
)