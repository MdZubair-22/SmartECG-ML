import pandas as pd
import numpy as np


# ============================================================
# PATH
# ============================================================

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\ml_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("FEATURE QUALITY ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# FEATURE COLUMNS
# ============================================================

metadata = [
    "record",
    "patient_id",
    "sample",
    "class",
    "fold"
]

features = [
    column
    for column in df.columns
    if column not in metadata
]


print("\nNumber of features:")
print(len(features))


print("\nFeature names:")

for feature in features:
    print("-", feature)


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(
    df[features].dtypes
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = (
    df[features]
    .isna()
    .sum()
)

print(missing)

print(
    "\nTotal missing values:",
    missing.sum()
)


# ============================================================
# INFINITE VALUES
# ============================================================

numeric_features = (
    df[features]
    .select_dtypes(
        include=np.number
    )
)

infinite = np.isinf(
    numeric_features
).sum()

print("\n" + "=" * 70)
print("INFINITE VALUES")
print("=" * 70)

print(infinite)

print(
    "\nTotal infinite values:",
    infinite.sum()
)


# ============================================================
# CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

constant_features = []

for feature in features:

    if df[feature].nunique() <= 1:

        constant_features.append(
            feature
        )


if constant_features:

    print(
        constant_features
    )

else:

    print(
        "No constant features found."
    )


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

stats = (
    df[features]
    .describe()
    .T
)

print(stats)


# ============================================================
# HIGH CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("HIGHLY CORRELATED FEATURES")
print("=" * 70)

correlation = (
    df[features]
    .corr()
    .abs()
)

high_corr = []

for i in range(
    len(correlation.columns)
):

    for j in range(
        i + 1,
        len(correlation.columns)
    ):

        value = correlation.iloc[
            i,
            j
        ]

        if value >= 0.90:

            high_corr.append({

                "feature_1":
                    correlation.columns[i],

                "feature_2":
                    correlation.columns[j],

                "correlation":
                    value
            })


if high_corr:

    high_corr_df = pd.DataFrame(
        high_corr
    )

    print(
        high_corr_df
        .sort_values(
            "correlation",
            ascending=False
        )
        .to_string(index=False)
    )

else:

    print(
        "No feature pairs with correlation >= 0.90."
    )


# ============================================================
# CLASS-WISE MEANS
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE FEATURE MEANS")
print("=" * 70)

class_means = (
    df
    .groupby("class")[features]
    .mean()
)

print(
    class_means
)


# ============================================================
# SAVE STATISTICS
# ============================================================

stats_path = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\results\feature_statistics.csv"

stats.to_csv(
    stats_path
)


print("\nFeature statistics saved to:")

print(stats_path)