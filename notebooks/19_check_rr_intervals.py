import pandas as pd
import numpy as np


INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\ml_dataset.csv"


df = pd.read_csv(
    INPUT_PATH,
    dtype={"patient_id": str}
)


print("=" * 70)
print("RR INTERVAL QUALITY CHECK")
print("=" * 70)


# ------------------------------------------------------------
# BASIC STATISTICS
# ------------------------------------------------------------

print("\nRR statistics:")

print(
    df[
        [
            "prev_rr",
            "next_rr",
            "mean_rr"
        ]
    ].describe()
)


# ------------------------------------------------------------
# VERY LARGE RR
# ------------------------------------------------------------

threshold = 3.0

large_rr = df[
    (df["prev_rr"] > threshold)
    |
    (df["next_rr"] > threshold)
]


print("\n" + "=" * 70)

print(
    f"RR intervals > {threshold} seconds"
)

print("=" * 70)

print(
    "Rows:",
    len(large_rr)
)


if len(large_rr) > 0:

    print(
        large_rr[
            [
                "record",
                "patient_id",
                "sample",
                "class",
                "prev_rr",
                "next_rr"
            ]
        ]
        .sort_values(
            "prev_rr",
            ascending=False
        )
        .head(30)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# EXTREME RR
# ------------------------------------------------------------

print("\n" + "=" * 70)

print("EXTREME RR INTERVALS")

print("=" * 70)


extreme = df[
    (df["prev_rr"] > 10)
    |
    (df["next_rr"] > 10)
]


print(
    "Rows with RR > 10 sec:",
    len(extreme)
)


if len(extreme) > 0:

    print(
        extreme[
            [
                "record",
                "patient_id",
                "sample",
                "class",
                "prev_rr",
                "next_rr"
            ]
        ]
        .to_string(index=False)
    )


# ------------------------------------------------------------
# CLASS-WISE RR
# ------------------------------------------------------------

print("\n" + "=" * 70)

print("CLASS-WISE RR STATISTICS")

print("=" * 70)


class_rr = (
    df
    .groupby("class")[
        [
            "prev_rr",
            "next_rr",
            "mean_rr"
        ]
    ]
    .median()
)


print(
    class_rr
)


# ------------------------------------------------------------
# RECORD-WISE EXTREMES
# ------------------------------------------------------------

print("\n" + "=" * 70)

print("RECORDS CONTAINING RR > 10 SEC")

print("=" * 70)


if len(extreme) > 0:

    print(
        extreme["record"]
        .value_counts()
    )

else:

    print(
        "None"
    )