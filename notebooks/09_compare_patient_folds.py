import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------
# PATH
# --------------------------------------------------

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_dataset_patientwise.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

X = df.drop(columns=["class"])
y = df["class"]
groups = df["patient_id"]


# --------------------------------------------------
# TEST 3-FOLD
# --------------------------------------------------

print("=" * 70)
print("3-FOLD PATIENT-WISE STRATIFIED CROSS-VALIDATION")
print("=" * 70)

sgkf = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

df["fold"] = -1

for fold, (train_idx, test_idx) in enumerate(
    sgkf.split(X, y, groups)
):

    df.loc[test_idx, "fold"] = fold


# --------------------------------------------------
# ANALYZE FOLDS
# --------------------------------------------------

for fold in range(3):

    fold_data = df[df["fold"] == fold]

    print("\n" + "=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    print(
        "Patients:",
        fold_data["patient_id"].nunique()
    )

    print(
        "Beats:",
        len(fold_data)
    )

    print("\nClass counts:")

    counts = (
        fold_data["class"]
        .value_counts()
        .reindex(
            ["N", "S", "V", "F", "Q"],
            fill_value=0
        )
    )

    print(counts)

    print("\nClass percentages:")

    percentages = (
        counts / len(fold_data) * 100
    )

    print(percentages.round(2))


# --------------------------------------------------
# LEAKAGE CHECK
# --------------------------------------------------

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

leakage = False

for fold_a in range(3):

    patients_a = set(
        df[df["fold"] == fold_a]["patient_id"]
    )

    for fold_b in range(
        fold_a + 1,
        3
    ):

        patients_b = set(
            df[df["fold"] == fold_b]["patient_id"]
        )

        overlap = (
            patients_a.intersection(patients_b)
        )

        if overlap:

            leakage = True

            print(
                f"Leakage between fold "
                f"{fold_a} and {fold_b}:"
            )

            print(overlap)


if not leakage:

    print(
        "SUCCESS: No patient appears "
        "in multiple folds."
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\patient_3folds.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved to:")
print(OUTPUT_PATH)