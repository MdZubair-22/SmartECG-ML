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

print("=" * 70)
print("PATIENT-WISE STRATIFIED CROSS-VALIDATION")
print("=" * 70)

print("\nTotal beats:", len(df))
print("Total patients:", df["patient_id"].nunique())


# --------------------------------------------------
# CREATE STRATIFIED GROUP K-FOLD
# --------------------------------------------------

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# CREATE FOLDS
# --------------------------------------------------

df["fold"] = -1

X = df.drop(columns=["class"])
y = df["class"]
groups = df["patient_id"]


for fold, (train_idx, test_idx) in enumerate(
    sgkf.split(X, y, groups)
):

    df.loc[test_idx, "fold"] = fold


# --------------------------------------------------
# CHECK FOLDS
# --------------------------------------------------

print("\n" + "=" * 70)
print("FOLD DISTRIBUTION")
print("=" * 70)

for fold in range(5):

    fold_data = df[df["fold"] == fold]

    print("\nFold", fold)

    print(
        "Patients:",
        fold_data["patient_id"].nunique()
    )

    print(
        "Beats:",
        len(fold_data)
    )

    print("\nClass distribution:")

    print(
        fold_data["class"]
        .value_counts()
        .sort_index()
    )


# --------------------------------------------------
# CHECK PATIENT LEAKAGE
# --------------------------------------------------

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

leakage_found = False

for fold_a in range(5):

    patients_a = set(
        df[df["fold"] == fold_a]["patient_id"]
    )

    for fold_b in range(fold_a + 1, 5):

        patients_b = set(
            df[df["fold"] == fold_b]["patient_id"]
        )

        overlap = patients_a.intersection(
            patients_b
        )

        if overlap:

            leakage_found = True

            print(
                f"Leakage between fold {fold_a} "
                f"and fold {fold_b}:",
                overlap
            )


if not leakage_found:

    print(
        "SUCCESS: No patient appears in multiple folds."
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\patient_stratified_folds.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved to:")
print(OUTPUT_PATH)