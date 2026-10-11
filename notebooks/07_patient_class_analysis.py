import pandas as pd

# --------------------------------------------------
# PATH
# --------------------------------------------------

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_dataset_patientwise.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("PATIENT × ECG CLASS ANALYSIS")
print("=" * 70)

print("\nTotal patients:", df["patient_id"].nunique())
print("Total beats:", len(df))


# --------------------------------------------------
# PATIENT × CLASS COUNT
# --------------------------------------------------

patient_class = pd.crosstab(
    df["patient_id"],
    df["class"]
)

# Make sure all five classes exist as columns
for class_name in ["N", "S", "V", "F", "Q"]:

    if class_name not in patient_class.columns:
        patient_class[class_name] = 0


# Arrange columns
patient_class = patient_class[
    ["N", "S", "V", "F", "Q"]
]


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\n" + "=" * 70)
print("BEATS PER PATIENT AND CLASS")
print("=" * 70)

print(patient_class.to_string())


# --------------------------------------------------
# NUMBER OF PATIENTS CONTAINING EACH CLASS
# --------------------------------------------------

patients_per_class = (
    patient_class
    .gt(0)
    .sum()
)

print("\n" + "=" * 70)
print("NUMBER OF PATIENTS CONTAINING EACH CLASS")
print("=" * 70)

print(patients_per_class)


# --------------------------------------------------
# PERCENTAGE OF PATIENTS
# --------------------------------------------------

patient_percentage = (
    patients_per_class
    / df["patient_id"].nunique()
    * 100
)

print("\n" + "=" * 70)
print("PERCENTAGE OF PATIENTS CONTAINING EACH CLASS")
print("=" * 70)

print(patient_percentage.round(2))


# --------------------------------------------------
# RARE CLASS PATIENTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("PATIENTS WITH F CLASS")
print("=" * 70)

print(
    patient_class[
        patient_class["F"] > 0
    ][["F"]]
)


print("\n" + "=" * 70)
print("PATIENTS WITH S CLASS")
print("=" * 70)

print(
    patient_class[
        patient_class["S"] > 0
    ][["S"]]
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\patient_class_distribution.csv"

patient_class.to_csv(
    OUTPUT_PATH
)

print("\nSaved patient-class table to:")
print(OUTPUT_PATH)