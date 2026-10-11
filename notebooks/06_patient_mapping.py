import pandas as pd

# --------------------------------------------------
# PATH
# --------------------------------------------------

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_annotations_mapped.csv"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_dataset_patientwise.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("PATIENT-WISE DATASET PREPARATION")
print("=" * 60)

print("\nOriginal records:")
print(df["record"].nunique())


# --------------------------------------------------
# PATIENT MAPPING
# --------------------------------------------------

# MIT-BIH contains 48 records from 47 subjects.
# Records 201 and 202 belong to the same subject.

df["patient_id"] = df["record"].astype(str)

df.loc[
    df["record"].isin([201, 202]),
    "patient_id"
] = "P201_202"


# --------------------------------------------------
# COUNT PATIENTS
# --------------------------------------------------

print("\nUnique patients:")
print(df["patient_id"].nunique())


# --------------------------------------------------
# RECORD → PATIENT TABLE
# --------------------------------------------------

record_patient = (
    df[["record", "patient_id"]]
    .drop_duplicates()
    .sort_values("record")
)

print("\nRecord → Patient mapping:")
print(record_patient.to_string(index=False))


# --------------------------------------------------
# PATIENT SAMPLE COUNTS
# --------------------------------------------------

patient_counts = (
    df.groupby("patient_id")
    .size()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("BEATS PER PATIENT")
print("=" * 60)

print(patient_counts)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved patient-wise dataset to:")
print(OUTPUT_PATH)