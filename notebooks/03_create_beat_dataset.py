import os
import wfdb
import pandas as pd

# --------------------------------------------------
# PATH
# --------------------------------------------------

DATA_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\raw"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_annotations.csv"


# --------------------------------------------------
# BEAT ANNOTATIONS
# --------------------------------------------------

BEAT_SYMBOLS = {
    "N", "L", "R", "B",
    "A", "a", "J", "S",
    "V", "r",
    "F",
    "e", "j", "n", "E",
    "/",
    "f",
    "Q", "?"
}


# --------------------------------------------------
# FIND RECORDS
# --------------------------------------------------

record_names = sorted(
    file.replace(".hea", "")
    for file in os.listdir(DATA_PATH)
    if file.endswith(".hea")
)

print("Total records:", len(record_names))


# --------------------------------------------------
# CREATE DATA
# --------------------------------------------------

dataset = []

for record_name in record_names:

    record_path = os.path.join(DATA_PATH, record_name)

    print("Processing record:", record_name)

    try:

        # Read ECG
        record = wfdb.rdrecord(record_path)

        # Read reference annotations
        annotation = wfdb.rdann(
            record_path,
            "atr"
        )

        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol
        ):

            # Keep only heartbeat annotations
            if symbol in BEAT_SYMBOLS:

                dataset.append({

                    "record": record_name,

                    "sample": int(sample),

                    "time_seconds": sample / record.fs,

                    "annotation": symbol

                })

    except Exception as e:

        print(
            "Error in record",
            record_name,
            ":",
            e
        )


# --------------------------------------------------
# DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(dataset)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)

print("BEAT DATASET CREATED")

print("=" * 60)

print("\nTotal beat annotations:", len(df))

print("\nAnnotation distribution:")

print(
    df["annotation"]
    .value_counts()
)

print("\nFirst 20 rows:")

print(
    df.head(20)
)

print(
    "\nSaved to:",
    OUTPUT_PATH
)