import os
import wfdb
from collections import Counter

# Folder containing MIT-BIH records
DATA_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\raw"

# Find all .hea files
record_files = [
    file for file in os.listdir(DATA_PATH)
    if file.endswith(".hea")
]

# Get record names
record_names = sorted(
    file.replace(".hea", "")
    for file in record_files
)

print("=" * 60)
print("MIT-BIH DATASET INSPECTION")
print("=" * 60)

print("\nTotal records found:", len(record_names))

print("\nRecord names:")
print(record_names)

# Store total annotation counts
total_annotations = Counter()

# Store annotation counts for each record
record_annotation_counts = {}

# Inspect every record
for record_name in record_names:

    record_path = os.path.join(DATA_PATH, record_name)

    try:
        # Read annotations
        annotation = wfdb.rdann(record_path, "atr")

        counts = Counter(annotation.symbol)

        record_annotation_counts[record_name] = counts

        # Add to total
        total_annotations.update(counts)

        print(
            f"\nRecord {record_name}: "
            f"{len(annotation.symbol)} annotations"
        )

    except Exception as e:
        print(
            f"\nCould not read record {record_name}: {e}"
        )


# -------------------------------------------------
# TOTAL ANNOTATION DISTRIBUTION
# -------------------------------------------------

print("\n" + "=" * 60)
print("TOTAL ANNOTATION DISTRIBUTION")
print("=" * 60)

for symbol, count in total_annotations.most_common():

    print(f"{symbol:>5} : {count}")


# -------------------------------------------------
# SUMMARY
# -------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("Total records:", len(record_names))

print(
    "Total annotations:",
    sum(total_annotations.values())
)

print(
    "Different annotation symbols:",
    len(total_annotations)
)