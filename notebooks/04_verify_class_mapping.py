import pandas as pd

# --------------------------------------------------
# PATH
# --------------------------------------------------

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_annotations.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("ORIGINAL ANNOTATION COUNTS")
print("=" * 60)

print(df["annotation"].value_counts())


# --------------------------------------------------
# AAMI-STYLE MAPPING
# --------------------------------------------------

CLASS_MAPPING = {

    # Normal
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # Supraventricular
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # Ventricular
    "V": "V",
    "E": "V",

    # Fusion
    "F": "F",

    # Unknown / other
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "?": "Q"
}


# --------------------------------------------------
# CHECK FOR UNMAPPED SYMBOLS
# --------------------------------------------------

unique_symbols = set(df["annotation"].unique())

unmapped = unique_symbols - set(CLASS_MAPPING.keys())

print("\n" + "=" * 60)
print("UNMAPPED SYMBOLS")
print("=" * 60)

print(unmapped)


# --------------------------------------------------
# MAP CLASSES
# --------------------------------------------------

df["class"] = df["annotation"].map(CLASS_MAPPING)


# --------------------------------------------------
# CHECK MISSING MAPPINGS
# --------------------------------------------------

missing = df["class"].isna().sum()

print("\nRows without a class:", missing)


# --------------------------------------------------
# CLASS DISTRIBUTION
# --------------------------------------------------

print("\n" + "=" * 60)
print("ML CLASS DISTRIBUTION")
print("=" * 60)

print(
    df["class"]
    .value_counts()
)


# --------------------------------------------------
# PERCENTAGE
# --------------------------------------------------

print("\n" + "=" * 60)
print("ML CLASS PERCENTAGE")
print("=" * 60)

percentage = (
    df["class"]
    .value_counts(normalize=True)
    * 100
)

print(
    percentage.round(2)
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_annotations_mapped.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved mapped dataset to:")
print(OUTPUT_PATH)