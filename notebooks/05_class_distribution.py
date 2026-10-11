import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# LOAD MAPPED DATASET
# --------------------------------------------------

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\beat_annotations_mapped.csv"

df = pd.read_csv(INPUT_PATH)

# --------------------------------------------------
# CLASS COUNTS
# --------------------------------------------------

class_counts = df["class"].value_counts()

print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(class_counts)

# --------------------------------------------------
# PERCENTAGE
# --------------------------------------------------

class_percentage = (
    df["class"]
    .value_counts(normalize=True)
    * 100
)

print("\nCLASS PERCENTAGE")
print("=" * 60)

print(class_percentage.round(2))

# --------------------------------------------------
# BAR CHART
# --------------------------------------------------

plt.figure(figsize=(9, 6))

bars = plt.bar(
    class_counts.index,
    class_counts.values
)

plt.title(
    "MIT-BIH ECG Beat Class Distribution"
)

plt.xlabel(
    "ECG Class"
)

plt.ylabel(
    "Number of Beats"
)

plt.grid(
    axis="y",
    alpha=0.3
)

# Add numbers above bars
for bar, count in zip(bars, class_counts.values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Save figure
OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\results\class_distribution.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

plt.show()

print("\nFigure saved to:")
print(OUTPUT_PATH)