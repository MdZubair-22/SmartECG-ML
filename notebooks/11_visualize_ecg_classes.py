import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATH
# --------------------------------------------------

INPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\ecg_beats.npz"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = np.load(INPUT_PATH)

signals = data["signals"]
labels = data["labels"]


print("=" * 70)
print("ECG BEAT VISUALIZATION")
print("=" * 70)

print("Signal shape:", signals.shape)
print("Labels shape:", labels.shape)


# --------------------------------------------------
# CLASSES
# --------------------------------------------------

classes = ["N", "S", "V", "F", "Q"]


# --------------------------------------------------
# PLOT ONE EXAMPLE FROM EACH CLASS
# --------------------------------------------------

plt.figure(figsize=(12, 10))

plot_number = 1

for class_name in classes:

    indices = np.where(labels == class_name)[0]

    if len(indices) == 0:
        print("No samples found for:", class_name)
        continue

    index = indices[0]

    signal = signals[index]

    plt.subplot(5, 1, plot_number)

    plt.plot(signal)

    plt.title(
        f"Class {class_name} - Example ECG Beat"
    )

    plt.xlabel("Sample")

    plt.ylabel("Amplitude")

    plt.grid(alpha=0.3)

    plot_number += 1


plt.tight_layout()

plt.show()