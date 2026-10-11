import os
import numpy as np
import wfdb

from scipy.signal import butter, sosfiltfilt


# ============================================================
# PATHS
# ============================================================

DATA_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\raw"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\filtered_records"


# ============================================================
# FILTER SETTINGS
# ============================================================

LOWCUT = 0.5
HIGHCUT = 40.0

FILTER_ORDER = 4

LEAD_INDEX = 0


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


# ============================================================
# BANDPASS FILTER FUNCTION
# ============================================================

def bandpass_filter(signal, fs):

    nyquist = fs / 2

    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist

    sos = butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        signal
    )

    return filtered


# ============================================================
# FIND RECORDS
# ============================================================

record_names = sorted(
    file.replace(".hea", "")
    for file in os.listdir(DATA_PATH)
    if file.endswith(".hea")
)


print("=" * 70)
print("ECG PREPROCESSING")
print("=" * 70)

print("Total records:", len(record_names))

print("Filter:")
print(f"Bandpass {LOWCUT}–{HIGHCUT} Hz")

print("Lead:")
print("MLII")


# ============================================================
# PROCESS EACH RECORD
# ============================================================

for record_name in record_names:

    print(
        f"\nProcessing record {record_name}..."
    )

    record_path = os.path.join(
        DATA_PATH,
        record_name
    )

    # Read ECG
    record = wfdb.rdrecord(
        record_path
    )

    fs = record.fs

    # Select MLII
    signal = record.p_signal[
        :, LEAD_INDEX
    ]

    # Filter
    filtered_signal = bandpass_filter(
        signal,
        fs
    )

    # Save
    output_file = os.path.join(
        OUTPUT_PATH,
        f"{record_name}.npy"
    )

    np.save(
        output_file,
        filtered_signal.astype(
            np.float32
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("PREPROCESSING COMPLETE")

print("=" * 70)

print(
    "Filtered records saved to:"
)

print(OUTPUT_PATH)