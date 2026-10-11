import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BEAT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\filtered_ecg_beats.npz"

CAUSAL_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\causal_ecg_features.csv"

OUTPUT_PATH = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\processed\causal_morphology_features.csv"


# ============================================================
# SETTINGS
# ============================================================

FS = 360

# Annotation position:
# 0.25 sec * 360 Hz = 90 samples
ANNOTATION_INDEX = 90


# ============================================================
# LOAD ECG BEATS
# ============================================================

print("=" * 70)
print("CAUSAL ECG MORPHOLOGY FEATURE EXTRACTION")
print("=" * 70)


data = np.load(
    BEAT_PATH,
    allow_pickle=True
)


signals = data["signals"]

records = data["records"]

patients = data["patients"]

samples = data["samples"]


print(
    "Signal shape:",
    signals.shape
)

print(
    "Annotation index:",
    ANNOTATION_INDEX
)


# ============================================================
# IMPORTANT
# ============================================================

print("\nCausal rule:")
print(
    "Only samples 0 to 90 are used."
)

print(
    "Samples after annotation are NOT used."
)


# ============================================================
# FEATURE FUNCTION
# ============================================================

def extract_features(signal):

    # --------------------------------------------------------
    # CAUSAL PART ONLY
    # --------------------------------------------------------

    x = signal[
        :ANNOTATION_INDEX + 1
    ]


    # --------------------------------------------------------
    # BASIC MORPHOLOGY
    # --------------------------------------------------------

    minimum = np.min(x)

    maximum = np.max(x)

    amplitude_range = (
        maximum - minimum
    )

    mean_value = np.mean(x)

    std_value = np.std(x)

    rms = np.sqrt(
        np.mean(
            x ** 2
        )
    )


    # --------------------------------------------------------
    # ANNOTATION-POINT AMPLITUDE
    # --------------------------------------------------------

    annotation_amplitude = (
        signal[ANNOTATION_INDEX]
    )


    # --------------------------------------------------------
    # PRE-ANNOTATION AMPLITUDE
    # --------------------------------------------------------

    pre_10 = signal[
        max(0, ANNOTATION_INDEX - 10)
        :
        ANNOTATION_INDEX
    ]

    pre_20 = signal[
        max(0, ANNOTATION_INDEX - 20)
        :
        ANNOTATION_INDEX
    ]

    pre_30 = signal[
        max(0, ANNOTATION_INDEX - 30)
        :
        ANNOTATION_INDEX
    ]


    pre10_mean = np.mean(
        pre_10
    )

    pre20_mean = np.mean(
        pre_20
    )

    pre30_mean = np.mean(
        pre_30
    )


    # --------------------------------------------------------
    # LOCAL AMPLITUDE DIFFERENCES
    # --------------------------------------------------------

    annotation_vs_pre10 = (
        annotation_amplitude
        -
        pre10_mean
    )

    annotation_vs_pre20 = (
        annotation_amplitude
        -
        pre20_mean
    )

    annotation_vs_pre30 = (
        annotation_amplitude
        -
        pre30_mean
    )


    # --------------------------------------------------------
    # SLOPE FEATURES
    # --------------------------------------------------------

    if len(pre_10) >= 2:

        slope_10 = (
            annotation_amplitude
            -
            pre_10[0]
        ) / len(pre_10)

    else:

        slope_10 = 0.0


    if len(pre_20) >= 2:

        slope_20 = (
            annotation_amplitude
            -
            pre_20[0]
        ) / len(pre_20)

    else:

        slope_20 = 0.0


    if len(pre_30) >= 2:

        slope_30 = (
            annotation_amplitude
            -
            pre_30[0]
        ) / len(pre_30)

    else:

        slope_30 = 0.0


    # --------------------------------------------------------
    # MAXIMUM SLOPE
    # --------------------------------------------------------

    derivative = np.diff(
        x
    )


    if len(derivative) > 0:

        max_positive_slope = np.max(
            derivative
        )

        max_negative_slope = np.min(
            derivative
        )

        mean_absolute_slope = np.mean(
            np.abs(derivative)
        )

    else:

        max_positive_slope = 0.0

        max_negative_slope = 0.0

        mean_absolute_slope = 0.0


    # --------------------------------------------------------
    # ENERGY
    # --------------------------------------------------------

    total_energy = np.sum(
        x ** 2
    )


    # --------------------------------------------------------
    # PRE-ANNOTATION ENERGY
    # --------------------------------------------------------

    pre10_energy = np.sum(
        pre_10 ** 2
    )

    pre20_energy = np.sum(
        pre_20 ** 2
    )

    pre30_energy = np.sum(
        pre_30 ** 2
    )


    # --------------------------------------------------------
    # ENERGY RATIOS
    # --------------------------------------------------------

    if total_energy > 0:

        pre10_energy_ratio = (
            pre10_energy
            /
            total_energy
        )

        pre20_energy_ratio = (
            pre20_energy
            /
            total_energy
        )

        pre30_energy_ratio = (
            pre30_energy
            /
            total_energy
        )

    else:

        pre10_energy_ratio = 0.0

        pre20_energy_ratio = 0.0

        pre30_energy_ratio = 0.0


    # --------------------------------------------------------
    # ZERO CROSSINGS
    # --------------------------------------------------------

    zero_crossings = np.sum(
        np.diff(
            np.signbit(x)
        )
    )


    # --------------------------------------------------------
    # LOCAL PEAK INFORMATION
    # --------------------------------------------------------

    peak_index = np.argmax(
        np.abs(x)
    )

    distance_to_peak = (
        ANNOTATION_INDEX
        -
        peak_index
    )


    # --------------------------------------------------------
    # WAVEFORM ASYMMETRY
    # --------------------------------------------------------

    midpoint = len(x) // 2

    first_half = x[
        :midpoint
    ]

    second_half = x[
        midpoint:
    ]


    first_energy = np.sum(
        first_half ** 2
    )

    second_energy = np.sum(
        second_half ** 2
    )


    if (
        first_energy
        +
        second_energy
    ) > 0:

        energy_asymmetry = (
            first_energy
            -
            second_energy
        ) / (
            first_energy
            +
            second_energy
        )

    else:

        energy_asymmetry = 0.0


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "morph_annotation_amplitude":
            annotation_amplitude,

        "morph_min":
            minimum,

        "morph_max":
            maximum,

        "morph_range":
            amplitude_range,

        "morph_mean":
            mean_value,

        "morph_std":
            std_value,

        "morph_rms":
            rms,

        "morph_pre10_mean":
            pre10_mean,

        "morph_pre20_mean":
            pre20_mean,

        "morph_pre30_mean":
            pre30_mean,

        "morph_annotation_pre10_diff":
            annotation_vs_pre10,

        "morph_annotation_pre20_diff":
            annotation_vs_pre20,

        "morph_annotation_pre30_diff":
            annotation_vs_pre30,

        "morph_slope10":
            slope_10,

        "morph_slope20":
            slope_20,

        "morph_slope30":
            slope_30,

        "morph_max_positive_slope":
            max_positive_slope,

        "morph_max_negative_slope":
            max_negative_slope,

        "morph_mean_abs_slope":
            mean_absolute_slope,

        "morph_pre10_energy_ratio":
            pre10_energy_ratio,

        "morph_pre20_energy_ratio":
            pre20_energy_ratio,

        "morph_pre30_energy_ratio":
            pre30_energy_ratio,

        "morph_zero_crossings":
            zero_crossings,

        "morph_distance_to_peak":
            distance_to_peak,

        "morph_energy_asymmetry":
            energy_asymmetry
    }


# ============================================================
# EXTRACT
# ============================================================

feature_rows = []


print(
    "\nExtracting morphology features..."
)


for i in range(
    len(signals)
):

    features = extract_features(
        signals[i]
    )

    features["record"] = str(
        records[i]
    )

    features["patient_id"] = str(
        patients[i]
    )

    features["sample"] = int(
        samples[i]
    )

    feature_rows.append(
        features
    )


morphology_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# REORDER
# ============================================================

metadata_columns = [
    "record",
    "patient_id",
    "sample"
]


feature_columns = [
    col
    for col in morphology_df.columns
    if col not in metadata_columns
]


morphology_df = morphology_df[
    metadata_columns
    +
    feature_columns
]


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

numeric_columns = [
    col
    for col in morphology_df.columns
    if col not in metadata_columns
]


morphology_df[
    numeric_columns
] = morphology_df[
    numeric_columns
].replace(
    [np.inf, -np.inf],
    np.nan
)


print(
    "\nMissing values:"
)

print(
    morphology_df[
        numeric_columns
    ].isna().sum()
    .sum()
)


# ============================================================
# DROP INVALID
# ============================================================

before = len(
    morphology_df
)


morphology_df = morphology_df.dropna()


after = len(
    morphology_df
)


print(
    "\nRows before cleaning:",
    before
)

print(
    "Rows after cleaning:",
    after
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_PATH
    ),
    exist_ok=True
)


morphology_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "MORPHOLOGY EXTRACTION COMPLETE"
)

print("=" * 70)


print(
    "Rows:",
    len(morphology_df)
)

print(
    "Morphology features:",
    len(feature_columns)
)


print(
    "\nFeature columns:"
)

for feature in feature_columns:

    print(
        " -",
        feature
    )


print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)
