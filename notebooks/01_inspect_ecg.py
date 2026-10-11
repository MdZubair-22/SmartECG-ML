import wfdb
import matplotlib.pyplot as plt

# Location of MIT-BIH record 100
record_path = r"C:\Users\MrNobuddy\Documents\ML Project\SmartECG-ML\data\raw\100"

# Read ECG signal
record = wfdb.rdrecord(record_path)

print("Sampling frequency:", record.fs)
print("Number of samples:", record.sig_len)
print("Number of channels:", record.n_sig)
print("Signal names:", record.sig_name)

# Read annotations
annotation = wfdb.rdann(record_path, "atr")

print("\nNumber of annotations:", len(annotation.sample))

print("\nFirst 20 annotation symbols:")
print(annotation.symbol[:20])

print("\nFirst 20 annotation sample locations:")
print(annotation.sample[:20])

# Plot ECG
wfdb.plot_wfdb(
    record=record,
    annotation=annotation,
    title="MIT-BIH Record 100"
)

plt.show()