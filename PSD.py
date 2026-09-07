import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path


# ============================================================
# 1. File path
# ============================================================

file_path = r"D:\Project\brain electrode\Raw data\Fig 4\Figure 4b.xlsx"

output_dir = Path(file_path).parent / "PSD_analysis_1"
output_dir.mkdir(exist_ok=True)


# ============================================================
# 2. Read Excel
# ============================================================

df = pd.read_excel(file_path)

time = df["Time"].to_numpy(dtype=float)

channels = ["Ch.1", "Ch.2", "Ch.3", "Ch.4"]

data = df[channels].to_numpy(dtype=float)


# ============================================================
# 3. Sampling rate
# ============================================================

fs = 40000.0  # Hz

# Check the sampling interval from Time column
dt = np.median(np.diff(time))
fs_from_time = 1.0 / dt

print("========================================")
print("Data information")
print("========================================")

print(f"Number of samples : {len(time)}")
print(f"Duration          : {time[-1] - time[0]:.4f} s")
print(f"Sampling rate     : {fs:.1f} Hz")
print(f"Time-derived Fs   : {fs_from_time:.1f} Hz")

if abs(fs_from_time - fs) / fs > 0.01:
    print("\nWARNING:")
    print("The sampling rate calculated from the Time column")
    print("differs from 40 kHz by more than 1%.")

print("========================================")


# ============================================================
# 4. Remove DC component
# ============================================================

data = data - np.mean(data, axis=0)


# ============================================================
# 5. Welch PSD parameters
# ============================================================

nperseg = 4096
noverlap = 2048

frequency_resolution = fs / nperseg

print("\nPSD parameters")
print("========================================")
print(f"Window             : Hann")
print(f"nperseg            : {nperseg}")
print(f"noverlap           : {noverlap}")
print(f"Frequency resolution: {frequency_resolution:.3f} Hz")
print(f"Nyquist frequency  : {fs / 2:.0f} Hz")
print("========================================")


# ============================================================
# 6. Calculate Welch PSD
# ============================================================

psd_data = {}

for i, ch in enumerate(channels):

    f, Pxx = welch(
        data[:, i],
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density"
    )

    psd_data[ch] = Pxx


# ============================================================
# 7. Export PSD data
# ============================================================

psd_df = pd.DataFrame({
    "Frequency (Hz)": f
})

for ch in channels:
    psd_df[f"{ch} PSD (uV^2/Hz)"] = psd_data[ch]

psd_file = output_dir / "SPKC_PSD.xlsx"

psd_df.to_excel(
    psd_file,
    index=False
)

print(f"\nPSD data saved to:")
print(psd_file)


# ============================================================
# 8. Plot PSD: linear y-axis
# ============================================================

plt.figure(figsize=(8, 5))

for ch in channels:

    plt.plot(
        f,
        psd_data[ch],
        linewidth=1.2,
        label=ch
    )

plt.xlim(0, 5000)

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Power spectral density ($\mathrm{\mu V^2/Hz}$)")

plt.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    output_dir / "SPKC_PSD_linear.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. Plot PSD: logarithmic y-axis
# ============================================================

plt.figure(figsize=(8, 5))

for ch in channels:

    plt.plot(
        f,
        psd_data[ch],
        linewidth=1.2,
        label=ch
    )

plt.xlim(500, 5000)

plt.yscale("log")

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Power spectral density ($\mathrm{\mu V^2/Hz}$)")

plt.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    output_dir / "SPKC_PSD_log.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. Print basic PSD information
# ============================================================

print("\nPSD summary")
print("========================================")

for ch in channels:

    Pxx = psd_data[ch]

    # Frequency range: 500–5000 Hz
    mask = (f >= 500) & (f <= 5000)

    peak_index = np.argmax(Pxx[mask])

    f_selected = f[mask]
    P_selected = Pxx[mask]

    peak_frequency = f_selected[peak_index]
    peak_psd = P_selected[peak_index]

    print(
        f"{ch}: "
        f"peak frequency = {peak_frequency:.2f} Hz, "
        f"peak PSD = {peak_psd:.4e} uV^2/Hz"
    )

print("========================================")
print("\nAnalysis completed.")
