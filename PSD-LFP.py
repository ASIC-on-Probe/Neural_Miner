import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy import ndimage


# ============================================================
# 1. File settings
# ============================================================

file_path = r"D:\Project\brain electrode\16 channel raw data\HPC spike\neural_segment_1.xlsx"

output_dir = os.path.join(
    os.path.dirname(file_path),
    "LFP_PSD_results"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Analysis parameters
# ============================================================

# LFP sampling rate
FS = 1000.0  # Hz


# Saturation limit
CLIP_LIMIT = 250.0  # µV


# Minimum consecutive saturated samples
MIN_CLIP_LENGTH = 2


# Welch parameters
NPERSEG = 1024
NOVERLAP = 512


# PSD frequency range
FREQ_MIN = 0
FREQ_MAX = 500


# Channels
CHANNELS = [
    "Ch.1",
    "Ch.2",
    "Ch.3",
    "Ch.4"
]


# ============================================================
# 3. Read Excel
# ============================================================

print("=" * 70)
print("Reading LFP data")
print("=" * 70)

df = pd.read_excel(file_path)

print(f"Number of samples: {len(df)}")
print(f"Columns: {list(df.columns)}")


# Check required columns
required_columns = ["Time"] + CHANNELS

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Required column '{col}' was not found."
        )


time = df["Time"].to_numpy(
    dtype=float
)

data = df[CHANNELS].to_numpy(
    dtype=float
)


# ============================================================
# 4. Check sampling frequency
# ============================================================

dt = np.median(
    np.diff(time)
)

fs_from_time = 1.0 / dt

print("\nSampling frequency")
print("-" * 50)

print(
    f"Specified Fs        : "
    f"{FS:.2f} Hz"
)

print(
    f"Calculated from Time: "
    f"{fs_from_time:.2f} Hz"
)


if abs(fs_from_time - FS) > 1:

    print(
        "WARNING: Sampling frequency calculated "
        "from Time differs from specified FS."
    )


# ============================================================
# 5. Basic information
# ============================================================

n_samples, n_channels = data.shape

duration = n_samples / FS

print("\nData information")
print("-" * 50)

print(
    f"Samples             : "
    f"{n_samples}"
)

print(
    f"Channels            : "
    f"{n_channels}"
)

print(
    f"Duration            : "
    f"{duration:.3f} s"
)

print(
    f"Sampling rate       : "
    f"{FS:.1f} Hz"
)

print(
    f"Welch nperseg       : "
    f"{NPERSEG}"
)

print(
    f"Welch noverlap      : "
    f"{NOVERLAP}"
)

print(
    f"Frequency resolution: "
    f"{FS / NPERSEG:.4f} Hz"
)


# ============================================================
# 6. Detect continuous saturation
# ============================================================

print("\n")
print("=" * 70)
print("Detecting saturation segments")
print("=" * 70)


all_clip_segments = {}


for ch_idx, ch_name in enumerate(CHANNELS):

    signal = data[:, ch_idx]

    # Saturation mask
    clip_mask = (
        np.abs(signal) >= CLIP_LIMIT
    )

    total_clip_samples = int(
        np.sum(clip_mask)
    )

    clip_percentage = (
        total_clip_samples /
        len(signal) *
        100
    )

    # Find continuous saturation regions
    labels, num_labels = ndimage.label(
        clip_mask
    )

    segments = []

    for label_id in range(
        1,
        num_labels + 1
    ):

        indices = np.where(
            labels == label_id
        )[0]

        segment_length = len(
            indices
        )

        if segment_length >= MIN_CLIP_LENGTH:

            start_sample = int(
                indices[0]
            )

            end_sample = int(
                indices[-1]
            )

            start_time = (
                time[start_sample]
            )

            end_time = (
                time[end_sample]
            )

            duration_ms = (
                segment_length /
                FS *
                1000
            )

            segments.append(
                {
                    "start_sample":
                        start_sample,

                    "end_sample":
                        end_sample,

                    "length_samples":
                        segment_length,

                    "start_time_s":
                        start_time,

                    "end_time_s":
                        end_time,

                    "duration_ms":
                        duration_ms
                }
            )

    all_clip_segments[
        ch_name
    ] = segments


    print(f"\n{ch_name}")
    print("-" * 50)

    print(
        f"Total clipped samples: "
        f"{total_clip_samples}"
    )

    print(
        f"Clipped percentage: "
        f"{clip_percentage:.4f}%"
    )

    print(
        f"Continuous saturation "
        f"segments: {len(segments)}"
    )

    for i, seg in enumerate(
        segments,
        start=1
    ):

        print(
            f"Segment {i}: "
            f"samples "
            f"{seg['start_sample']}–"
            f"{seg['end_sample']} | "
            f"time "
            f"{seg['start_time_s']:.6f}–"
            f"{seg['end_time_s']:.6f} s | "
            f"{seg['length_samples']} samples | "
            f"{seg['duration_ms']:.3f} ms"
        )


# ============================================================
# 7. Export saturation information
# ============================================================

clip_rows = []


for ch_name, segments in (
    all_clip_segments.items()
):

    for i, seg in enumerate(
        segments,
        start=1
    ):

        clip_rows.append(
            {
                "Channel": ch_name,
                "Segment": i,
                "Start sample":
                    seg["start_sample"],

                "End sample":
                    seg["end_sample"],

                "Length (samples)":
                    seg["length_samples"],

                "Start time (s)":
                    seg["start_time_s"],

                "End time (s)":
                    seg["end_time_s"],

                "Duration (ms)":
                    seg["duration_ms"]
            }
        )


clip_info_df = pd.DataFrame(
    clip_rows
)


clip_excel_path = os.path.join(
    output_dir,
    "saturation_segments.xlsx"
)


clip_info_df.to_excel(
    clip_excel_path,
    index=False
)


print(
    "\nSaturation information saved:"
)

print(
    clip_excel_path
)


# ============================================================
# 8. Mark saturation segments as NaN
# ============================================================

clean_data = data.copy()


for ch_idx, ch_name in enumerate(
    CHANNELS
):

    for seg in all_clip_segments[
        ch_name
    ]:

        start = seg[
            "start_sample"
        ]

        end = seg[
            "end_sample"
        ]

        clean_data[
            start:end + 1,
            ch_idx
        ] = np.nan


# ============================================================
# 9. Plot saturation detection
# ============================================================

print(
    "\nGenerating saturation plots..."
)


for ch_idx, ch_name in enumerate(
    CHANNELS
):

    signal = data[:, ch_idx]

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        signal,
        linewidth=0.7
    )

    # Highlight saturation regions
    for seg in all_clip_segments[
        ch_name
    ]:

        plt.axvspan(
            seg["start_time_s"],
            seg["end_time_s"],
            alpha=0.3
        )

    # Saturation limits
    plt.axhline(
        CLIP_LIMIT,
        linestyle="--",
        linewidth=1
    )

    plt.axhline(
        -CLIP_LIMIT,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Voltage (µV)"
    )

    plt.title(
        f"{ch_name} - LFP Saturation Detection"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    fig_path = os.path.join(
        output_dir,
        f"{ch_name}_saturation_detection.png"
    )

    plt.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. Welch PSD function
# ============================================================

def calculate_psd_from_clean_segments(
    signal,
    fs,
    nperseg,
    noverlap
):

    """
    Calculate Welch PSD from continuous clean segments.

    Saturated regions are excluded.
    """

    valid_mask = np.isfinite(
        signal
    )

    # Identify continuous clean regions
    labels, num_labels = ndimage.label(
        valid_mask
    )

    psd_list = []
    window_counts = []

    frequency = None


    for label_id in range(
        1,
        num_labels + 1
    ):

        indices = np.where(
            labels == label_id
        )[0]

        segment_length = len(
            indices
        )

        # Segment must contain at least
        # one complete Welch window
        if segment_length < nperseg:

            continue

        segment = signal[
            indices
        ]


        # Welch PSD
        f, pxx = welch(
            segment,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density"
        )


        if frequency is None:

            frequency = f


        # Number of Welch windows
        step = (
            nperseg -
            noverlap
        )

        n_windows = (
            (segment_length -
             noverlap)
            //
            step
        )


        if n_windows < 1:

            continue


        psd_list.append(
            pxx
        )

        window_counts.append(
            n_windows
        )


    # No usable segment
    if len(psd_list) == 0:

        return None, None, 0


    psd_array = np.vstack(
        psd_list
    )


    window_counts = np.asarray(
        window_counts,
        dtype=float
    )


    # Weighted average
    combined_psd = np.average(
        psd_array,
        axis=0,
        weights=window_counts
    )


    total_windows = int(
        np.sum(window_counts)
    )


    return (
        frequency,
        combined_psd,
        total_windows
    )


# ============================================================
# 11. Calculate PSD
# ============================================================

print("\n")
print("=" * 70)
print("Calculating LFP Welch PSD")
print("=" * 70)


psd_results = {}


for ch_idx, ch_name in enumerate(
    CHANNELS
):

    print(
        f"\nProcessing {ch_name}..."
    )

    signal = clean_data[
        :, ch_idx
    ]


    f, pxx, n_windows = (
        calculate_psd_from_clean_segments(
            signal,
            FS,
            NPERSEG,
            NOVERLAP
        )
    )


    if f is None:

        print(
            "WARNING: No sufficiently "
            "long clean segment available."
        )

        continue


    psd_results[
        ch_name
    ] = {
        "frequency": f,
        "psd": pxx,
        "n_windows": n_windows
    }


    print(
        f"Valid Welch windows: "
        f"{n_windows}"
    )

    print(
        f"Frequency resolution: "
        f"{f[1] - f[0]:.4f} Hz"
    )


# ============================================================
# 12. Export PSD
# ============================================================

print("\n")
print("=" * 70)
print("Exporting PSD")
print("=" * 70)


if len(psd_results) > 0:

    first_channel = next(
        iter(psd_results)
    )

    f_reference = (
        psd_results[
            first_channel
        ]["frequency"]
    )


    psd_df = pd.DataFrame(
        {
            "Frequency_Hz":
                f_reference
        }
    )


    for ch_name in CHANNELS:

        if ch_name in psd_results:

            psd_df[
                f"{ch_name}_PSD_uV2_per_Hz"
            ] = (
                psd_results[
                    ch_name
                ]["psd"]
            )


    # 0–500 Hz
    frequency_mask = (
        (psd_df["Frequency_Hz"]
         >= FREQ_MIN)
        &
        (psd_df["Frequency_Hz"]
         <= FREQ_MAX)
    )


    psd_0_500_df = psd_df[
        frequency_mask
    ].copy()


    psd_excel_path = os.path.join(
        output_dir,
        "LFP_Welch_PSD_0_500Hz.xlsx"
    )


    psd_0_500_df.to_excel(
        psd_excel_path,
        index=False
    )


    print(
        f"PSD Excel saved to:\n"
        f"{psd_excel_path}"
    )


# ============================================================
# 13. Plot PSD
# ============================================================

print(
    "\nGenerating LFP PSD plot..."
)


plt.figure(
    figsize=(10, 6)
)


for ch_name in CHANNELS:

    if ch_name not in psd_results:

        continue


    f = psd_results[
        ch_name
    ]["frequency"]

    pxx = psd_results[
        ch_name
    ]["psd"]


    frequency_mask = (
        (f >= FREQ_MIN)
        &
        (f <= FREQ_MAX)
    )


    plt.semilogy(
        f[frequency_mask],
        pxx[frequency_mask],
        linewidth=1.2,
        label=ch_name
    )


plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    r"PSD ($\mu V^2$/Hz)"
)

plt.title(
    "LFP Power Spectral Density"
)

plt.xlim(
    FREQ_MIN,
    FREQ_MAX
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()

plt.tight_layout()


psd_fig_path = os.path.join(
    output_dir,
    "LFP_Welch_PSD_0_500Hz.png"
)


plt.savefig(
    psd_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. Final summary
# ============================================================

print("\n")
print("=" * 70)
print("LFP PSD analysis completed")
print("=" * 70)

print(
    f"\nOutput directory:\n"
    f"{output_dir}"
)

print("\nParameters:")
print(
    f"Sampling rate        = "
    f"{FS:.1f} Hz"
)

print(
    f"Saturation limit     = "
    f"±{CLIP_LIMIT:.1f} µV"
)

print(
    f"Minimum clip length  = "
    f"{MIN_CLIP_LENGTH} samples"
)

print(
    f"nperseg              = "
    f"{NPERSEG}"
)

print(
    f"noverlap             = "
    f"{NOVERLAP}"
)

print(
    f"Frequency resolution = "
    f"{FS / NPERSEG:.4f} Hz"
)

print("\nGenerated files:")
print(
    "1. saturation_segments.xlsx"
)
print(
    "2. Ch.1_saturation_detection.png"
)
print(
    "3. Ch.2_saturation_detection.png"
)
print(
    "4. Ch.3_saturation_detection.png"
)
print(
    "5. Ch.4_saturation_detection.png"
)
print(
    "6. LFP_Welch_PSD_0_500Hz.xlsx"
)
print(
    "7. LFP_Welch_PSD_0_500Hz.png"
)
