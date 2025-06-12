import os
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np
from scipy import signal
import pywt
import matplotlib.pyplot as plt

# File paths
input_folder = r"C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_reference\1124"
output_file = r"C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_reference\1124\interpolated_results.txt"


def denoise(time, raw_data):
    """
    Denoises the PPG signal using bandpass filtering, wavelet denoising, and smoothing.
    """
    raw_data = raw_data.astype(float)

    # Bandpass filter
    b, a = signal.butter(5, [0.5, 5], btype='bandpass', fs=51)
    filtered_ppg = signal.filtfilt(b, a, raw_data)

    # Wavelet transform denoising
    coeffs = pywt.wavedec(filtered_ppg, 'db4', level=6)
    thr = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(filtered_ppg)))
    coeffs_denoised = [pywt.threshold(c, thr, mode='soft') for c in coeffs]
    wavelet_denoised = pywt.waverec(coeffs_denoised, 'db4')

    # Smoothing with moving average
    window_size = round(0.01 * 51)
    denoised_ppg = np.convolve(wavelet_denoised, np.ones(window_size) / window_size, mode='same')

    # Align time with denoised data
    min_length = min(len(time), len(denoised_ppg))
    time_fixed = time[:min_length]
    denoised_data_fixed = denoised_ppg[:min_length]

    # Plot for debugging (can be removed in production)
    plt.figure(figsize=(10, 6))
    plt.plot(time_fixed, raw_data[:min_length], label="Raw Data", alpha=0.7)
    plt.plot(time_fixed, denoised_data_fixed, label="Denoised Data", alpha=0.7)
    plt.title("Signal Denoising")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.show()

    return time_fixed, denoised_data_fixed


# Remove existing output file
if os.path.exists(output_file):
    os.remove(output_file)

# Process all .txt files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_file = os.path.join(input_folder, filename)
        print(f"Processing file: {input_file}")
        try:
            # Load data
            data = pd.read_csv(input_file, sep="\t", header=None, names=["time", "value"], usecols=[0, 1], skiprows=2)

            # Parse time
            data["time"] = pd.to_datetime(data["time"], format="%Y/%m/%d %H:%M:%S.%f ", errors="coerce")
            data = data.dropna(subset=["time"])
            data["time_numeric"] = (data["time"] - data["time"].min()).dt.total_seconds()

            # Interpolation
            new_times = pd.date_range(start=data["time"].min(), end=data["time"].max(), freq="10L")
            new_times_numeric = (new_times - data["time"].min()).total_seconds()
            cubic_spline = CubicSpline(data["time_numeric"], data["value"])
            new_values = cubic_spline(new_times_numeric)

            # Denoising
            time_fixed, denoised_data = denoise(new_times_numeric, new_values)

            # Save results
            interpolated_data = pd.DataFrame({"time": new_times[:len(time_fixed)].strftime("%H:%M:%S.%f").str[:-3],
                                              "value": denoised_data})
            interpolated_data.to_csv(output_file, sep="\t", index=False, header=False, mode='a')

        except Exception as e:
            print(f"Error processing file {input_file}: {e}")

print(f"All files have been processed. Results saved to {output_file}.")
