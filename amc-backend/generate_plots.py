import os
import numpy as np
import matplotlib.pyplot as plt

from amc.constants import ModulationType
from amc.visualization import (
    plot_constellation,
    plot_fft,
    plot_psd,
    plot_waterfall,
    plot_spectrogram,
    plot_iq,
    plot_eye_diagram
)

def main():
    dataset_dir = "dataset"
    output_dir = os.path.join("results", "plots")
    os.makedirs(output_dir, exist_ok=True)

    modulations = [mod.value for mod in ModulationType]
    snr_target = 20  # High SNR for clean visualizations
    sample_rate = 1000000.0  # Default 1 MHz

    print(f"Generating publication-quality figures under: {output_dir}")

    for mod in modulations:
        # Resolve target filepath: dataset/raw/<Modulation>/snr_20/sample_000.npy
        file_path = os.path.join(dataset_dir, "raw", mod, f"snr_{snr_target}", "sample_000.npy")
        
        if not os.path.exists(file_path):
            print(f"Warning: Sample file not found at: {file_path}. Skipping {mod}.")
            continue
            
        print(f"Plotting charts for modulation: {mod}...")
        
        # Load sample IQ data
        signal = np.load(file_path)

        # 1. Constellation Diagram
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_constellation(signal, title=f"{mod} Constellation (SNR = {snr_target}dB)", ax=ax)
        fig.savefig(os.path.join(output_dir, f"{mod}_constellation.png"))
        plt.close(fig)

        # 2. FFT Spectrum
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_fft(signal, sample_rate=sample_rate, title=f"{mod} Magnitude Spectrum", ax=ax)
        fig.savefig(os.path.join(output_dir, f"{mod}_fft.png"))
        plt.close(fig)

        # 3. PSD Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_psd(signal, sample_rate=sample_rate, nfft=256, title=f"{mod} Power Spectral Density", ax=ax)
        fig.savefig(os.path.join(output_dir, f"{mod}_psd.png"))
        plt.close(fig)

        # 4. 2D Spectrogram Heatmap
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_spectrogram(signal, sample_rate=sample_rate, nperseg=64, noverlap=32, title=f"{mod} Spectrogram", ax=ax)
        fig.savefig(os.path.join(output_dir, f"{mod}_spectrogram.png"))
        plt.close(fig)

        # 5. Time Domain IQ Waveform
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_iq(signal, sample_rate=sample_rate, max_points=150, title=f"{mod} Waveform", ax=ax)
        fig.savefig(os.path.join(output_dir, f"{mod}_iq.png"))
        plt.close(fig)

        # 6. Eye Diagram (Only for single-carrier digital schemes like BPSK/QPSK/QAM)
        if mod in ["BPSK", "QPSK", "16QAM", "64QAM"]:
            sps = 8  # Default samples per symbol
            fig = plt.figure(figsize=(10, 5))
            plot_eye_diagram(signal, samples_per_symbol=sps, symbols_to_show=2, title=f"{mod} Eye Diagram", fig=fig)
            fig.savefig(os.path.join(output_dir, f"{mod}_eye_diagram.png"))
            plt.close(fig)

        # 7. 3D Waterfall
        # Limit signal length slightly for clean mesh rendering
        waterfall_signal = signal[:512]
        fig = plt.figure(figsize=(10, 6))
        plot_waterfall(waterfall_signal, sample_rate=sample_rate, nperseg=64, noverlap=32, title=f"{mod} 3D Waterfall", fig=fig)
        fig.savefig(os.path.join(output_dir, f"{mod}_waterfall.png"))
        plt.close(fig)

    print("\nAll figures generated and saved successfully.")

if __name__ == "__main__":
    main()
