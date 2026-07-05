from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from amc.visualization.base import setup_publication_style

def plot_fft(
    signal: np.ndarray,
    sample_rate: float,
    title: str = "FFT Magnitude Spectrum",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots the magnitude spectrum of the complex signal using direct FFT.

    Args:
        signal: 1D complex numpy array of IQ samples.
        sample_rate: Sample rate in Hz.
        title: Title of the plot.
        ax: Optional matplotlib axes.

    Returns:
        plt.Axes: Plot axes.
    """
    setup_publication_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    n_samples = len(signal)
    if n_samples == 0:
        return ax

    # Compute FFT and shift center to DC
    fft_vals = np.fft.fftshift(np.fft.fft(signal))
    fft_mag = 20 * np.log10(np.abs(fft_vals) / n_samples + 1e-9)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1.0 / sample_rate))

    ax.plot(freqs / 1e3, fft_mag, color='#2ca02c', linewidth=1.0)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.set_ylim([np.min(fft_mag) - 5, np.max(fft_mag) + 5])
    return ax
