from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from amc.visualization.base import setup_publication_style

def plot_psd(
    signal: np.ndarray,
    sample_rate: float,
    nfft: int = 256,
    title: str = "Power Spectral Density (Welch's Method)",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots the Power Spectral Density (PSD) of the complex signal.

    Args:
        signal: 1D complex numpy array of IQ samples.
        sample_rate: Sample rate in Hz.
        nfft: FFT size for window segments.
        title: Title of the plot.
        ax: Optional matplotlib axes.

    Returns:
        plt.Axes: Plot axes.
    """
    setup_publication_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Welch method for complex signals (return_onesided=False)
    freqs, psd = welch(
        signal,
        fs=sample_rate,
        window='hann',
        nperseg=min(len(signal), nfft),
        nfft=nfft,
        return_onesided=False
    )

    # Shift frequencies and PSD for centered display (DC in middle)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    
    psd_db = 10 * np.log10(psd + 1e-12)

    ax.plot(freqs / 1e3, psd_db, color='#d62728', linewidth=1.2)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(title)
    return ax
