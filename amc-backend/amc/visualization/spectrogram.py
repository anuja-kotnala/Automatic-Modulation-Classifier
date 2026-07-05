from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from amc.visualization.base import setup_publication_style

def plot_spectrogram(
    signal: np.ndarray,
    sample_rate: float,
    nperseg: int = 128,
    noverlap: int = 64,
    title: str = "Spectrogram (STFT)",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots a 2D spectrogram heatmap of the signal.

    Args:
        signal: 1D complex numpy array of IQ samples.
        sample_rate: Sample rate in Hz.
        nperseg: Length of each segment.
        noverlap: Number of overlapping points.
        title: Title of the plot.
        ax: Optional matplotlib axes.

    Returns:
        plt.Axes: Plot axes.
    """
    setup_publication_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Compute spectrogram
    f, t_coords, Sxx = spectrogram(
        signal,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False
    )

    # Center frequencies
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    # Plot colormesh
    pcm = ax.pcolormesh(
        t_coords * 1e3,  # ms
        f / 1e3,         # kHz
        Sxx_db,
        shading='gouraud',
        cmap='viridis'
    )

    # Add colorbar
    fig = ax.get_figure()
    if fig:
        fig.colorbar(pcm, ax=ax, label="Power (dB)")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    return ax
