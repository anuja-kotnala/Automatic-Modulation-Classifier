from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.signal import spectrogram
from amc.visualization.base import setup_publication_style

def plot_waterfall(
    signal: np.ndarray,
    sample_rate: float,
    nperseg: int = 128,
    noverlap: int = 64,
    title: str = "3D Spectral Waterfall",
    fig: Optional[plt.Figure] = None
) -> plt.Axes:
    """
    Plots a 3D time-frequency waterfall spectrum.

    Args:
        signal: 1D complex numpy array of IQ samples.
        sample_rate: Sample rate in Hz.
        nperseg: Length of each segment.
        noverlap: Number of overlapping points.
        title: Title of the plot.
        fig: Optional matplotlib figure.

    Returns:
        plt.Axes: Plot axes (3D projection).
    """
    setup_publication_style()
    if fig is None:
        fig = plt.figure(figsize=(10, 6))

    # Compute short-time spectrogram
    f, t_coords, Sxx = spectrogram(
        signal,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False
    )

    # Shift frequencies to DC-centered
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    # Create meshgrid for 3D plotting
    T, F = np.meshgrid(t_coords * 1e3, f / 1e3)  # ms and kHz

    # Add 3D axes
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(
        F, T, Sxx_db,
        cmap=colormaps['viridis'],
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1
    )

    ax.set_xlabel("Frequency (kHz)", labelpad=8)
    ax.set_ylabel("Time (ms)", labelpad=8)
    ax.set_zlabel("Power (dB)", labelpad=8)
    ax.set_title(title)
    
    # Rotate for better perspective
    ax.view_init(elev=30, azim=-45)
    
    return ax
