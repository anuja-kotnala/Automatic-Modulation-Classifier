from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from amc.visualization.base import setup_publication_style

def plot_iq(
    signal: np.ndarray,
    sample_rate: float,
    max_points: int = 200,
    title: str = "IQ Time-Domain Waveform",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots I (real) and Q (imaginary) components of the signal over time.

    Args:
        signal: 1D complex numpy array of IQ samples.
        sample_rate: Sample rate in Hz.
        max_points: Max points to display (truncates if signal is longer).
        title: Title of the plot.
        ax: Optional matplotlib axes.

    Returns:
        plt.Axes: Plot axes.
    """
    setup_publication_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    n_display = min(len(signal), max_points)
    sig_subset = signal[:n_display]
    
    t = np.arange(n_display) / sample_rate

    ax.plot(t * 1e6, sig_subset.real, color='#1f77b4', label='In-phase (I)', linewidth=1.5)
    ax.plot(t * 1e6, sig_subset.imag, color='#ff7f0e', label='Quadrature (Q)', linewidth=1.5, linestyle='--')
    
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return ax
