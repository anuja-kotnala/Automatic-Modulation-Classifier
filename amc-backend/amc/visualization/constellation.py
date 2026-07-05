from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from amc.visualization.base import setup_publication_style

def plot_constellation(
    signal: np.ndarray,
    title: str = "Constellation Diagram",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots a publication-quality IQ constellation scatter plot.

    Args:
        signal: 1D complex numpy array of IQ samples.
        title: Title of the plot.
        ax: Optional matplotlib axes.

    Returns:
        plt.Axes: Plot axes.
    """
    setup_publication_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(signal.real, signal.imag, color='#1f77b4', alpha=0.7, edgecolors='none', s=12, label='IQ Samples')
    
    # Reference origin lines
    ax.axhline(0, color='#333333', linewidth=1.0, zorder=1)
    ax.axvline(0, color='#333333', linewidth=1.0, zorder=1)
    
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    
    # Tight bounds for visual balance
    max_val = np.max(np.abs(signal)) * 1.15 if len(signal) > 0 else 1.5
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    
    return ax
