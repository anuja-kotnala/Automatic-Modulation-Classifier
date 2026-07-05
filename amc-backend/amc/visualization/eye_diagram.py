from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from amc.visualization.base import setup_publication_style

def plot_eye_diagram(
    signal: np.ndarray,
    samples_per_symbol: int,
    symbols_to_show: int = 2,
    title: str = "Eye Diagram",
    fig: Optional[plt.Figure] = None
) -> plt.Figure:
    """
    Plots an eye diagram for both In-phase and Quadrature channels.

    Args:
        signal: 1D complex numpy array of IQ samples.
        samples_per_symbol: Samples per symbol factor.
        symbols_to_show: Symbol interval span per trace (typically 2 or 3).
        title: Global plot title.
        fig: Optional matplotlib figure.

    Returns:
        plt.Figure: The matplotlib figure.
    """
    setup_publication_style()
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    else:
        axes = fig.axes
        if len(axes) >= 2:
            ax1, ax2 = axes[0], axes[1]
        else:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, sharey=ax1)

    trace_len = symbols_to_show * samples_per_symbol
    num_traces = len(signal) // trace_len

    t = np.arange(trace_len) / float(samples_per_symbol)

    for i in range(num_traces):
        idx_start = i * trace_len
        idx_end = idx_start + trace_len
        segment = signal[idx_start:idx_end]
        
        ax1.plot(t, segment.real, color='#1f77b4', alpha=0.15, linewidth=0.8)
        ax2.plot(t, segment.imag, color='#ff7f0e', alpha=0.15, linewidth=0.8)

    ax1.set_title("In-Phase (I)")
    ax1.set_xlabel("Time (Symbols)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.set_title("Quadrature (Q)")
    ax2.set_xlabel("Time (Symbols)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
