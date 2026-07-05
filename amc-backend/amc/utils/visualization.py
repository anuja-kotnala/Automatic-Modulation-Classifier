from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt

def plot_constellation(signal: np.ndarray, title: str = "Constellation Diagram", ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plots the constellation diagram (I/Q scatter plot) of the complex signal.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(signal.real, signal.imag, color='blue', alpha=0.6, edgecolors='none', s=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=1.2, alpha=0.7)
    ax.axvline(0, color='black', linewidth=1.2, alpha=0.7)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return ax


def plot_psd(
    frequencies: np.ndarray,
    psd_db: np.ndarray,
    title: str = "Power Spectral Density",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots the Power Spectral Density (PSD) of a signal.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(frequencies, psd_db, color='red', linewidth=1.5)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB/Hz)")
    ax.set_title(title)
    return ax


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Plots a heat map of the classification confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(confusion_matrix.shape[1]),
        yticks=np.arange(confusion_matrix.shape[0]),
        xticklabels=classes, 
        yticklabels=classes,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    # Rotate label ticks
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if confusion_matrix.dtype == float else 'd'
    thresh = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j, i, format(confusion_matrix[i, j], fmt),
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black"
            )
    fig.tight_layout()
    return fig


def plot_filter_response(
    taps: np.ndarray,
    samples_per_symbol: int,
    title: str = "Pulse Shaping Filter Response"
) -> plt.Figure:
    """
    Plots the impulse response and frequency response of a pulse shaping filter.

    Args:
        taps: Array of filter coefficients.
        samples_per_symbol: Samples per symbol factor.
        title: Global plot title.

    Returns:
        plt.Figure: The matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    # Plot Impulse Response (taps)
    t = np.arange(-len(taps) // 2 + 1, len(taps) // 2 + 1) / float(samples_per_symbol)
    ax1.stem(t, taps, basefmt=" ")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlabel("Time (Symbol Periods)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Impulse Response")

    # Plot Frequency Response
    n_fft = 1024
    w, h = scipy_freqz(taps, 1, worN=n_fft, whole=True)
    # Align frequency axis
    freqs = (np.arange(n_fft) / n_fft) - 0.5
    h_shift = np.fft.fftshift(h)
    response_db = 20 * np.log10(np.abs(h_shift) + 1e-9)

    ax2.plot(freqs * samples_per_symbol, response_db, color='green')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlabel("Normalized Frequency (x Symbol Rate)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim([-60, 5])
    ax2.set_title("Frequency Response")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

# Local simple implementation of freqz to avoid strict scipy dependencies if not compiled yet
def scipy_freqz(b: np.ndarray, a: int, worN: int, whole: bool = True):
    # standard discrete time frequency response using FFT
    w = np.linspace(0, 2 * np.pi, worN, endpoint=False)
    # shift frequencies if centered whole
    h = np.fft.fft(b, worN)
    return w, h

