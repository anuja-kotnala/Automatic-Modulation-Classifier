import numpy as np

def add_awgn(signal: np.ndarray, target_snr_db: float) -> np.ndarray:
    """
    Applies Additive White Gaussian Noise (AWGN) to a complex baseband signal.

    Args:
        signal: One-dimensional complex signal.
        target_snr_db: Target Signal-to-Noise Ratio (SNR) in dB.

    Returns:
        np.ndarray: Complex signal with added noise.
    """
    sig_power = np.mean(np.abs(signal) ** 2)
    if sig_power == 0:
        sig_power = 1.0
    noise_power = sig_power / (10 ** (target_snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


def design_rrc_filter(num_taps: int, alpha: float, samples_per_symbol: int) -> np.ndarray:
    """
    Designs a Root Raised Cosine (RRC) filter impulse response.

    Args:
        num_taps: Number of taps/coefficients in the filter. Must be odd for symmetry.
        alpha: Roll-off factor (typically between 0.2 and 0.5).
        samples_per_symbol: Upsampling/oversampling factor (T_s = samples_per_symbol).

    Returns:
        np.ndarray: Real filter coefficients.
    """
    # Ensure num_taps is odd for nice symmetrical window
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / float(samples_per_symbol)
    h = np.zeros(len(t))
    
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-9:
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den
            
    # Normalize filter energy to 1
    h /= np.sqrt(np.sum(h ** 2))
    return h


def normalize_signal_power(signal: np.ndarray) -> np.ndarray:
    """
    Normalizes a complex signal to have unit average power (variance = 1).

    Args:
        signal: Complex input signal.

    Returns:
        np.ndarray: Normalized complex signal.
    """
    sig_power = np.mean(np.abs(signal) ** 2)
    if sig_power > 0:
        return signal / np.sqrt(sig_power)
    return signal
