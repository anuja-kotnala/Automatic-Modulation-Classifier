import numpy as np
from amc.channel.base import BaseImpairment

class TimingOffsetImpairment(BaseImpairment):
    """
    Timing Offset (fractional sample delay) impairment.
    Uses frequency-domain phase rotation for fractional phase shifting.
    """

    def __init__(self, fractional_delay: float = 0.25):
        """
        Args:
            fractional_delay: Fractional delay in samples (e.g. 0.25 means delay of 0.25 samples).
        """
        self.fractional_delay = fractional_delay

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        n_samples = len(signal)
        if n_samples == 0 or self.fractional_delay == 0.0:
            return signal

        # Perform FFT
        sig_fft = np.fft.fft(signal)
        
        # Calculate frequencies
        # normalized frequency from -0.5 to 0.5 (scaled by n_samples)
        freqs = np.fft.fftfreq(n_samples)
        
        # Apply phase shift representing the time-domain delay: H(f) = exp(-j * 2 * pi * f * delay)
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * self.fractional_delay)
        
        # IFFT back to time domain
        return np.fft.ifft(sig_fft * phase_shift)
