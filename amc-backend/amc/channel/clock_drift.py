import numpy as np
from amc.channel.base import BaseImpairment

class ClockDriftImpairment(BaseImpairment):
    """
    Clock Drift impairment (sample rate drift/mismatch in transmitter or receiver oscillator).
    Simulated using linear interpolation/resampling.
    """

    def __init__(self, ppm: float = 20.0):
        """
        Args:
            ppm: Parts-per-million offset (e.g. 20.0 means drift factor of 20e-6).
        """
        self.ppm = ppm

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        n_samples = len(signal)
        if n_samples == 0 or self.ppm == 0.0:
            return signal

        # Drift ratio: 1 + ppm * 1e-6
        drift_factor = 1.0 + (self.ppm * 1e-6)
        
        # Original time indices
        t_orig = np.arange(n_samples)
        
        # Resampled time indices (slightly stretched or compressed)
        t_new = np.arange(0, n_samples, drift_factor)
        
        # Interpolate real and imaginary parts
        real_interp = np.interp(t_new, t_orig, signal.real)
        imag_interp = np.interp(t_new, t_orig, signal.imag)
        
        resampled = real_interp + 1j * imag_interp
        
        # Adjust length back to the original length to maintain block structures
        if len(resampled) < n_samples:
            # Pad with zeros
            pad_len = n_samples - len(resampled)
            resampled = np.pad(resampled, (0, pad_len), mode='constant')
        else:
            # Truncate
            resampled = resampled[:n_samples]
            
        return resampled
