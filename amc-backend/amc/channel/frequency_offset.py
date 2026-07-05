import numpy as np
from amc.channel.base import BaseImpairment

class FrequencyOffsetImpairment(BaseImpairment):
    """
    Carrier Frequency Offset (CFO) impairment.
    """

    def __init__(self, offset_hz: float = 50.0):
        self.offset_hz = offset_hz

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        n_samples = len(signal)
        if n_samples == 0:
            return signal

        t = np.arange(n_samples) / sample_rate
        offset_phase = 2 * np.pi * self.offset_hz * t
        
        # Apply phase rotation
        return signal * np.exp(1j * offset_phase)
