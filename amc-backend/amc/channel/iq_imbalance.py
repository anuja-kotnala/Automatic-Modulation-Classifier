import numpy as np
from amc.channel.base import BaseImpairment

class IQImbalanceImpairment(BaseImpairment):
    """
    IQ Imbalance impairment (amplitude and phase mismatch between I and Q branches).
    Uses the standard baseband equivalent model: y(t) = K1 * x(t) + K2 * x*(t).
    """

    def __init__(self, amplitude_imbalance_db: float = 0.5, phase_imbalance_deg: float = 3.0):
        self.amplitude_imbalance_db = amplitude_imbalance_db
        self.phase_imbalance_deg = phase_imbalance_deg

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        if len(signal) == 0:
            return signal

        # Amplitude imbalance factor
        g = 10 ** (self.amplitude_imbalance_db / 20.0)
        # Phase imbalance in radians
        phi = np.radians(self.phase_imbalance_deg)

        # Coefficients for the baseband model
        # K1 = 0.5 * (1 + g * exp(-j * phi))
        # K2 = 0.5 * (1 - g * exp(j * phi))
        K1 = 0.5 * (1.0 + g * np.exp(-1j * phi))
        K2 = 0.5 * (1.0 - g * np.exp(1j * phi))

        # Impaired signal: K1 * x + K2 * x_conjugate
        return K1 * signal + K2 * np.conj(signal)
