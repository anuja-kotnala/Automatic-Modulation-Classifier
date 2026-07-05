import numpy as np
from amc.channel.base import BaseImpairment

class AWGNImpairment(BaseImpairment):
    """
    Additive White Gaussian Noise (AWGN) impairment.
    """

    def __init__(self, snr_db: float = 20.0, seed: int = None):
        self.snr_db = snr_db
        self.seed = seed

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)

        sig_power = np.mean(np.abs(signal) ** 2)
        if sig_power == 0:
            sig_power = 1.0

        noise_power = sig_power / (10 ** (self.snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )
        return signal + noise
