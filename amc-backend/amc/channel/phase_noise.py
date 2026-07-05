import numpy as np
from amc.channel.base import BaseImpairment

class PhaseNoiseImpairment(BaseImpairment):
    """
    Phase Noise impairment modeled as a Wiener process (random walk of phase).
    """

    def __init__(self, std_dev: float = 0.05, seed: int = None):
        """
        Args:
            std_dev: Standard deviation of phase increment at each step (rad).
            seed: Random seed.
        """
        self.std_dev = std_dev
        self.seed = seed

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        n_samples = len(signal)
        if n_samples == 0:
            return signal

        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate phase increments
        phase_increments = np.random.normal(0, self.std_dev, n_samples)
        
        # Integrate to get total phase noise over time (Wiener process)
        phase_noise = np.cumsum(phase_increments)
        
        return signal * np.exp(1j * phase_noise)
