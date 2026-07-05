from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from amc.constants import ModulationType

class BaseModulator(ABC):
    """Abstract base class for all signal generators and modulators."""

    def __init__(self, modulation_type: ModulationType, sample_rate: float, samples_per_symbol: int):
        self.modulation_type = modulation_type
        self.sample_rate = sample_rate
        self.samples_per_symbol = samples_per_symbol

    @abstractmethod
    def generate(self, num_samples: int, snr_db: Optional[float] = None) -> np.ndarray:
        """
        Generates complex baseband IQ samples for the given modulation type.

        Args:
            num_samples: Total number of complex samples to generate.
            snr_db: Optional SNR in dB to apply AWGN to the signal. If None, signal is noise-free.

        Returns:
            np.ndarray: One-dimensional complex numpy array (dtype=np.complex64).
        """
        pass

    @abstractmethod
    def get_constellation(self) -> np.ndarray:
        """
        Returns the theoretical constellation mapping for this modulation scheme.

        Returns:
            np.ndarray: Complex points mapping symbols to IQ plane.
        """
        pass

