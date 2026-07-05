from abc import ABC, abstractmethod
import numpy as np

class BaseImpairment(ABC):
    """Abstract base class for all channel impairments and propagation effects."""

    @abstractmethod
    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Applies the impairment to the complex baseband signal.

        Args:
            signal: 1D complex numpy array of IQ samples.
            sample_rate: Sample rate of the signal in Hz.

        Returns:
            np.ndarray: Impaired 1D complex numpy array of the same datatype.
        """
        pass
