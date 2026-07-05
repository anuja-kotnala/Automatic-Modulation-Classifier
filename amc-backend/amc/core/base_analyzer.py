from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class BaseSpectrumAnalyzer(ABC):
    """Abstract base class for spectrum analyzer modules."""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    @abstractmethod
    def compute_psd(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Power Spectral Density (PSD) of the input signal.

        Args:
            signal: One-dimensional complex IQ signal.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Frequencies and power levels (in dB/Hz or linear scale).
        """
        pass

    @abstractmethod
    def estimate_occupied_bandwidth(self, signal: np.ndarray, power_percentage: float = 99.0) -> float:
        """
        Estimates the occupied bandwidth of the signal.

        Args:
            signal: One-dimensional complex IQ signal.
            power_percentage: Percentage of power to contain (default 99.0%).

        Returns:
            float: Estimated bandwidth in Hz.
        """
        pass

    @abstractmethod
    def detect_peaks(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Detects spectral peaks in the signal frequency domain.

        Args:
            signal: One-dimensional complex IQ signal.

        Returns:
            Dict[str, Any]: Peak frequencies, peak powers, and related statistics.
        """
        pass
