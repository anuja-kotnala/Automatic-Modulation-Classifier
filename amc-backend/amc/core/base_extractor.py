from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extraction methodologies (e.g., spectral, high-order statistics, wavelets)."""

    @abstractmethod
    def extract(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Extracts features from the given complex IQ signal.

        Args:
            signal: One-dimensional complex numpy array (baseband IQ samples).

        Returns:
            Dict[str, Any]: Dictionary mapping feature names to their extracted scalar/vector values.
        """
        pass
