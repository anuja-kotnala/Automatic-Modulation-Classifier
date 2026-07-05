from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple
import numpy as np
from amc.constants import ModulationType

class BaseClassifier(ABC):
    """Abstract base class for AMC classifier models (ML/DL/Decision Trees)."""

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Trains the classification model.

        Args:
            x_train: Training inputs (features or raw signals).
            y_train: Training labels (encoded modulation classes).
            x_val: Validation inputs.
            y_val: Validation labels.

        Returns:
            Dict[str, Any]: Dictionary containing training metrics/history.
        """
        pass

    @abstractmethod
    def predict(self, signal_or_features: np.ndarray) -> Tuple[ModulationType, float]:
        """
        Classifies the input signal/features and predicts the modulation type.

        Args:
            signal_or_features: Preprocessed IQ signal or extracted feature array.

        Returns:
            Tuple[ModulationType, float]: Predicted ModulationType and confidence score/probability.
        """
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """Saves the trained model state to disk."""
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        """Loads the model state from disk."""
        pass
