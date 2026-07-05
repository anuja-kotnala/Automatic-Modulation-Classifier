from abc import ABC, abstractmethod
import numpy as np

class BasePulseFilter(ABC):
    """Abstract base class for all pulse shaping filters."""

    def __init__(self, samples_per_symbol: int, num_symbols_span: int = 8):
        """
        Args:
            samples_per_symbol: Oversampling factor.
            num_symbols_span: Truncation length in symbols (filter delay / window).
        """
        self.samples_per_symbol = samples_per_symbol
        self.num_symbols_span = num_symbols_span
        self.num_taps = num_symbols_span * samples_per_symbol + 1

    @abstractmethod
    def get_taps(self) -> np.ndarray:
        """
        Generates and returns the filter impulse response coefficients (taps).

        Returns:
            np.ndarray: One-dimensional real float numpy array of coefficients.
        """
        pass

    def shape(self, symbols: np.ndarray) -> np.ndarray:
        """
        Upsamples the symbols sequence and applies pulse shaping filter convolution.

        Args:
            symbols: 1D array of complex communication symbols.

        Returns:
            np.ndarray: Convolved complex signal of length n_symbols * samples_per_symbol.
        """
        # Upsample by inserting sps - 1 zeros between symbols
        n_symbols = len(symbols)
        sps = self.samples_per_symbol
        
        upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
        upsampled[::sps] = symbols

        taps = self.get_taps()
        
        # Convolve (mode='same' aligns filter center with symbols)
        shaped = np.convolve(upsampled, taps, mode='same')
        
        # Ensure convolved output matches the target upsampled length exactly
        return shaped[:n_symbols * sps]

