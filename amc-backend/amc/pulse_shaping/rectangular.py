import numpy as np
from amc.pulse_shaping.base import BasePulseFilter

class RectangularFilter(BasePulseFilter):
    """
    Rectangular pulse shaping filter (Zero-Order Hold).
    """

    def __init__(self, samples_per_symbol: int, num_symbols_span: int = 1):
        # Span is exactly 1 symbol, meaning number of taps is samples_per_symbol
        super().__init__(samples_per_symbol, num_symbols_span)
        self.num_taps = samples_per_symbol

    def get_taps(self) -> np.ndarray:
        # Vector of ones normalized to unit energy
        h = np.ones(self.num_taps, dtype=np.float32)
        h /= np.sqrt(np.sum(h ** 2))
        return h
