import numpy as np
from amc.pulse_shaping.base import BasePulseFilter

class RaisedCosineFilter(BasePulseFilter):
    """
    Raised Cosine (RC) pulse shaping filter.
    """

    def __init__(self, samples_per_symbol: int, num_symbols_span: int = 8, roll_off: float = 0.35):
        super().__init__(samples_per_symbol, num_symbols_span)
        self.roll_off = roll_off

    def get_taps(self) -> np.ndarray:
        sps = self.samples_per_symbol
        alpha = self.roll_off
        
        # t runs from -num_symbols_span/2 to +num_symbols_span/2 symbols
        t = np.arange(-self.num_taps // 2 + 1, self.num_taps // 2 + 1) / float(sps)
        h = np.zeros(len(t))

        for i, ti in enumerate(t):
            if ti == 0.0:
                h[i] = 1.0
            elif abs(abs(2.0 * alpha * ti) - 1.0) < 1e-9:
                h[i] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * alpha))
            else:
                h[i] = np.sinc(ti) * np.cos(np.pi * alpha * ti) / (1.0 - (2.0 * alpha * ti) ** 2)

        # Normalize energy to unit energy
        h /= np.sqrt(np.sum(h ** 2))
        return h.astype(np.float32)
