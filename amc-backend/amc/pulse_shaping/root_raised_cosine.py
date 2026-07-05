import numpy as np
from amc.pulse_shaping.base import BasePulseFilter

class RootRaisedCosineFilter(BasePulseFilter):
    """
    Root Raised Cosine (RRC) pulse shaping filter.
    """

    def __init__(self, samples_per_symbol: int, num_symbols_span: int = 8, roll_off: float = 0.35):
        super().__init__(samples_per_symbol, num_symbols_span)
        self.roll_off = roll_off

    def get_taps(self) -> np.ndarray:
        sps = self.samples_per_symbol
        alpha = self.roll_off
        
        t = np.arange(-self.num_taps // 2 + 1, self.num_taps // 2 + 1) / float(sps)
        h = np.zeros(len(t))

        for i, ti in enumerate(t):
            if ti == 0.0:
                h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
            elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-9:
                h[i] = (alpha / np.sqrt(2.0)) * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
                )
            else:
                num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
                den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
                h[i] = num / den

        # Normalize energy to unit energy
        h /= np.sqrt(np.sum(h ** 2))
        return h.astype(np.float32)
