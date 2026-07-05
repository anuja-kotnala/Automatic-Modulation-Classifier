from typing import Optional
import numpy as np
from amc.core.base_generator import BaseModulator
from amc.constants import ModulationType
from amc.utils.dsp_utils import add_awgn
from amc.pulse_shaping.filter_utils import get_pulse_filter

class QPSKModulator(BaseModulator):
    """
    Quadrature Phase Shift Keying (QPSK) Signal Generator.
    """

    def __init__(
        self,
        carrier_frequency: float = 0.0,
        sample_rate: float = 1000000.0,
        symbol_rate: float = 125000.0,
        phase_offset: float = 0.0,
        amplitude: float = 1.0,
        message_frequency: float = 0.0,
        random_seed: Optional[int] = None,
        roll_off: float = 0.35,
        pulse_shaping: str = "rrc"
    ):
        sps = int(np.round(sample_rate / symbol_rate))
        super().__init__(ModulationType.QPSK, sample_rate, samples_per_symbol=max(1, sps))
        self.carrier_frequency = carrier_frequency
        self.symbol_rate = symbol_rate
        self.phase_offset = phase_offset
        self.amplitude = amplitude
        self.message_frequency = message_frequency
        self.random_seed = random_seed
        self.roll_off = roll_off
        self.pulse_shaping = pulse_shaping

    def get_constellation(self) -> np.ndarray:
        # Standard QPSK constellation: exp(j * (pi/4 + k * pi/2)) normalized to unit power
        return np.array([
            (1.0 + 1j) / np.sqrt(2.0),
            (-1.0 + 1j) / np.sqrt(2.0),
            (-1.0 - 1j) / np.sqrt(2.0),
            (1.0 - 1j) / np.sqrt(2.0)
        ], dtype=np.complex64)

    def generate(self, num_samples: int, snr_db: Optional[float] = None) -> np.ndarray:
        """
        Generates QPSK modulated complex IQ signals with pulse shaping.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        sps = self.samples_per_symbol
        num_symbols = int(np.ceil(num_samples / sps)) + 10

        # Generate random symbols (0 to 3)
        symbol_indices = np.random.randint(0, 4, num_symbols)
        
        # Map to QPSK constellation
        constellation = self.get_constellation()
        symbols = constellation[symbol_indices]

        # Use pulse shaping filter
        pulse_filter = get_pulse_filter(
            filter_type=self.pulse_shaping,
            samples_per_symbol=sps,
            roll_off=self.roll_off
        )
        shaped = pulse_filter.shape(symbols)

        # Slice to required size
        iq_signal = shaped[:num_samples]

        # Apply carrier frequency shift and phase offset
        t = np.arange(num_samples) / self.sample_rate
        carrier = self.amplitude * np.exp(1j * (2 * np.pi * self.carrier_frequency * t + self.phase_offset))
        iq_signal = iq_signal * carrier

        if snr_db is not None:
            iq_signal = add_awgn(iq_signal, snr_db)

        return iq_signal.astype(np.complex64)

