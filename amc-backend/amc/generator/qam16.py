from typing import Optional
import numpy as np
from amc.core.base_generator import BaseModulator
from amc.constants import ModulationType
from amc.utils.dsp_utils import add_awgn
from amc.pulse_shaping.filter_utils import get_pulse_filter

class QAM16Modulator(BaseModulator):
    """
    16-state Quadrature Amplitude Modulation (16QAM) Signal Generator.
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
        super().__init__(ModulationType.QAM16, sample_rate, samples_per_symbol=max(1, sps))
        self.carrier_frequency = carrier_frequency
        self.symbol_rate = symbol_rate
        self.phase_offset = phase_offset
        self.amplitude = amplitude
        self.message_frequency = message_frequency
        self.random_seed = random_seed
        self.roll_off = roll_off
        self.pulse_shaping = pulse_shaping

    def get_constellation(self) -> np.ndarray:
        # Standard 16QAM grid normalized to average unit power (RMS power = 1.0)
        # Coordinates: I, Q in {-3, -1, 1, 3}
        points = []
        for i in [-3, -1, 1, 3]:
            for q in [-3, -1, 1, 3]:
                points.append(complex(i, q))
        # Normalization factor for average power of 10 is sqrt(10)
        return np.array(points, dtype=np.complex64) / np.sqrt(10.0)

    def generate(self, num_samples: int, snr_db: Optional[float] = None) -> np.ndarray:
        """
        Generates 16QAM modulated complex IQ signals with pulse shaping.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        sps = self.samples_per_symbol
        num_symbols = int(np.ceil(num_samples / sps)) + 10

        # Generate random symbol indices (0 to 15)
        symbol_indices = np.random.randint(0, 16, num_symbols)
        
        # Map to 16QAM constellation
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

