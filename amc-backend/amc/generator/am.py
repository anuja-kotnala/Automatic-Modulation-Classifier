from typing import Optional
import numpy as np
from amc.core.base_generator import BaseModulator
from amc.constants import ModulationType

class AMModulator(BaseModulator):
    """
    Amplitude Modulation (AM) Generator.
    Generates complex IQ baseband or passband signals.
    """

    def __init__(
        self,
        carrier_frequency: float = 0.0,
        sample_rate: float = 1000000.0,
        symbol_rate: float = 10000.0,  # Unused for analog, kept for common interface
        phase_offset: float = 0.0,
        amplitude: float = 1.0,
        message_frequency: float = 1000.0,
        random_seed: Optional[int] = None,
        modulation_index: float = 0.8
    ):
        super().__init__(ModulationType.AM, sample_rate, samples_per_symbol=1)
        self.carrier_frequency = carrier_frequency
        self.symbol_rate = symbol_rate
        self.phase_offset = phase_offset
        self.amplitude = amplitude
        self.message_frequency = message_frequency
        self.random_seed = random_seed
        self.modulation_index = modulation_index

    def generate(self, num_samples: int, snr_db: Optional[float] = None) -> np.ndarray:
        """
        Generates complex IQ data for AM.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        t = np.arange(num_samples) / self.sample_rate

        # Generate message signal (sinusoid or complex tone)
        message = np.cos(2 * np.pi * self.message_frequency * t)

        # Complex baseband AM: A * (1 + m_index * message) * e^(j * phase_offset)
        # Shift to carrier_frequency by multiplying with e^(j * 2 * pi * f_c * t)
        carrier_phase = 2 * np.pi * self.carrier_frequency * t + self.phase_offset
        iq_signal = self.amplitude * (1.0 + self.modulation_index * message) * np.exp(1j * carrier_phase)

        if snr_db is not None:
            # Calculate signal power and add AWGN
            sig_power = np.mean(np.abs(iq_signal) ** 2)
            noise_power = sig_power / (10 ** (snr_db / 10.0))
            noise = np.sqrt(noise_power / 2.0) * (
                np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            )
            iq_signal += noise

        return iq_signal.astype(np.complex64)

    def get_constellation(self) -> np.ndarray:
        raise TypeError("Analog Amplitude Modulation (AM) does not have a discrete constellation.")
