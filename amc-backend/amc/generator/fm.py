from typing import Optional
import numpy as np
from amc.core.base_generator import BaseModulator
from amc.constants import ModulationType

class FMModulator(BaseModulator):
    """
    Frequency Modulation (FM) Generator.
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
        modulation_index: float = 5.0
    ):
        super().__init__(ModulationType.FM, sample_rate, samples_per_symbol=1)
        self.carrier_frequency = carrier_frequency
        self.symbol_rate = symbol_rate
        self.phase_offset = phase_offset
        self.amplitude = amplitude
        self.message_frequency = message_frequency
        self.random_seed = random_seed
        self.modulation_index = modulation_index

    def generate(self, num_samples: int, snr_db: Optional[float] = None) -> np.ndarray:
        """
        Generates complex IQ data for FM.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        t = np.arange(num_samples) / self.sample_rate

        # Phase is the integral of the message signal
        # For message = cos(2 * pi * f_m * t), integral is sin(2 * pi * f_m * t) / (2 * pi * f_m)
        # FM instantaneous phase: 2 * pi * f_c * t + beta * sin(2 * pi * f_m * t) + phase_offset
        # beta is the modulation index (modulation_index = delta_f / f_m)
        inst_phase = (
            2 * np.pi * self.carrier_frequency * t
            + self.modulation_index * np.sin(2 * np.pi * self.message_frequency * t)
            + self.phase_offset
        )

        iq_signal = self.amplitude * np.exp(1j * inst_phase)

        if snr_db is not None:
            sig_power = np.mean(np.abs(iq_signal) ** 2)
            noise_power = sig_power / (10 ** (snr_db / 10.0))
            noise = np.sqrt(noise_power / 2.0) * (
                np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            )
            iq_signal += noise

        return iq_signal.astype(np.complex64)

    def get_constellation(self) -> np.ndarray:
        raise TypeError("Analog Frequency Modulation (FM) does not have a discrete constellation.")
