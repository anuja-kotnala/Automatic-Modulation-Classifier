from typing import List, Optional
import numpy as np
from amc.channel.base import BaseImpairment

class MultipathImpairment(BaseImpairment):
    """
    Frequency selective multipath channel modeled as a tapped delay line (FIR filter).
    """

    def __init__(self, delays: List[float], gains_db: List[float], seed: Optional[int] = None):
        """
        Args:
            delays: List of path delays in seconds.
            gains_db: List of path gains in dB.
            seed: Optional seed for random path phases.
        """
        self.delays = delays
        self.gains_db = gains_db
        self.seed = seed

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        n_samples = len(signal)
        if n_samples == 0:
            return signal

        if self.seed is not None:
            np.random.seed(self.seed)

        # Convert gains from dB to linear and associate a random phase with each tap
        gains_linear = 10 ** (np.array(self.gains_db) / 20.0)
        phases = np.random.uniform(0, 2 * np.pi, len(self.delays))
        complex_gains = gains_linear * np.exp(1j * phases)

        # Initialize output signal
        output = np.zeros_like(signal, dtype=np.complex128)

        # Apply tapped delay line
        for delay_sec, gain in zip(self.delays, complex_gains):
            # Calculate delay in integer samples
            delay_samples = int(np.round(delay_sec * sample_rate))
            if delay_samples >= n_samples:
                continue

            # Shift signal and multiply by tap gain
            shifted_signal = np.zeros_like(signal)
            if delay_samples > 0:
                shifted_signal[delay_samples:] = signal[:-delay_samples]
            else:
                shifted_signal = signal
            
            output += gain * shifted_signal

        # Normalize average output power to match input power
        in_power = np.mean(np.abs(signal) ** 2)
        out_power = np.mean(np.abs(output) ** 2)
        if in_power > 0 and out_power > 0:
            output = output * np.sqrt(in_power / out_power)

        return output.astype(np.complex64)
