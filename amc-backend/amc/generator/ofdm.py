from typing import Optional, List
import numpy as np
from amc.core.base_generator import BaseModulator
from amc.constants import ModulationType
from amc.utils.dsp_utils import add_awgn

class OFDMModulator(BaseModulator):
    """
    Orthogonal Frequency Division Multiplexing (OFDM) Signal Generator.
    """

    def __init__(
        self,
        carrier_frequency: float = 0.0,
        sample_rate: float = 1000000.0,
        symbol_rate: float = 10000.0,  # Unused directly, baseband symbol rate derived from parameters
        phase_offset: float = 0.0,
        amplitude: float = 1.0,
        message_frequency: float = 0.0,  # Unused for digital, kept for common interface
        random_seed: Optional[int] = None,
        fft_size: int = 64,
        num_subcarriers: int = 48,
        cyclic_prefix: int = 16,
        pilot_carriers: Optional[List[int]] = None,
        pilot_value: complex = 1.0 + 0j
    ):
        super().__init__(ModulationType.OFDM, sample_rate, samples_per_symbol=fft_size + cyclic_prefix)
        self.carrier_frequency = carrier_frequency
        self.symbol_rate = symbol_rate
        self.phase_offset = phase_offset
        self.amplitude = amplitude
        self.message_frequency = message_frequency
        self.random_seed = random_seed
        self.fft_size = fft_size
        self.num_subcarriers = num_subcarriers
        self.cyclic_prefix = cyclic_prefix
        
        # If no pilot carriers specified, set default ones (e.g. standard indices like -21, -7, 7, 21)
        if pilot_carriers is None:
            self.pilot_carriers = [-21, -7, 7, 21]
        else:
            self.pilot_carriers = pilot_carriers
            
        self.pilot_value = pilot_value

    def get_constellation(self) -> np.ndarray:
        # Standard QPSK constellation for mapping OFDM subcarriers
        return np.array([
            (1.0 + 1j) / np.sqrt(2.0),
            (-1.0 + 1j) / np.sqrt(2.0),
            (-1.0 - 1j) / np.sqrt(2.0),
            (1.0 - 1j) / np.sqrt(2.0)
        ], dtype=np.complex64)

    def generate(self, num_samples: int, snr_db: Optional[float] = None) -> np.ndarray:
        """
        Generates OFDM complex IQ signal stream.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # OFDM block size including Cyclic Prefix
        block_size = self.fft_size + self.cyclic_prefix
        num_blocks = int(np.ceil(num_samples / block_size))

        # We map subcarriers centering around the DC carrier (index 0 is DC, which we leave empty/0)
        # Total subcarriers = num_subcarriers
        # Active subcarriers indices in centered array:
        half_sc = self.num_subcarriers // 2
        active_indices = list(range(-half_sc, 0)) + list(range(1, half_sc + 1))

        # Separate active indices into pilot and data carriers
        pilot_sc = [x for x in self.pilot_carriers if x in active_indices]
        data_sc = [x for x in active_indices if x not in pilot_sc]

        # Map centered subcarrier index to FFT indices (0 to fft_size - 1)
        # Negative indices wrap around: -k -> fft_size - k
        def to_fft_idx(sc_idx: int) -> int:
            if sc_idx < 0:
                return self.fft_size + sc_idx
            return sc_idx

        # Initialize the output signal array
        time_domain_signal = []

        for _ in range(num_blocks):
            # Frequency domain symbol buffer
            freq_symbols = np.zeros(self.fft_size, dtype=np.complex64)

            # Insert pilots
            for p in pilot_sc:
                freq_symbols[to_fft_idx(p)] = self.pilot_value

            # Generate random QPSK symbols for data carriers
            data_bits = np.random.randint(0, 4, len(data_sc))
            data_symbols = self.get_constellation()[data_bits]

            for idx, d in enumerate(data_sc):
                freq_symbols[to_fft_idx(d)] = data_symbols[idx]

            # Perform IFFT to transition to time domain
            # We scale standard IFFT output to normalize energy
            time_symbol = np.fft.ifft(freq_symbols) * np.sqrt(self.fft_size)

            # Insert Cyclic Prefix (copy the last 'cyclic_prefix' samples of time_symbol to the front)
            cp_part = time_symbol[-self.cyclic_prefix:]
            ofdm_block = np.concatenate([cp_part, time_symbol])

            time_domain_signal.append(ofdm_block)

        # Flatten list to array and slice to the target length
        iq_signal = np.concatenate(time_domain_signal)[:num_samples]

        # Apply carrier frequency shift, amplitude, and phase offset
        t = np.arange(num_samples) / self.sample_rate
        carrier = self.amplitude * np.exp(1j * (2 * np.pi * self.carrier_frequency * t + self.phase_offset))
        iq_signal = iq_signal * carrier

        if snr_db is not None:
            iq_signal = add_awgn(iq_signal, snr_db)

        return iq_signal.astype(np.complex64)
