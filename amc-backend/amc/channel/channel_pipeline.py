from typing import List
import numpy as np
from amc.config import ChannelConfig
from amc.channel.base import BaseImpairment
from amc.channel.awgn import AWGNImpairment
from amc.channel.rayleigh import RayleighFadingImpairment
from amc.channel.rician import RicianFadingImpairment
from amc.channel.frequency_offset import FrequencyOffsetImpairment
from amc.channel.phase_noise import PhaseNoiseImpairment
from amc.channel.iq_imbalance import IQImbalanceImpairment
from amc.channel.timing_offset import TimingOffsetImpairment
from amc.channel.multipath import MultipathImpairment
from amc.channel.clock_drift import ClockDriftImpairment

class ChannelPipeline:
    """
    Orchestrates the sequential application of multiple channel impairments
    based on configuration.
    """

    def __init__(self, config: ChannelConfig, seed: int = None):
        self.config = config
        self.seed = seed
        self.impairments: List[BaseImpairment] = []
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """
        Builds the active impairments chain based on the configuration state.
        Typical physical ordering:
        1. Clock Drift / Resampling
        2. Multipath / Frequency Selective fading
        3. Flat Rayleigh/Rician fading
        4. Timing Offset
        5. IQ Imbalance
        6. Phase Noise
        7. Frequency Offset
        8. AWGN Noise (Added last)
        """
        # 1. Clock Drift
        if self.config.clock_drift.enabled:
            self.impairments.append(ClockDriftImpairment(ppm=self.config.clock_drift.ppm))
            
        # 2. Multipath
        if self.config.multipath.enabled:
            self.impairments.append(MultipathImpairment(
                delays=self.config.multipath.delays,
                gains_db=self.config.multipath.gains_db,
                seed=self.seed
            ))
            
        # 3. Rayleigh / Rician fading
        if self.config.rayleigh.enabled:
            self.impairments.append(RayleighFadingImpairment(
                doppler_freq=self.config.rayleigh.doppler_freq,
                seed=self.seed
            ))
        elif self.config.rician.enabled:
            self.impairments.append(RicianFadingImpairment(
                k_factor=self.config.rician.k_factor,
                doppler_freq=self.config.rician.doppler_freq,
                seed=self.seed
            ))

        # 4. Timing Offset
        if self.config.timing_offset.enabled:
            self.impairments.append(TimingOffsetImpairment(fractional_delay=self.config.timing_offset.fractional_delay))

        # 5. IQ Imbalance
        if self.config.iq_imbalance.enabled:
            self.impairments.append(IQImbalanceImpairment(
                amplitude_imbalance_db=self.config.iq_imbalance.amplitude_imbalance_db,
                phase_imbalance_deg=self.config.iq_imbalance.phase_imbalance_deg
            ))

        # 6. Phase Noise
        if self.config.phase_noise.enabled:
            self.impairments.append(PhaseNoiseImpairment(std_dev=self.config.phase_noise.std_dev, seed=self.seed))

        # 7. Frequency Offset
        if self.config.frequency_offset.enabled:
            self.impairments.append(FrequencyOffsetImpairment(offset_hz=self.config.frequency_offset.offset_hz))

        # 8. AWGN
        if self.config.awgn.enabled:
            self.impairments.append(AWGNImpairment(snr_db=self.config.awgn.snr_db, seed=self.seed))

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Sequentially runs the complex signal through all built channel impairments.
        """
        processed_signal = signal.copy()
        for impairment in self.impairments:
            processed_signal = impairment.apply(processed_signal, sample_rate)
        return processed_signal.astype(np.complex64)
