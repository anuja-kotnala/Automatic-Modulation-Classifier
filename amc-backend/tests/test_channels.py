import unittest
import numpy as np
from amc.config import ChannelConfig, AppConfig
from amc.channel import (
    AWGNImpairment,
    RayleighFadingImpairment,
    RicianFadingImpairment,
    FrequencyOffsetImpairment,
    PhaseNoiseImpairment,
    IQImbalanceImpairment,
    TimingOffsetImpairment,
    MultipathImpairment,
    ClockDriftImpairment,
    ChannelPipeline
)

class TestChannels(unittest.TestCase):

    def setUp(self):
        # Generate 1024 complex test signal (QPSK-like)
        self.sample_rate = 1000000.0
        self.signal = (np.random.choice([-1, 1], 1024) + 1j * np.random.choice([-1, 1], 1024)).astype(np.complex64) / np.sqrt(2.0)

    def test_awgn(self):
        imp = AWGNImpairment(snr_db=10.0, seed=42)
        out = imp.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)
        self.assertFalse(np.array_equal(self.signal, out))

    def test_rayleigh(self):
        imp = RayleighFadingImpairment(doppler_freq=50.0, seed=42)
        out = imp.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)

    def test_rician(self):
        imp = RicianFadingImpairment(k_factor=5.0, doppler_freq=20.0, seed=42)
        out = imp.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)

    def test_frequency_offset(self):
        imp = FrequencyOffsetImpairment(offset_hz=100.0)
        out = imp.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)

    def test_phase_noise(self):
        imp = PhaseNoiseImpairment(std_dev=0.02, seed=42)
        out = imp.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)

    def test_iq_imbalance(self):
        imp = IQImbalanceImpairment(amplitude_imbalance_db=1.0, phase_imbalance_deg=5.0)
        out = imp.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)

    def test_timing_offset(self):
        imp = TimingOffsetImpairment(fractional_delay=0.4)
        out = imp.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)
        self.assertEqual(out.dtype, np.complex128) # FFT returns float64 double precision complex by default

    def test_multipath(self):
        imp = MultipathImpairment(delays=[0.0, 1e-6], gains_db=[0.0, -6.0], seed=42)
        out = imp.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)

    def test_clock_drift(self):
        imp = ClockDriftImpairment(ppm=50.0)
        out = imp.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)

    def test_pipeline(self):
        # Configure everything enabled
        config = ChannelConfig()
        config.awgn.enabled = True
        config.rayleigh.enabled = True
        config.frequency_offset.enabled = True
        config.phase_noise.enabled = True
        config.iq_imbalance.enabled = True
        config.timing_offset.enabled = True
        config.multipath.enabled = True
        config.clock_drift.enabled = True

        pipeline = ChannelPipeline(config, seed=42)
        self.assertEqual(len(pipeline.impairments), 8)
        
        out = pipeline.apply(self.signal, self.sample_rate)
        self.assertEqual(len(out), 1024)
        self.assertEqual(out.dtype, np.complex64)

if __name__ == "__main__":
    unittest.main()
