import unittest
import numpy as np
from amc.generator import (
    AMModulator,
    FMModulator,
    BPSKModulator,
    QPSKModulator,
    QAM16Modulator,
    QAM64Modulator,
    OFDMModulator
)
from amc.constants import ModulationType

class TestModulators(unittest.TestCase):

    def test_am_generator(self):
        mod = AMModulator(carrier_frequency=10000.0, sample_rate=100000.0, random_seed=42)
        signal = mod.generate(1024, snr_db=20.0)
        self.assertEqual(len(signal), 1024)
        self.assertEqual(signal.dtype, np.complex64)

    def test_fm_generator(self):
        mod = FMModulator(carrier_frequency=10000.0, sample_rate=100000.0, random_seed=42)
        signal = mod.generate(1024, snr_db=15.0)
        self.assertEqual(len(signal), 1024)
        self.assertEqual(signal.dtype, np.complex64)

    def test_bpsk_generator(self):
        mod = BPSKModulator(carrier_frequency=20000.0, sample_rate=200000.0, symbol_rate=25000.0, random_seed=42)
        signal = mod.generate(1024, snr_db=10.0)
        self.assertEqual(len(signal), 1024)
        self.assertEqual(signal.dtype, np.complex64)
        
        # Test constellation
        const = mod.get_constellation()
        self.assertEqual(len(const), 2)

    def test_qpsk_generator(self):
        mod = QPSKModulator(carrier_frequency=0.0, sample_rate=100000.0, symbol_rate=10000.0, random_seed=42)
        signal = mod.generate(1024, snr_db=None)
        self.assertEqual(len(signal), 1024)
        self.assertEqual(signal.dtype, np.complex64)

    def test_qam16_generator(self):
        mod = QAM16Modulator(carrier_frequency=5000.0, sample_rate=100000.0, symbol_rate=10000.0, random_seed=42)
        signal = mod.generate(512, snr_db=30.0)
        self.assertEqual(len(signal), 512)
        self.assertEqual(signal.dtype, np.complex64)

    def test_qam64_generator(self):
        mod = QAM64Modulator(carrier_frequency=0.0, sample_rate=100000.0, symbol_rate=10000.0, random_seed=42)
        signal = mod.generate(512, snr_db=None)
        self.assertEqual(len(signal), 512)
        self.assertEqual(signal.dtype, np.complex64)

    def test_ofdm_generator(self):
        mod = OFDMModulator(
            carrier_frequency=25000.0,
            sample_rate=1000000.0,
            fft_size=64,
            num_subcarriers=48,
            cyclic_prefix=16,
            random_seed=42
        )
        signal = mod.generate(1000, snr_db=12.0)
        self.assertEqual(len(signal), 1000)
        self.assertEqual(signal.dtype, np.complex64)

if __name__ == "__main__":
    unittest.main()
