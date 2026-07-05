import unittest
import numpy as np
from amc.pulse_shaping import (
    RaisedCosineFilter,
    RootRaisedCosineFilter,
    RectangularFilter,
    get_pulse_filter
)

class TestPulseShaping(unittest.TestCase):

    def test_raised_cosine(self):
        sps = 8
        filt = RaisedCosineFilter(samples_per_symbol=sps, num_symbols_span=6, roll_off=0.35)
        taps = filt.get_taps()
        self.assertEqual(len(taps), 6 * sps + 1)
        self.assertTrue(np.allclose(np.sum(taps**2), 1.0, atol=1e-5))

        symbols = np.array([1+0j, -1+0j, 1+0j])
        shaped = filt.shape(symbols)
        self.assertEqual(len(shaped), len(symbols) * sps)

    def test_root_raised_cosine(self):
        sps = 4
        filt = RootRaisedCosineFilter(samples_per_symbol=sps, num_symbols_span=8, roll_off=0.25)
        taps = filt.get_taps()
        self.assertEqual(len(taps), 8 * sps + 1)
        self.assertTrue(np.allclose(np.sum(taps**2), 1.0, atol=1e-5))

    def test_rectangular(self):
        sps = 10
        filt = RectangularFilter(samples_per_symbol=sps)
        taps = filt.get_taps()
        self.assertEqual(len(taps), sps)
        self.assertTrue(np.allclose(np.sum(taps**2), 1.0, atol=1e-5))

        symbols = np.array([1+1j, -1-1j])
        shaped = filt.shape(symbols)
        self.assertEqual(len(shaped), len(symbols) * sps)

    def test_factory(self):
        rrc = get_pulse_filter("root_raised_cosine", 8, roll_off=0.35)
        self.assertIsInstance(rrc, RootRaisedCosineFilter)

        rc = get_pulse_filter("rc", 4)
        self.assertIsInstance(rc, RaisedCosineFilter)

        rect = get_pulse_filter("rectangular", 8)
        self.assertIsInstance(rect, RectangularFilter)

        with self.assertRaises(ValueError):
            get_pulse_filter("unknown", 8)

if __name__ == "__main__":
    unittest.main()
