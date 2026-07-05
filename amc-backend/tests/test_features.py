import unittest
import numpy as np
from amc.features import (
    FeatureExtractor,
    CumulantFeatureExtractor,
    SpectralFeatureExtractor,
    WaveletFeatureExtractor
)

class TestFeatures(unittest.TestCase):

    def setUp(self):
        # 256 complex QPSK-like samples
        self.signal = (np.random.choice([-1, 1], 256) + 1j * np.random.choice([-1, 1], 256)).astype(np.complex64) / np.sqrt(2.0)

    def test_feature_extractor_returns_correct_number_of_features(self):
        extractor = FeatureExtractor()
        features = extractor.extract(self.signal)
        
        # Verify number of features is in the range of 30-40 (we have 34 features)
        self.assertTrue(30 <= len(features) <= 40, f"Found {len(features)} features, expected between 30 and 40.")

        # Ensure no NaNs or Infs
        for name, val in features.items():
            self.assertFalse(np.isnan(val), f"Feature {name} is NaN.")
            self.assertFalse(np.isinf(val), f"Feature {name} is Inf.")

    def test_sub_extractors(self):
        # Cumulants
        cum_ext = CumulantFeatureExtractor(orders=[20, 21, 40, 42])
        cum_feats = cum_ext.extract(self.signal)
        self.assertEqual(len(cum_feats), 4)
        self.assertIn("C20", cum_feats)

        # Spectral
        spec_ext = SpectralFeatureExtractor()
        spec_feats = spec_ext.extract(self.signal)
        self.assertTrue(len(spec_feats) > 0)
        self.assertIn("spectral_entropy", spec_feats)

        # Wavelet
        wav_ext = WaveletFeatureExtractor()
        wav_feats = wav_ext.extract(self.signal)
        self.assertIn("wavelet_mean", wav_feats)

if __name__ == "__main__":
    unittest.main()
