import os
import unittest
import numpy as np
import joblib

from amc.config import AppConfig
from amc.generator.bpsk import BPSKModulator
from amc.channel.channel_pipeline import ChannelPipeline
from amc.features.extractors import FeatureExtractor

class TestEndToEndIntegration(unittest.TestCase):

    def test_pipeline_integration(self):
        # 1. Load config
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.yaml"))
        self.assertTrue(os.path.exists(config_path))
        config = AppConfig.load_from_yaml(config_path)

        # 2. Generate signal
        sps = 8
        modulator = BPSKModulator(
            carrier_frequency=1000.0,
            sample_rate=config.generator.sample_rate,
            symbol_rate=125000.0,
            random_seed=42
        )
        raw_signal = modulator.generate(num_samples=1024)
        self.assertEqual(len(raw_signal), 1024)

        # 3. Apply channel pipeline
        pipeline = ChannelPipeline(config.channel, seed=42)
        impaired_signal = pipeline.apply(raw_signal, config.generator.sample_rate)
        self.assertEqual(len(impaired_signal), 1024)

        # 4. Extract features
        extractor = FeatureExtractor()
        features = extractor.extract(impaired_signal)
        self.assertTrue(len(features) >= 34)

        # 5. ML Predict (if models exist)
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "models"))
        rf_path = os.path.join(models_dir, "randomforest_model.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        le_path = os.path.join(models_dir, "label_encoder.pkl")

        if os.path.exists(rf_path) and os.path.exists(scaler_path) and os.path.exists(le_path):
            rf = joblib.load(rf_path)
            scaler = joblib.load(scaler_path)
            le = joblib.load(le_path)

            # Format to 2D array matching scaler expectations
            feature_names = sorted([k for k in features.keys()])
            feature_vector = np.array([features[k] for k in feature_names]).reshape(1, -1)
            
            scaled_vector = scaler.transform(feature_vector)
            prediction = rf.predict(scaled_vector)
            
            pred_class = le.inverse_transform(prediction)
            self.assertTrue(len(pred_class) > 0)

if __name__ == "__main__":
    unittest.main()
