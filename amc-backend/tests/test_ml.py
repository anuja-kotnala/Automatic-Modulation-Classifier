import os
import unittest
import joblib
from sklearn.preprocessing import StandardScaler

class TestMachineLearningModels(unittest.TestCase):

    def test_model_loading(self):
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "models"))
        
        # Verify fitted scaler loads
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        self.assertTrue(os.path.exists(scaler_path), f"Scaler file not found at: {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        self.assertIsInstance(scaler, StandardScaler)

        # Verify a classifier model loads (e.g. RandomForest)
        rf_path = os.path.join(models_dir, "randomforest_model.pkl")
        self.assertTrue(os.path.exists(rf_path), f"RandomForest model file not found at: {rf_path}")
        
        rf_model = joblib.load(rf_path)
        self.assertIsNotNone(rf_model)
        self.assertTrue(hasattr(rf_model, "predict"))

if __name__ == "__main__":
    unittest.main()
