import os
import unittest
import numpy as np
from fastapi.testclient import TestClient

from app import app

class TestRESTApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("version", data)
        self.assertIn("models_loaded", data)
        self.assertTrue(isinstance(data["models_loaded"], dict))

    def test_plots_endpoint(self):
        response = self.client.get("/plots")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(isinstance(data, list))

    def test_results_endpoint(self):
        response = self.client.get("/results")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("summary", data)
        self.assertIn("available_files", data)

    def test_generate_signal_endpoint(self):
        payload = {
            "modulation_type": "BPSK",
            "snr_db": 15.0,
            "sample_rate": 1000000.0,
            "num_samples": 128
        }
        response = self.client.post("/generate-signal", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["modulation_type"], "BPSK")
        self.assertEqual(len(data["iq_data"]), 2)
        self.assertEqual(len(data["iq_data"][0]), 128)
        self.assertIsNotNone(data["file_path"])

    def test_predict_endpoint(self):
        # First generate a signal to test prediction
        gen_payload = {
            "modulation_type": "QPSK",
            "snr_db": 20.0,
            "sample_rate": 1000000.0,
            "num_samples": 1024
        }
        gen_response = self.client.post("/generate-signal", json=gen_payload)
        self.assertEqual(gen_response.status_code, 200)
        gen_data = gen_response.json()
        
        # Test predict with iq_data
        predict_payload = {
            "iq_data": gen_data["iq_data"],
            "model_name": "randomforest"
        }
        pred_response = self.client.post("/predict", json=predict_payload)
        self.assertEqual(pred_response.status_code, 200)
        pred_data = pred_response.json()
        self.assertTrue(pred_data["success"])
        self.assertIn("predicted_class", pred_data)
        self.assertIn("confidence", pred_data)
        self.assertTrue(len(pred_data["features"]) >= 34)

    def test_generate_plots_endpoint(self):
        # Generate raw signal
        gen_payload = {
            "modulation_type": "16QAM",
            "snr_db": 20.0,
            "sample_rate": 1000000.0,
            "num_samples": 256
        }
        gen_response = self.client.post("/generate-signal", json=gen_payload)
        self.assertEqual(gen_response.status_code, 200)
        gen_data = gen_response.json()

        # Generate plots
        plot_payload = {
            "iq_data": gen_data["iq_data"],
            "modulation_type": "16QAM",
            "sample_rate": 1000000.0
        }
        plot_response = self.client.post("/generate-plots", json=plot_payload)
        self.assertEqual(plot_response.status_code, 200)
        plot_data = plot_response.json()
        self.assertTrue(plot_data["success"])
        self.assertTrue(len(plot_data["plots"]) >= 6)

if __name__ == "__main__":
    unittest.main()
