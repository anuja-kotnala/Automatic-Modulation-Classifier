import os
import unittest
import numpy as np
from fastapi.testclient import TestClient

from app import app

class TestDLRESTApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_dl_endpoint(self):
        response = self.client.get("/health-dl")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("models_loaded", data)
        self.assertTrue(isinstance(data["models_loaded"], dict))
        for model in ["cnn1d", "cnnlstm", "cnn2d"]:
            self.assertIn(model, data["models_loaded"])

    def test_predict_dl_endpoint_cnn1d(self):
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
        
        # Test predict with cnn1d model
        predict_payload = {
            "iq_data": gen_data["iq_data"],
            "model_name": "cnn1d"
        }
        pred_response = self.client.post("/predict-dl", json=predict_payload)
        self.assertEqual(pred_response.status_code, 200)
        pred_data = pred_response.json()
        self.assertTrue(pred_data["success"])
        self.assertIn("predicted_class", pred_data)
        self.assertIn("confidence", pred_data)
        self.assertIsNotNone(pred_data["probabilities"])
        self.assertTrue(len(pred_data["features"]) >= 34)

    def test_predict_dl_endpoint_cnnlstm(self):
        gen_payload = {
            "modulation_type": "BPSK",
            "snr_db": 20.0,
            "sample_rate": 1000000.0,
            "num_samples": 1024
        }
        gen_response = self.client.post("/generate-signal", json=gen_payload)
        self.assertEqual(gen_response.status_code, 200)
        gen_data = gen_response.json()
        
        predict_payload = {
            "iq_data": gen_data["iq_data"],
            "model_name": "cnnlstm"
        }
        pred_response = self.client.post("/predict-dl", json=predict_payload)
        self.assertEqual(pred_response.status_code, 200)
        pred_data = pred_response.json()
        self.assertTrue(pred_data["success"])
        self.assertIn("predicted_class", pred_data)

    def test_predict_dl_endpoint_cnn2d(self):
        gen_payload = {
            "modulation_type": "16QAM",
            "snr_db": 20.0,
            "sample_rate": 1000000.0,
            "num_samples": 1024
        }
        gen_response = self.client.post("/generate-signal", json=gen_payload)
        self.assertEqual(gen_response.status_code, 200)
        gen_data = gen_response.json()
        
        predict_payload = {
            "iq_data": gen_data["iq_data"],
            "model_name": "cnn2d"
        }
        pred_response = self.client.post("/predict-dl", json=predict_payload)
        self.assertEqual(pred_response.status_code, 200)
        pred_data = pred_response.json()
        self.assertTrue(pred_data["success"])
        self.assertIn("predicted_class", pred_data)

if __name__ == "__main__":
    unittest.main()
