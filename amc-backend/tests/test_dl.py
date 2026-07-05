import unittest
import torch
from train_dl import CNN1D, CNNLSTM, CNN2D

class TestDeepLearningModels(unittest.TestCase):

    def test_model_forward_passes(self):
        batch_size = 4
        num_classes = 7

        # 1. CNN1D - Raw IQ representation input: shape (batch_size, 2, 1024)
        x_raw = torch.randn(batch_size, 2, 1024)
        cnn1d = CNN1D(in_channels=2, sequence_length=1024, num_classes=num_classes)
        out = cnn1d(x_raw)
        self.assertEqual(out.shape, (batch_size, num_classes))

        # 2. CNNLSTM - shape (batch_size, 2, 1024)
        cnnlstm = CNNLSTM(in_channels=2, sequence_length=1024, num_classes=num_classes)
        out = cnnlstm(x_raw)
        self.assertEqual(out.shape, (batch_size, num_classes))

        # 3. CNN2D - Spectrogram representation input: shape (batch_size, 2, 64, 31)
        x_spec = torch.randn(batch_size, 2, 64, 31)
        cnn2d = CNN2D(in_channels=2, height=64, width=31, num_classes=num_classes)
        out = cnn2d(x_spec)
        self.assertEqual(out.shape, (batch_size, num_classes))

    def test_get_model_helper(self):
        from train_dl import get_model
        model_names = ["cnn1d", "cnnlstm", "cnn2d"]
        for m_name in model_names:
            model = get_model(m_name, in_channels=2, seq_len=1024, num_classes=5)
            self.assertIsInstance(model, torch.nn.Module)

if __name__ == "__main__":
    unittest.main()
