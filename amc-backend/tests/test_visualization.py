import unittest
import numpy as np
import matplotlib.pyplot as plt
from amc.visualization import (
    plot_constellation,
    plot_fft,
    plot_psd,
    plot_waterfall,
    plot_spectrogram,
    plot_iq,
    plot_eye_diagram
)

class TestVisualization(unittest.TestCase):

    def setUp(self):
        # 128 complex IQ sample vector
        self.signal = np.exp(1j * np.linspace(0, 10 * np.pi, 128))
        self.sample_rate = 100000.0

    def test_plots(self):
        # Test constellation
        ax = plot_constellation(self.signal)
        self.assertIsNotNone(ax)
        plt.close(ax.get_figure())

        # Test FFT
        ax = plot_fft(self.signal, self.sample_rate)
        self.assertIsNotNone(ax)
        plt.close(ax.get_figure())

        # Test PSD
        ax = plot_psd(self.signal, self.sample_rate)
        self.assertIsNotNone(ax)
        plt.close(ax.get_figure())

        # Test Spectrogram
        ax = plot_spectrogram(self.signal, self.sample_rate, nperseg=32, noverlap=16)
        self.assertIsNotNone(ax)
        plt.close(ax.get_figure())

        # Test IQ Waveform
        ax = plot_iq(self.signal, self.sample_rate)
        self.assertIsNotNone(ax)
        plt.close(ax.get_figure())

        # Test Eye Diagram
        fig = plot_eye_diagram(self.signal, samples_per_symbol=8)
        self.assertIsNotNone(fig)
        plt.close(fig)

        # Test 3D Waterfall
        ax = plot_waterfall(self.signal, self.sample_rate, nperseg=32, noverlap=16)
        self.assertIsNotNone(ax)
        plt.close(ax.get_figure())

if __name__ == "__main__":
    unittest.main()
