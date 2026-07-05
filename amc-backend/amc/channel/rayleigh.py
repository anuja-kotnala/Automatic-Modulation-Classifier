import numpy as np
from amc.channel.base import BaseImpairment

class RayleighFadingImpairment(BaseImpairment):
    """
    Flat Rayleigh fading channel using a Jakes-like Doppler spectrum.
    """

    def __init__(self, doppler_freq: float = 10.0, seed: int = None):
        self.doppler_freq = doppler_freq
        self.seed = seed

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        n_samples = len(signal)
        if n_samples == 0:
            return signal

        if self.seed is not None:
            np.random.seed(self.seed)

        # If Doppler frequency is 0, fading is static
        if self.doppler_freq <= 0:
            h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2.0)
            return signal * h

        # Generate complex white Gaussian noise in time domain
        w = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2.0)
        
        # Transform to frequency domain
        W = np.fft.fft(w)
        freqs = np.fft.fftfreq(n_samples, d=1.0/sample_rate)

        # Design Jakes spectrum filter: H(f) = 1 / (1 - (f/f_d)^2)^0.25 for |f| < f_d
        H = np.zeros(n_samples)
        f_ratio = freqs / self.doppler_freq
        valid_indices = np.abs(f_ratio) < 1.0
        
        # Handle division by zero at boundary
        H[valid_indices] = 1.0 / np.sqrt(np.sqrt(1.0 - f_ratio[valid_indices]**2 + 1e-6))
        
        # Apply filter and reconstruct time domain coefficients
        H_filtered = W * H
        h = np.fft.ifft(H_filtered)
        
        # Normalize channel energy to unit average power
        h_power = np.mean(np.abs(h)**2)
        if h_power > 0:
            h /= np.sqrt(h_power)

        return signal * h
