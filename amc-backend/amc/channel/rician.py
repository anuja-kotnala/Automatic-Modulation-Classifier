import numpy as np
from amc.channel.base import BaseImpairment
from amc.channel.rayleigh import RayleighFadingImpairment

class RicianFadingImpairment(BaseImpairment):
    """
    Flat Rician fading channel featuring a dominant Line-Of-Sight (LOS) path
    and a diffuse multipath component.
    """

    def __init__(self, k_factor: float = 4.0, doppler_freq: float = 5.0, seed: int = None):
        self.k_factor = k_factor  # K-factor in linear scale
        self.doppler_freq = doppler_freq
        self.seed = seed

    def apply(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        n_samples = len(signal)
        if n_samples == 0:
            return signal

        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate diffuse Rayleigh component
        rayleigh_channel = RayleighFadingImpairment(doppler_freq=self.doppler_freq, seed=self.seed)
        # Apply rayleigh to ones vector to get Rayleigh coefficients h_diffuse
        h_diffuse = rayleigh_channel.apply(np.ones(n_samples, dtype=np.complex64), sample_rate)

        # Generate dominant LOS component with a small Doppler shift
        t = np.arange(n_samples) / sample_rate
        # We assume LOS path has a Doppler shift of doppler_freq * cos(theta), say 0.7 * doppler_freq
        los_doppler = 0.7 * self.doppler_freq
        h_los = np.exp(1j * 2 * np.pi * los_doppler * t)

        # Combine components based on Rician K-factor: h = sqrt(K/(K+1))*h_los + sqrt(1/(K+1))*h_diffuse
        const_los = np.sqrt(self.k_factor / (self.k_factor + 1.0))
        const_diffuse = np.sqrt(1.0 / (self.k_factor + 1.0))
        
        h = const_los * h_los + const_diffuse * h_diffuse

        # Normalize average channel power to 1
        h_power = np.mean(np.abs(h)**2)
        if h_power > 0:
            h /= np.sqrt(h_power)

        return signal * h
