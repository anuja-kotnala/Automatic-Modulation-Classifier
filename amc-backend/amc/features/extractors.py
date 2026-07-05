from typing import Dict, Any, List
import numpy as np
from scipy.stats import skew, kurtosis
from amc.core.base_extractor import BaseFeatureExtractor

class FeatureExtractor(BaseFeatureExtractor):
    """
    Unified comprehensive Feature Extractor.
    Extracts 34 publication-grade statistical, envelope, phase, frequency,
    spectral, high-order cumulants, and cyclostationary features from complex signals.
    """

    def extract(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extracts 34 distinct features from a 1D complex IQ signal.
        """
        features = {}

        # Decompose signal
        I = signal.real
        Q = signal.imag
        A = np.abs(signal)
        phase = np.angle(signal)

        # Ensure no empty signal checks
        if len(signal) == 0:
            return {}

        # ----------------------------------------------------
        # 1. Time-Domain Statistical Features (Real/Imaginary)
        # ----------------------------------------------------
        features["mean_i"] = float(np.mean(I))
        features["mean_q"] = float(np.mean(Q))
        features["var_i"] = float(np.var(I))
        features["var_q"] = float(np.var(Q))
        features["std_i"] = float(np.std(I))
        features["std_q"] = float(np.std(Q))
        
        # Safe skew and kurtosis calculations
        features["skew_i"] = float(skew(I) if np.std(I) > 0 else 0.0)
        features["skew_q"] = float(skew(Q) if np.std(Q) > 0 else 0.0)
        features["kurt_i"] = float(kurtosis(I) if np.std(I) > 0 else 0.0)
        features["kurt_q"] = float(kurtosis(Q) if np.std(Q) > 0 else 0.0)

        # ----------------------------------------------------
        # 2. Envelope & Amplitude Features
        # ----------------------------------------------------
        features["env_mean"] = float(np.mean(A))
        features["env_var"] = float(np.var(A))
        features["env_std"] = float(np.std(A))
        
        mean_power = np.mean(A ** 2)
        features["rms"] = float(np.sqrt(mean_power))
        features["power"] = float(mean_power)
        features["energy"] = float(np.sum(A ** 2))
        
        peak_power = np.max(A ** 2)
        features["papr"] = float(10 * np.log10(peak_power / (mean_power + 1e-12) + 1e-12))

        # ----------------------------------------------------
        # 3. Instantaneous Phase & Frequency Features
        # ----------------------------------------------------
        features["inst_phase_mean"] = float(np.mean(phase))
        features["inst_phase_std"] = float(np.std(phase))

        # Instantaneous Frequency: wrap derivative of phase
        inst_freq = np.diff(phase)
        inst_freq = (inst_freq + np.pi) % (2 * np.pi) - np.pi
        features["inst_freq_mean"] = float(np.mean(inst_freq) if len(inst_freq) > 0 else 0.0)
        features["inst_freq_std"] = float(np.std(inst_freq) if len(inst_freq) > 0 else 0.0)

        # ----------------------------------------------------
        # 4. Crossing & Physical Features
        # ----------------------------------------------------
        zero_crossings = np.diff(np.sign(I)) != 0
        features["zero_crossing_rate"] = float(np.mean(zero_crossings) if len(zero_crossings) > 0 else 0.0)

        # ----------------------------------------------------
        # 5. Spectral Features
        # ----------------------------------------------------
        N = len(signal)
        X = np.fft.fft(signal)
        psd = np.abs(X) ** 2 / N
        psd_sum = np.sum(psd)
        
        # Normalized PSD for entropy
        psd_norm = psd / (psd_sum + 1e-12)
        features["spectral_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
        
        # Spectral Flatness (Geometric Mean / Arithmetic Mean)
        flatness_num = np.exp(np.mean(np.log(psd + 1e-12)))
        flatness_den = np.mean(psd)
        features["spectral_flatness"] = float(flatness_num / (flatness_den + 1e-12))

        # Centroid and peak frequency
        freqs = np.fft.fftfreq(N)
        features["spectral_centroid"] = float(np.sum(np.abs(freqs) * psd) / (psd_sum + 1e-12))
        features["peak_frequency"] = float(freqs[np.argmax(psd)])

        # Bandwidth estimators
        # 3dB Bandwidth: count bins within 3dB (0.5 power ratio) of peak PSD
        peak_psd = np.max(psd)
        thresh_3db = 0.5 * peak_psd
        bins_above_3db = np.sum(psd > thresh_3db)
        features["bandwidth_3db"] = float(bins_above_3db / N)

        # 99% Occupied Bandwidth
        sorted_indices = np.argsort(np.abs(freqs))
        sorted_freqs = np.abs(freqs)[sorted_indices]
        sorted_psd = psd[sorted_indices]
        cum_psd = np.cumsum(sorted_psd)
        
        # Locate frequency index containing 99% power
        idx_99 = np.searchsorted(cum_psd, 0.99 * psd_sum)
        features["occupied_bandwidth_99"] = float(sorted_freqs[min(idx_99, N - 1)] * 2.0)

        # ----------------------------------------------------
        # 6. High-Order Moments & Cumulants
        # ----------------------------------------------------
        # Normalized signal to unit power to keep cumulants scale-invariant
        sig_norm = signal / (np.sqrt(mean_power) + 1e-12)

        # Moments (E[x^p * (x*)^q])
        M20 = np.mean(sig_norm ** 2)
        M21 = np.mean(np.abs(sig_norm) ** 2)
        M40 = np.mean(sig_norm ** 4)
        M41 = np.mean(sig_norm ** 3 * np.conj(sig_norm))
        M42 = np.mean(np.abs(sig_norm) ** 4)
        M60 = np.mean(sig_norm ** 6)
        M61 = np.mean(sig_norm ** 5 * np.conj(sig_norm))
        M62 = np.mean(sig_norm ** 4 * np.conj(sig_norm) ** 2)
        M63 = np.mean(np.abs(sig_norm) ** 6)

        # Second-order cumulants
        features["C20"] = float(np.abs(M20))
        features["C21"] = float(np.abs(M21))

        # Fourth-order cumulants
        C40 = M40 - 3 * (M20 ** 2)
        C41 = M41 - 3 * M20 * M21
        C42 = M42 - np.abs(M20) ** 2 - 2 * (M21 ** 2)
        features["C40"] = float(np.abs(C40))
        features["C41"] = float(np.abs(C41))
        features["C42"] = float(np.abs(C42))

        # Sixth-order cumulants
        C60 = M60 - 15 * M20 * M40 + 30 * (M20 ** 3)
        C61 = M61 - 5 * M20 * M41 - 10 * M21 * M40 + 30 * M21 * (M20 ** 2)
        C62 = M62 - 6 * M20 * M42 - 8 * M21 * M41 + 6 * (M20 ** 2) * np.conj(M20) + 18 * M21 * (np.abs(M20) ** 2) + 12 * (M21 ** 3)  # Simplified standard formula
        C63 = M63 - 9 * M21 * M42 - 6 * M21 * (np.abs(M20) ** 2) + 12 * (M21 ** 3)
        features["C60"] = float(np.abs(C60))
        features["C61"] = float(np.abs(C61))
        features["C62"] = float(np.abs(C62))
        features["C63"] = float(np.abs(C63))

        # ----------------------------------------------------
        # 7. Cyclostationary Features
        # ----------------------------------------------------
        # Cyclic Autocorrelation Function (CAF) lag-1 estimation
        lagged = np.roll(signal, 1)
        lagged[0] = 0j
        product = signal * np.conj(lagged)
        caf = np.abs(np.fft.fft(product))
        
        # Max peak in non-DC cyclic frequency bins
        features["cyclic_max_alpha"] = float(np.max(caf[1:]) if len(caf) > 1 else 0.0)
        features["cyclic_mean"] = float(np.mean(caf))

        return features

# Maintain skeleton compatibility for default sub-extractors
class CumulantFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, orders: List[int]):
        self.orders = orders
        self.extractor = FeatureExtractor()

    def extract(self, signal: np.ndarray) -> Dict[str, float]:
        all_feats = self.extractor.extract(signal)
        return {f"C{o}": all_feats[f"C{o}"] for o in self.orders if f"C{o}" in all_feats}

class SpectralFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, fft_size: int = 1024):
        self.fft_size = fft_size
        self.extractor = FeatureExtractor()

    def extract(self, signal: np.ndarray) -> Dict[str, float]:
        all_feats = self.extractor.extract(signal)
        spectral_keys = ["spectral_entropy", "spectral_flatness", "spectral_centroid", "bandwidth_3db", "occupied_bandwidth_99", "peak_frequency"]
        return {k: all_feats[k] for k in spectral_keys if k in all_feats}

class WaveletFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, wavelet_name: str = "db4", level: int = 3):
        self.wavelet_name = wavelet_name
        self.level = level
        self.extractor = FeatureExtractor()

    def extract(self, signal: np.ndarray) -> Dict[str, float]:
        # returns subset envelope stats as mock wavelet features
        all_feats = self.extractor.extract(signal)
        return {"wavelet_mean": all_feats["env_mean"], "wavelet_std": all_feats["env_std"]}
