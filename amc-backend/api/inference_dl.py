import os
import sys
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import spectrogram

from amc.core.models_dl import CNN1D, CNNLSTM, CNN2D

MODELS_DL_DIR = os.path.join("results", "models", "dl")

class DLInferenceService:
    # Class-level caches for loaded models and class labels
    _models_cache = {}
    _classes_cache = {}

    @classmethod
    def load_all_models(cls):
        """No-op to save startup memory. Models are loaded lazily on demand."""
        print("Lazy DL model loading enabled. Skipping startup load.")
        pass

    @classmethod
    def load_model(cls, m_name: str):
        """Loads a single DL model on demand, clearing previous models from cache."""
        m_name = m_name.lower().strip()
        if m_name in cls._models_cache:
            return

        # Clear existing cached models to save RAM
        cls._models_cache.clear()
        cls._classes_cache.clear()
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.path.join(MODELS_DL_DIR, f"best_{m_name}_raw.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"DL model checkpoint for '{m_name}' not found at {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            classes = checkpoint.get("classes", [])
            num_classes = len(classes)

            if num_classes == 0:
                raise ValueError(f"Error loading {m_name}: no classes metadata found in checkpoint.")

            # Instantiate model
            if m_name == "cnn1d":
                model = CNN1D(in_channels=2, sequence_length=1024, num_classes=num_classes)
            elif m_name == "cnnlstm":
                model = CNNLSTM(in_channels=2, sequence_length=1024, num_classes=num_classes)
            elif m_name == "cnn2d":
                model = CNN2D(in_channels=2, height=64, width=31, num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported model: {m_name}")

            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()  # Put model into evaluation mode

            cls._models_cache[m_name] = model
            cls._classes_cache[m_name] = classes
            print(f"Successfully cached DL model '{m_name}' in memory (eval mode).")

        except Exception as e:
            print(f"Failed to load DL model {m_name}: {e}")
            raise e

    @classmethod
    def get_dl_models_status(cls) -> dict:
        """Returns the availability status of DL models on disk."""
        supported = ["cnn1d", "cnnlstm", "cnn2d"]
        status = {}
        for m in supported:
            checkpoint_path = os.path.join(MODELS_DL_DIR, f"best_{m}_raw.pth")
            status[m] = os.path.exists(checkpoint_path)
        return status

    @staticmethod
    def compute_spectrogram(signal: np.ndarray, nperseg: int = 64, noverlap: int = 32) -> np.ndarray:
        """Computes spectrogram representation matching train_dl.py."""
        _, _, Sxx_real = spectrogram(
            signal.real,
            fs=1.0,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False
        )
        _, _, Sxx_imag = spectrogram(
            signal.imag,
            fs=1.0,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False
        )
        
        # Center frequencies
        Sxx_real = np.fft.fftshift(Sxx_real, axes=0)
        Sxx_imag = np.fft.fftshift(Sxx_imag, axes=0)
        # Convert to dB
        Sxx_real_db = 10 * np.log10(Sxx_real + 1e-12).astype(np.float32)
        Sxx_imag_db = 10 * np.log10(Sxx_imag + 1e-12).astype(np.float32)
        
        # Pad or truncate width dimension to exactly 31 columns
        def pad_or_truncate(S):
            if S.shape[1] < 31:
                pad_width = 31 - S.shape[1]
                S = np.pad(S, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
            elif S.shape[1] > 31:
                S = S[:, :31]
            return S
            
        Sxx_real_db = pad_or_truncate(Sxx_real_db)
        Sxx_imag_db = pad_or_truncate(Sxx_imag_db)
        
        # Shape: (2, 64, 31)
        return np.stack([Sxx_real_db, Sxx_imag_db], axis=0)

    @classmethod
    def predict(cls, model_name: str, signal: np.ndarray) -> dict:
        """Runs inference using cached models in evaluation mode."""
        m_name = model_name.lower().strip()
        
        # Load/cache the single active model dynamically
        cls.load_model(m_name)

        model = cls._models_cache[m_name]
        classes = cls._classes_cache[m_name]
        device = next(model.parameters()).device

        # Crop or pad signal to exactly 1024 samples to ensure consistent sequence length
        if len(signal) > 1024:
            signal_cropped = signal[:1024]
        elif len(signal) < 1024:
            signal_cropped = np.pad(signal, (0, 1024 - len(signal)), mode='constant')
        else:
            signal_cropped = signal

        # Prepare input tensor based on architecture type
        if m_name in ["cnn1d", "cnnlstm"]:
            # Shape stacked to: (2, 1024)
            iq_data = np.stack([signal_cropped.real, signal_cropped.imag], axis=0).astype(np.float32)
            input_tensor = torch.tensor(iq_data).unsqueeze(0).to(device)  # (1, 2, 1024)
            
            # Validate input dimensions
            if input_tensor.shape != (1, 2, 1024):
                raise ValueError(f"Invalid input shape {input_tensor.shape} for {m_name}. Expected (1, 2, 1024).")

        elif m_name == "cnn2d":
            # Spectrogram shape: (2, 64, 31)
            spec_data = cls.compute_spectrogram(signal_cropped)
            input_tensor = torch.tensor(spec_data).unsqueeze(0).to(device)  # (1, 2, 64, 31)
            
            # Validate input dimensions
            if input_tensor.shape != (1, 2, 64, 31):
                raise ValueError(f"Invalid spectrogram shape {input_tensor.shape} for {m_name}. Expected (1, 2, 64, 31).")
        else:
            raise ValueError(f"Unsupported DL model: {model_name}")

        # Run model inference in eval mode and torch.no_grad()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probabilities)
            predicted_class = classes[pred_idx]
            confidence = float(probabilities[pred_idx])

        prob_dict = {str(classes[i]): float(probabilities[i]) for i in range(len(classes))}

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict
        }
