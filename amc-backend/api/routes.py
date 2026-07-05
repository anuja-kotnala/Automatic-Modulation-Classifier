import os
import time
import glob
import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional

from amc.constants import ModulationType, MODULATION_MAP
from amc.generator import *
from amc.channel.channel_pipeline import ChannelPipeline
from amc.config import AppConfig
from amc.features.extractors import FeatureExtractor

from api.schemas import (
    HealthResponse,
    PlotFile,
    EvaluationResults,
    PredictRequest,
    PredictResponse,
    GenerateSignalRequest,
    GenerateSignalResponse,
    GeneratePlotsRequest,
    GeneratePlotsResponse,
    GeneratedPlotItem,
    PredictDLRequest,
    HealthDLResponse
)
from api.inference_dl import DLInferenceService

router = APIRouter()

# Helper to resolve string to ModulationType
def resolve_modulation_type(mod_str: str) -> ModulationType:
    val = mod_str.upper().strip()
    if val in MODULATION_MAP:
        return MODULATION_MAP[val]
    try:
        return ModulationType[val]
    except KeyError:
        pass
    if val in ["16QAM", "QAM16"]:
        return ModulationType.QAM16
    if val in ["64QAM", "QAM64"]:
        return ModulationType.QAM64
    raise HTTPException(status_code=400, detail=f"Unsupported modulation type: {mod_str}")

# Helper factory to get modulator (from main_generate_dataset)
def get_modulator_instance(mod_type: ModulationType, sample_rate: float, seed: int):
    if mod_type == ModulationType.AM:
        return AMModulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.FM:
        return FMModulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.BPSK:
        return BPSKModulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.QPSK:
        return QPSKModulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.QAM16:
        return QAM16Modulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.QAM64:
        return QAM64Modulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.OFDM:
        return OFDMModulator(sample_rate=sample_rate, random_seed=seed)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported modulation type: {mod_type}")

# Check loaded models status
def get_models_status() -> Dict[str, bool]:
    models_dir = os.path.join("results", "models")
    supported = ["randomforest", "svm", "gradientboosting", "xgboost", "lightgbm", "knn", "logisticregression"]
    status = {}
    for model in supported:
        path = os.path.join(models_dir, f"{model}_model.pkl")
        status[model] = os.path.exists(path)
    return status

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Returns the health status of the REST API, version, and the loaded status of classification models.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_loaded=get_models_status()
    )

@router.get("/plots", response_model=List[PlotFile])
def list_plots():
    """
    Lists the generated plot files available in the system.
    """
    plots = []
    # Plot directories
    plots_dir = os.path.join("results", "plots")
    eval_dir = os.path.join("results", "ml_eval")
    
    if os.path.exists(plots_dir):
        for f in os.listdir(plots_dir):
            if f.endswith((".png", ".jpg", ".jpeg")):
                plots.append(PlotFile(
                    name=f,
                    category="signal",
                    url=f"/static/plots/{f}"
                ))
                
    if os.path.exists(eval_dir):
        for f in os.listdir(eval_dir):
            if f.endswith((".png", ".jpg", ".jpeg")):
                plots.append(PlotFile(
                    name=f,
                    category="evaluation",
                    url=f"/static/ml_eval/{f}"
                ))
    return plots

@router.get("/results", response_model=EvaluationResults)
def get_results():
    """
    Returns unified evaluation results summary and metrics from both ML and DL models.
    """
    import csv
    summary_data = []

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def safe_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def is_not_na(val):
        return val is not None and val != "" and val.lower() != "nan"

    # 1. Read ML summary
    ml_summary_path = os.path.join("results", "ml_eval", "model_performance_summary.csv")
    if os.path.exists(ml_summary_path):
        try:
            with open(ml_summary_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    accuracy = safe_float(row.get("Accuracy")) or 0.0
                    precision = safe_float(row.get("Precision")) or 0.0
                    recall = safe_float(row.get("Recall")) or 0.0
                    f1_score = safe_float(row.get("F1_Score")) or 0.0
                    
                    # Keep extra fields for extensibility
                    extra = {}
                    for col, val in row.items():
                        if col not in ["Classifier", "Accuracy", "Precision", "Recall", "F1_Score", "CV_Mean", "CV_Std", "Macro_AUC"]:
                            try:
                                extra[col] = int(val)
                            except ValueError:
                                try:
                                    extra[col] = float(val)
                                except ValueError:
                                    extra[col] = val

                    cv_mean_val = row.get("CV_Mean")
                    cv_std_val = row.get("CV_Std")
                    macro_auc_val = row.get("Macro_AUC")

                    summary_data.append({
                        "model_name": str(row.get("Classifier", "Unknown")),
                        "model_type": "ml",
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "cv_mean": safe_float(cv_mean_val) if is_not_na(cv_mean_val) else None,
                        "cv_std": safe_float(cv_std_val) if is_not_na(cv_std_val) else None,
                        "macro_auc": safe_float(macro_auc_val) if is_not_na(macro_auc_val) else None,
                        # Fallbacks/aliases for frontend backwards compatibility
                        "model": str(row.get("Classifier", "Unknown")),
                        "Classifier": str(row.get("Classifier", "Unknown")),
                        **extra
                    })
        except Exception as e:
            print(f"Warning: Failed to read ML performance summary: {e}")

    # 2. Read DL summary
    dl_summary_path = os.path.join("results", "dl_eval", "dl_model_performance_summary.csv")
    if os.path.exists(dl_summary_path):
        try:
            with open(dl_summary_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    accuracy = safe_float(row.get("Test Accuracy")) or 0.0
                    precision = safe_float(row.get("Precision")) or 0.0
                    recall = safe_float(row.get("Recall")) or 0.0
                    f1_score = safe_float(row.get("F1 Score")) or 0.0

                    # Parse SNR accuracies into a dictionary
                    snr_accuracy = {}
                    for snr in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
                        col_name = f"Acc_SNR_{snr}dB"
                        if col_name in row:
                            val = row[col_name]
                            if is_not_na(val):
                                snr_accuracy[str(snr)] = safe_float(val)

                    # Keep extra fields for extensibility
                    extra = {}
                    exclude_cols = [
                        "Model", "Test Accuracy", "Precision", "Recall", "F1 Score", 
                        "Parameters", "Training Time (s)", "Avg Epoch Time (s)", 
                        "Model Size (MB)", "Best Val Acc", "Best Val Loss", 
                        "Avg Inference Time/Sample (ms)"
                    ]
                    for col, val in row.items():
                        if col not in exclude_cols and not col.startswith("Acc_SNR_"):
                            try:
                                extra[col] = int(val)
                            except ValueError:
                                try:
                                    extra[col] = float(val)
                                except ValueError:
                                    extra[col] = val

                    params_val = row.get("Parameters")
                    train_time_val = row.get("Training Time (s)")
                    epoch_time_val = row.get("Avg Epoch Time (s)")
                    size_val = row.get("Model Size (MB)")
                    val_acc_val = row.get("Best Val Acc")
                    val_loss_val = row.get("Best Val Loss")
                    inf_time_val = row.get("Avg Inference Time/Sample (ms)")

                    summary_data.append({
                        "model_name": str(row.get("Model", "Unknown")),
                        "model_type": "dl",
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "parameters": safe_int(params_val) if is_not_na(params_val) else None,
                        "training_time": safe_float(train_time_val) if is_not_na(train_time_val) else None,
                        "avg_epoch_time": safe_float(epoch_time_val) if is_not_na(epoch_time_val) else None,
                        "model_size_mb": safe_float(size_val) if is_not_na(size_val) else None,
                        "best_val_accuracy": safe_float(val_acc_val) if is_not_na(val_acc_val) else None,
                        "best_val_loss": safe_float(val_loss_val) if is_not_na(val_loss_val) else None,
                        "inference_time_ms": safe_float(inf_time_val) if is_not_na(inf_time_val) else None,
                        "snr_accuracy": snr_accuracy if snr_accuracy else None,
                        # Fallbacks/aliases for frontend backwards compatibility
                        "model": str(row.get("Model", "Unknown")),
                        "Classifier": str(row.get("Model", "Unknown")),
                        **extra
                    })
        except Exception as e:
            print(f"Warning: Failed to read DL performance summary: {e}")

    # 3. Sort by accuracy descending
    summary_data = sorted(summary_data, key=lambda x: x.get("accuracy", 0.0), reverse=True)

    # 4. Get available plots files (both ML and DL plots)
    files = []
    ml_eval_dir = os.path.join("results", "ml_eval")
    if os.path.exists(ml_eval_dir):
        files.extend([f for f in os.listdir(ml_eval_dir) if f.endswith((".png", ".csv"))])

    dl_eval_dir = os.path.join("results", "dl_eval")
    if os.path.exists(dl_eval_dir):
        files.extend([f for f in os.listdir(dl_eval_dir) if f.endswith((".png", ".csv"))])

    return EvaluationResults(
        summary=summary_data,
        available_files=files
    )

@router.post("/predict", response_model=PredictResponse)
def predict_signal(request: PredictRequest):
    """
    Classifies a signal either by reading a stored npy file or accepting an raw IQ array.
    """
    # 1. Resolve signal data
    signal = None
    if request.file_path:
        # Load from dataset raw file
        full_path = os.path.join("dataset", request.file_path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"Signal file not found at {request.file_path}")
        try:
            signal = np.load(full_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load numpy file: {str(e)}")
    elif request.iq_data:
        if len(request.iq_data) != 2:
            raise HTTPException(status_code=400, detail="iq_data must be a 2D list of shape [2, N] (real and imag parts)")
        try:
            real = np.array(request.iq_data[0])
            imag = np.array(request.iq_data[1])
            signal = real + 1j * imag
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid signal arrays: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either file_path or iq_data")
        
    if len(signal) == 0:
        raise HTTPException(status_code=400, detail="Signal length cannot be 0")

    # 2. Extract features
    try:
        extractor = FeatureExtractor()
        features = extractor.extract(signal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # 3. Load classification models
    models_dir = os.path.join("results", "models")
    model_name = request.model_name.lower().strip()
    model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    le_path = os.path.join(models_dir, "label_encoder.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' file not found. Check health endpoint for loaded status.")
    if not os.path.exists(scaler_path) or not os.path.exists(le_path):
        raise HTTPException(status_code=500, detail="Scaler or Label Encoder binary is missing from models folder.")

    try:
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)

        # Reorder and format feature vector alphabetically matching training structure
        feature_names = sorted([k for k in features.keys()])
        feature_vector = np.array([features[k] for k in feature_names]).reshape(1, -1)
        
        scaled_vector = scaler.transform(feature_vector)
        
        # Predict class
        pred = clf.predict(scaled_vector)
        predicted_class = str(le.inverse_transform(pred)[0])
        
        # Predict probabilities
        probabilities = None
        confidence = 1.0
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(scaled_vector)[0]
            probabilities = {str(le.classes_[i]): float(proba[i]) for i in range(len(proba))}
            confidence = float(np.max(proba))
            
        return PredictResponse(
            success=True,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            features=features
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification prediction failed: {str(e)}")

@router.post("/predict-dl", response_model=PredictResponse)
def predict_signal_dl(request: PredictDLRequest):
    """
    Classifies a signal using Deep Learning PyTorch models.
    """
    # 1. Resolve signal data
    signal = None
    if request.file_path:
        full_path = os.path.join("dataset", request.file_path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"Signal file not found at {request.file_path}")
        try:
            signal = np.load(full_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load numpy file: {str(e)}")
    elif request.iq_data:
        if len(request.iq_data) != 2:
            raise HTTPException(status_code=400, detail="iq_data must be a 2D list of shape [2, N] (real and imag parts)")
        try:
            real = np.array(request.iq_data[0])
            imag = np.array(request.iq_data[1])
            signal = real + 1j * imag
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid signal arrays: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either file_path or iq_data")
        
    if len(signal) == 0:
        raise HTTPException(status_code=400, detail="Signal length cannot be 0")

    # 2. Extract features (to keep PredictResponse compatibility)
    try:
        extractor = FeatureExtractor()
        features = extractor.extract(signal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # 3. Predict using DL inference service
    try:
        res = DLInferenceService.predict(request.model_name, signal)
        return PredictResponse(
            success=True,
            predicted_class=res["predicted_class"],
            confidence=res["confidence"],
            probabilities=res["probabilities"],
            features=features
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DL prediction failed: {str(e)}")

@router.get("/health-dl", response_model=HealthDLResponse)
def health_dl_check():
    """
    Returns the loaded status of DL classification models.
    """
    return HealthDLResponse(
        status="ok",
        models_loaded=DLInferenceService.get_dl_models_status()
    )

@router.post("/generate-signal", response_model=GenerateSignalResponse)
def generate_signal(request: GenerateSignalRequest):
    """
    Generates a simulated IQ signal with impairments, saves it, and returns the raw values.
    """
    try:
        mod_type = resolve_modulation_type(request.modulation_type)
        seed = hash((request.modulation_type, request.snr_db, time.time())) % (2**32)
        np.random.seed(seed)
        
        modulator = get_modulator_instance(mod_type, request.sample_rate, seed)
        raw_signal = modulator.generate(request.num_samples, snr_db=None)
        
        # Build channel config and override with request values
        config = AppConfig.load_from_yaml("configs/default_config.yaml")
        ch_config = config.channel
        
        # Override AWGN
        ch_config.awgn.enabled = True
        ch_config.awgn.snr_db = float(request.snr_db)
        
        if request.impairments:
            imp = request.impairments
            if imp.awgn_enabled is not None:
                ch_config.awgn.enabled = imp.awgn_enabled
            if imp.awgn_snr_db is not None:
                ch_config.awgn.snr_db = imp.awgn_snr_db
                
            if imp.rayleigh_enabled is not None:
                ch_config.rayleigh.enabled = imp.rayleigh_enabled
            if imp.rayleigh_doppler_freq is not None:
                ch_config.rayleigh.doppler_freq = imp.rayleigh_doppler_freq
                
            if imp.rician_enabled is not None:
                ch_config.rician.enabled = imp.rician_enabled
            if imp.rician_k_factor is not None:
                ch_config.rician.k_factor = imp.rician_k_factor
            if imp.rician_doppler_freq is not None:
                ch_config.rician.doppler_freq = imp.rician_doppler_freq
                
            if imp.frequency_offset_enabled is not None:
                ch_config.frequency_offset.enabled = imp.frequency_offset_enabled
            if imp.frequency_offset_hz is not None:
                ch_config.frequency_offset.offset_hz = imp.frequency_offset_hz
                
            if imp.phase_noise_enabled is not None:
                ch_config.phase_noise.enabled = imp.phase_noise_enabled
            if imp.phase_noise_std_dev is not None:
                ch_config.phase_noise.std_dev = imp.phase_noise_std_dev
                
            if imp.iq_imbalance_enabled is not None:
                ch_config.iq_imbalance.enabled = imp.iq_imbalance_enabled
            if imp.iq_imbalance_amplitude_db is not None:
                ch_config.iq_imbalance.amplitude_imbalance_db = imp.iq_imbalance_amplitude_db
            if imp.iq_imbalance_phase_deg is not None:
                ch_config.iq_imbalance.phase_imbalance_deg = imp.iq_imbalance_phase_deg
                
            if imp.timing_offset_enabled is not None:
                ch_config.timing_offset.enabled = imp.timing_offset_enabled
            if imp.timing_offset_fractional_delay is not None:
                ch_config.timing_offset.fractional_delay = imp.timing_offset_fractional_delay
                
            if imp.multipath_enabled is not None:
                ch_config.multipath.enabled = imp.multipath_enabled
            if imp.multipath_delays is not None:
                ch_config.multipath.delays = imp.multipath_delays
            if imp.multipath_gains_db is not None:
                ch_config.multipath.gains_db = imp.multipath_gains_db
                
            if imp.clock_drift_enabled is not None:
                ch_config.clock_drift.enabled = imp.clock_drift_enabled
            if imp.clock_drift_ppm is not None:
                ch_config.clock_drift.ppm = imp.clock_drift_ppm
                
        pipeline = ChannelPipeline(ch_config, seed=seed)
        impaired_signal = pipeline.apply(raw_signal, request.sample_rate)
        
        # Save dynamically to a temp or raw path
        mod_name = mod_type.value
        out_dir = os.path.join("dataset", "raw", mod_name, f"snr_{int(request.snr_db)}")
        os.makedirs(out_dir, exist_ok=True)
        
        filename = f"gen_{int(time.time())}.npy"
        file_path = os.path.join(out_dir, filename)
        np.save(file_path, impaired_signal)
        
        rel_path = os.path.relpath(file_path, "dataset")
        
        iq_data = [impaired_signal.real.tolist(), impaired_signal.imag.tolist()]
        
        return GenerateSignalResponse(
            success=True,
            modulation_type=mod_name,
            snr_db=request.snr_db,
            iq_data=iq_data,
            file_path=rel_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")

@router.post("/generate-plots", response_model=GeneratePlotsResponse)
def generate_signal_plots(request: GeneratePlotsRequest):
    """
    Renders signal charts and figures and saves them under results/plots.
    """
    import matplotlib
    matplotlib.use('Agg')
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
    # 1. Resolve signal data
    signal = None
    if request.file_path:
        full_path = os.path.join("dataset", request.file_path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"Signal file not found at {request.file_path}")
        try:
            signal = np.load(full_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load numpy file: {str(e)}")
    elif request.iq_data:
        if len(request.iq_data) != 2:
            raise HTTPException(status_code=400, detail="iq_data must be a 2D list of shape [2, N]")
        try:
            real = np.array(request.iq_data[0])
            imag = np.array(request.iq_data[1])
            signal = real + 1j * imag
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid signal arrays: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either file_path or iq_data")
        
    try:
        mod_type = resolve_modulation_type(request.modulation_type)
        mod_name = mod_type.value
        
        output_dir = os.path.join("results", "plots")
        os.makedirs(output_dir, exist_ok=True)
        
        generated = []
        timestamp = int(time.time())
        sample_rate = request.sample_rate
        
        # 1. Constellation
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_constellation(signal, title=f"{mod_name} Constellation", ax=ax)
        filename = f"{mod_name}_constellation_{timestamp}.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        generated.append(GeneratedPlotItem(name="constellation", filename=filename, url=f"/static/plots/{filename}"))
        
        # 2. FFT
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_fft(signal, sample_rate=sample_rate, title=f"{mod_name} Magnitude Spectrum", ax=ax)
        filename = f"{mod_name}_fft_{timestamp}.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        generated.append(GeneratedPlotItem(name="fft", filename=filename, url=f"/static/plots/{filename}"))
        
        # 3. PSD
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_psd(signal, sample_rate=sample_rate, nfft=256, title=f"{mod_name} Power Spectral Density", ax=ax)
        filename = f"{mod_name}_psd_{timestamp}.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        generated.append(GeneratedPlotItem(name="psd", filename=filename, url=f"/static/plots/{filename}"))
        
        # 4. Spectrogram
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_spectrogram(signal, sample_rate=sample_rate, nperseg=64, noverlap=32, title=f"{mod_name} Spectrogram", ax=ax)
        filename = f"{mod_name}_spectrogram_{timestamp}.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        generated.append(GeneratedPlotItem(name="spectrogram", filename=filename, url=f"/static/plots/{filename}"))
        
        # 5. Time Domain Waveform
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_iq(signal, sample_rate=sample_rate, max_points=150, title=f"{mod_name} Waveform", ax=ax)
        filename = f"{mod_name}_iq_{timestamp}.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        generated.append(GeneratedPlotItem(name="waveform", filename=filename, url=f"/static/plots/{filename}"))
        
        # 6. Eye Diagram
        if mod_name in ["BPSK", "QPSK", "16QAM", "64QAM"]:
            sps = 8
            fig = plt.figure(figsize=(10, 5))
            plot_eye_diagram(signal, samples_per_symbol=sps, symbols_to_show=2, title=f"{mod_name} Eye Diagram", fig=fig)
            filename = f"{mod_name}_eye_{timestamp}.png"
            fig.savefig(os.path.join(output_dir, filename))
            plt.close(fig)
            generated.append(GeneratedPlotItem(name="eye_diagram", filename=filename, url=f"/static/plots/{filename}"))
            
        # 7. 3D Waterfall
        waterfall_signal = signal[:512]
        fig = plt.figure(figsize=(10, 6))
        plot_waterfall(waterfall_signal, sample_rate=sample_rate, nperseg=64, noverlap=32, title=f"{mod_name} 3D Waterfall", fig=fig)
        filename = f"{mod_name}_waterfall_{timestamp}.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        generated.append(GeneratedPlotItem(name="waterfall", filename=filename, url=f"/static/plots/{filename}"))
        
        return GeneratePlotsResponse(
            success=True,
            plots=generated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {str(e)}")
