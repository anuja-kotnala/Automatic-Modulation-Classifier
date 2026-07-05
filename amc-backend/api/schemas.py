from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status (e.g. 'ok')")
    version: str = Field(..., description="Project version")
    models_loaded: Dict[str, bool] = Field(..., description="Dictionary of models and their load status")

class PlotFile(BaseModel):
    name: str = Field(..., description="Filename of the plot")
    category: str = Field(..., description="Category of the plot (e.g. 'signal', 'evaluation')")
    url: str = Field(..., description="Web URL to access the plot image")

class ModelPerformanceRow(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model ('ml' or 'dl')")
    accuracy: float = Field(..., description="Overall or test accuracy")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 harmonic score")
    cv_mean: Optional[float] = Field(None, description="Cross-validation mean accuracy (ML only)")
    cv_std: Optional[float] = Field(None, description="Cross-validation standard deviation (ML only)")
    macro_auc: Optional[float] = Field(None, description="Macro Area Under ROC Curve (ML only)")
    parameters: Optional[int] = Field(None, description="Number of parameters (DL only)")
    training_time: Optional[float] = Field(None, description="Training time in seconds (DL only)")
    avg_epoch_time: Optional[float] = Field(None, description="Average epoch training time in seconds (DL only)")
    model_size_mb: Optional[float] = Field(None, description="Model checkpoint size in MB (DL only)")
    best_val_accuracy: Optional[float] = Field(None, description="Best validation accuracy achieved (DL only)")
    best_val_loss: Optional[float] = Field(None, description="Best validation loss achieved (DL only)")
    inference_time_ms: Optional[float] = Field(None, description="Average inference time per sample in milliseconds (DL only)")
    snr_accuracy: Optional[Dict[str, float]] = Field(None, description="Accuracy per SNR level (DL only)")
    
    # Aliases/fallbacks for frontend backward compatibility
    model: Optional[str] = Field(None, description="Legacy model name key")
    Classifier: Optional[str] = Field(None, description="Legacy classifier name key")

    class Config:
        extra = "allow"  # Keep schema extensible for future model metrics

class EvaluationResults(BaseModel):
    summary: List[ModelPerformanceRow] = Field(..., description="Unified list of model performance summary rows")
    available_files: List[str] = Field(..., description="List of available evaluation plot files")

class PredictRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Relative path of signal npy file under dataset directory")
    iq_data: Optional[List[List[float]]] = Field(None, description="2D list of shape [2, N] representing [real, imag] signal arrays")
    model_name: str = Field("randomforest", description="Classifier model to use: randomforest, svm, gradientboosting, xgboost, lightgbm, knn, logisticregression")

class PredictResponse(BaseModel):
    success: bool
    predicted_class: str = Field(..., description="Predicted modulation scheme name")
    confidence: float = Field(..., description="Prediction confidence/probability score")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Probabilities for each modulation type")
    features: Dict[str, float] = Field(..., description="Extracted features dictionary")

class PredictDLRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Relative path of signal npy file under dataset directory")
    iq_data: Optional[List[List[float]]] = Field(None, description="2D list of shape [2, N] representing [real, imag] signal arrays")
    model_name: str = Field("cnn1d", description="Deep learning model to use: cnn1d, cnnlstm, cnn2d")

class HealthDLResponse(BaseModel):
    status: str = Field(..., description="API health status (e.g. 'ok')")
    models_loaded: Dict[str, bool] = Field(..., description="Dictionary of DL models and their load status")

class ChannelImpairmentOverrides(BaseModel):
    awgn_enabled: Optional[bool] = None
    rayleigh_enabled: Optional[bool] = None
    rician_enabled: Optional[bool] = None
    frequency_offset_enabled: Optional[bool] = None
    phase_noise_enabled: Optional[bool] = None
    iq_imbalance_enabled: Optional[bool] = None
    timing_offset_enabled: Optional[bool] = None
    multipath_enabled: Optional[bool] = None
    clock_drift_enabled: Optional[bool] = None
    
    # Values
    awgn_snr_db: Optional[float] = None
    rayleigh_doppler_freq: Optional[float] = None
    rician_k_factor: Optional[float] = None
    rician_doppler_freq: Optional[float] = None
    frequency_offset_hz: Optional[float] = None
    phase_noise_std_dev: Optional[float] = None
    iq_imbalance_amplitude_db: Optional[float] = None
    iq_imbalance_phase_deg: Optional[float] = None
    timing_offset_fractional_delay: Optional[float] = None
    multipath_delays: Optional[List[float]] = None
    multipath_gains_db: Optional[List[float]] = None
    clock_drift_ppm: Optional[float] = None

class GenerateSignalRequest(BaseModel):
    modulation_type: str = Field(..., description="Modulation type (e.g. BPSK, QPSK, 16QAM, 64QAM, AM, FM, OFDM)")
    snr_db: float = Field(20.0, description="Signal-to-Noise ratio in dB")
    sample_rate: float = Field(1000000.0, description="Sample rate in Hz")
    num_samples: int = Field(1024, description="Number of samples to generate")
    impairments: Optional[ChannelImpairmentOverrides] = Field(None, description="Channel impairment overrides")

class GenerateSignalResponse(BaseModel):
    success: bool
    modulation_type: str
    snr_db: float
    iq_data: List[List[float]] = Field(..., description="2D list of shape [2, N] representing [real, imag] signal arrays")
    file_path: Optional[str] = Field(None, description="Relative file path where the signal was saved under dataset directory")

class GeneratePlotsRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Relative path of signal npy file under dataset directory")
    iq_data: Optional[List[List[float]]] = Field(None, description="2D list of shape [2, N] representing [real, imag] signal arrays")
    modulation_type: str = Field(..., description="Modulation scheme name")
    sample_rate: float = Field(1000000.0, description="Sample rate in Hz")

class GeneratedPlotItem(BaseModel):
    name: str = Field(..., description="Plot type (e.g. constellation, fft, psd, spectrogram, waveform, eye_diagram, waterfall)")
    filename: str = Field(..., description="Saved plot filename")
    url: str = Field(..., description="Web URL to access the plot image")

class GeneratePlotsResponse(BaseModel):
    success: bool
    plots: List[GeneratedPlotItem] = Field(..., description="List of generated diagnostic plots")
