import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import yaml

@dataclass
class ProjectConfig:
    name: str = "Automatic Modulation Classification & Spectrum Analyzer"
    version: str = "1.0.0"

@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file_path: str = "logs/amc_project.log"

@dataclass
class GeneratorConfig:
    sample_rate: float = 1000000.0
    center_frequency: float = 0.0
    samples_per_symbol: int = 8
    pulse_shaping: str = "rrc"
    roll_off: float = 0.35
    snr_range: List[int] = field(default_factory=lambda: [-10, 20])
    snr_step: int = 2
    samples_per_signal: int = 1024
    signals_per_modulation: int = 1000

@dataclass
class FeaturesConfig:
    fft_size: int = 1024
    window_type: str = "hamming"
    cumulants: List[int] = field(default_factory=lambda: [20, 21, 40, 42, 60, 63])
    wavelet_type: str = "db4"

@dataclass
class ClassifierConfig:
    type: str = "deep_learning"
    model_name: str = "CNN1D"
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    train_split: float = 0.8
    random_seed: int = 42

@dataclass
class AnalyzerConfig:
    nfft: int = 1024
    overlap: int = 512
    window: str = "hann"
    peak_threshold_db: float = -50.0
    min_peak_distance: int = 10

@dataclass
class AWGNConfig:
    enabled: bool = True
    snr_db: float = 20.0

@dataclass
class RayleighConfig:
    enabled: bool = False
    doppler_freq: float = 10.0

@dataclass
class RicianConfig:
    enabled: bool = False
    k_factor: float = 4.0
    doppler_freq: float = 5.0

@dataclass
class FrequencyOffsetConfig:
    enabled: bool = False
    offset_hz: float = 50.0

@dataclass
class PhaseNoiseConfig:
    enabled: bool = False
    std_dev: float = 0.05

@dataclass
class IQImbalanceConfig:
    enabled: bool = False
    amplitude_imbalance_db: float = 0.5
    phase_imbalance_deg: float = 3.0

@dataclass
class TimingOffsetConfig:
    enabled: bool = False
    fractional_delay: float = 0.25

@dataclass
class MultipathConfig:
    enabled: bool = False
    delays: List[float] = field(default_factory=lambda: [0.0, 1.2e-6, 2.5e-6])
    gains_db: List[float] = field(default_factory=lambda: [0.0, -3.0, -9.0])

@dataclass
class ClockDriftConfig:
    enabled: bool = False
    ppm: float = 20.0

@dataclass
class ChannelConfig:
    awgn: AWGNConfig = field(default_factory=AWGNConfig)
    rayleigh: RayleighConfig = field(default_factory=RayleighConfig)
    rician: RicianConfig = field(default_factory=RicianConfig)
    frequency_offset: FrequencyOffsetConfig = field(default_factory=FrequencyOffsetConfig)
    phase_noise: PhaseNoiseConfig = field(default_factory=PhaseNoiseConfig)
    iq_imbalance: IQImbalanceConfig = field(default_factory=IQImbalanceConfig)
    timing_offset: TimingOffsetConfig = field(default_factory=TimingOffsetConfig)
    multipath: MultipathConfig = field(default_factory=MultipathConfig)
    clock_drift: ClockDriftConfig = field(default_factory=ClockDriftConfig)

@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "AppConfig":
        """Loads and parses configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        project = ProjectConfig(**data.get("project", {}))
        logging = LoggingConfig(**data.get("logging", {}))
        generator = GeneratorConfig(**data.get("generator", {}))
        features = FeaturesConfig(**data.get("features", {}))
        classifier = ClassifierConfig(**data.get("classifier", {}))
        analyzer = AnalyzerConfig(**data.get("analyzer", {}))
        
        # Load nested channel config
        ch_data = data.get("channel", {})
        channel = ChannelConfig(
            awgn=AWGNConfig(**ch_data.get("awgn", {})),
            rayleigh=RayleighConfig(**ch_data.get("rayleigh", {})),
            rician=RicianConfig(**ch_data.get("rician", {})),
            frequency_offset=FrequencyOffsetConfig(**ch_data.get("frequency_offset", {})),
            phase_noise=PhaseNoiseConfig(**ch_data.get("phase_noise", {})),
            iq_imbalance=IQImbalanceConfig(**ch_data.get("iq_imbalance", {})),
            timing_offset=TimingOffsetConfig(**ch_data.get("timing_offset", {})),
            multipath=MultipathConfig(**ch_data.get("multipath", {})),
            clock_drift=ClockDriftConfig(**ch_data.get("clock_drift", {}))
        )

        return cls(
            project=project,
            logging=logging,
            generator=generator,
            features=features,
            classifier=classifier,
            analyzer=analyzer,
            channel=channel
        )

