"""
Automatic Modulation Classification (AMC) & Spectrum Analyzer Package.
Provides modules for generating simulated IQ baseband signals, extracting statistical and spectral features,
classifying modulations, and performing spectrum analysis.
"""

from amc.config import AppConfig
from amc.constants import ModulationType, MODULATION_MAP
from amc.logging_config import setup_logger

__version__ = "1.0.0"

__all__ = [
    "AppConfig",
    "ModulationType",
    "MODULATION_MAP",
    "setup_logger",
]
