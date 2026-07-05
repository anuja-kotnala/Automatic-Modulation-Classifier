from enum import Enum, unique

@unique
class ModulationType(Enum):
    """Supported modulation schemes for classification and generation."""
    AM = "AM"
    FM = "FM"
    BPSK = "BPSK"
    QPSK = "QPSK"
    QAM16 = "16QAM"
    QAM64 = "64QAM"
    OFDM = "OFDM"

# Mapping of string representations to ModulationType
MODULATION_MAP = {mod.value: mod for mod in ModulationType}

# Mathematical and physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
STANDARD_TEMPERATURE = 290.0  # Kelvin (for noise figure reference)
