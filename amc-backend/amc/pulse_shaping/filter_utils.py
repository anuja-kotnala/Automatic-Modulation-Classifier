from amc.pulse_shaping.base import BasePulseFilter
from amc.pulse_shaping.raised_cosine import RaisedCosineFilter
from amc.pulse_shaping.root_raised_cosine import RootRaisedCosineFilter
from amc.pulse_shaping.rectangular import RectangularFilter

def get_pulse_filter(
    filter_type: str,
    samples_per_symbol: int,
    roll_off: float = 0.35,
    num_symbols_span: int = 8
) -> BasePulseFilter:
    """
    Factory function to retrieve a pulse filter instance.

    Args:
        filter_type: Name of the filter ("rc", "raised_cosine", "rrc", "root_raised_cosine", "rect", "rectangular").
        samples_per_symbol: Oversampling factor.
        roll_off: Roll-off factor (ignored for rectangular).
        num_symbols_span: Symbol truncation length.

    Returns:
        BasePulseFilter: An instance of the chosen filter.
    """
    f_type = filter_type.lower().strip()
    
    if f_type in ["rc", "raised_cosine"]:
        return RaisedCosineFilter(
            samples_per_symbol=samples_per_symbol,
            num_symbols_span=num_symbols_span,
            roll_off=roll_off
        )
    elif f_type in ["rrc", "root_raised_cosine"]:
        return RootRaisedCosineFilter(
            samples_per_symbol=samples_per_symbol,
            num_symbols_span=num_symbols_span,
            roll_off=roll_off
        )
    elif f_type in ["rect", "rectangular"]:
        return RectangularFilter(
            samples_per_symbol=samples_per_symbol
        )
    else:
        raise ValueError(f"Unknown pulse shaping filter type: {filter_type}")
