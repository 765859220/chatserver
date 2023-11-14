import datetime
from typing import Optional

DEFAULT_PRECISION = 2


def get_date_time_str() -> str:
    """get data time as string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def number_to_string(
    num: float, units: Optional[str] = None, precision: int = DEFAULT_PRECISION
) -> str:
    if units is None:
        if num >= 1e12:
            magnitude, units = 1e12, "T"
        elif num >= 1e9:
            magnitude, units = 1e9, "G"
        elif num >= 1e6:
            magnitude, units = 1e6, "M"
        elif num >= 1e3:
            magnitude, units = 1e3, "K"
        elif num >= 1 or num == 0:
            magnitude, units = 1, ""
        elif num >= 1e-3:
            magnitude, units = 1e-3, "m"
        else:
            magnitude, units = 1e-6, "u"
    else:
        if units == "T":
            magnitude = 1e12
        elif units == "G":
            magnitude = 1e9
        elif units == "M":
            magnitude = 1e6
        elif units == "K":
            magnitude = 1e3
        elif units == "m":
            magnitude = 1e-3
        elif units == "u":
            magnitude = 1e-6
        else:
            magnitude = 1
    return f"{round(num / magnitude, precision):g} {units}"
