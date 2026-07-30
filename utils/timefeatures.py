"""Remote-sensing time features for HLS day-of-year stamps."""
from typing import List

import numpy as np
import pandas as pd


class TimeFeature:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class DayOfYearSin(TimeFeature):
    """Sine encoding of day of year (annual cycle)."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        dayofyear = index.dayofyear - 1
        days_in_year = index.is_leap_year.astype(int) + 365
        angle = 2 * np.pi * dayofyear / days_in_year
        return np.sin(angle)


class DayOfYearCos(TimeFeature):
    """Cosine encoding of day of year (annual cycle)."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        dayofyear = index.dayofyear - 1
        days_in_year = index.is_leap_year.astype(int) + 365
        angle = 2 * np.pi * dayofyear / days_in_year
        return np.cos(angle)


def time_features_from_frequency_str(freq_str: str, **_kwargs) -> List[TimeFeature]:
    """Return time-feature extractors.

    This package only supports remote-sensing mode ``rs`` (day-of-year sin/cos).
    """
    if str(freq_str).upper() != "RS":
        raise RuntimeError(
            f"Unsupported frequency {freq_str!r}. "
            "This package only supports freq='rs' (HLS remote sensing)."
        )
    return [DayOfYearSin(), DayOfYearCos()]


def time_features(dates, freq="rs", start_year: int = 2013, end_year: int = 2025):
    features = time_features_from_frequency_str(freq, start_year=start_year, end_year=end_year)
    return np.vstack([feat(dates) for feat in features])
