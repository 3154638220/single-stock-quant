"""Indicator helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA with the first valid value available after ``period`` samples."""
    if period <= 0:
        raise ValueError("period must be positive")
    s = pd.to_numeric(series, errors="coerce")
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def ema_seeded(series: pd.Series, period: int) -> pd.Series:
    """EMA seeded with SMA of the first ``period`` values to reduce cold-start bias.

    Standard ``ewm(adjust=False)`` initialises with the first observation, which
    biases long-period EMAs for months.  This variant primes the first EMA value
    with the SMA window so the recursive smoothing starts from a proper centre.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    s = pd.to_numeric(series, errors="coerce")
    sma = s.rolling(period, min_periods=period).mean()
    first_valid = sma.first_valid_index()
    if first_valid is None:
        return pd.Series(np.nan, index=s.index, dtype=float)
    alpha = 2.0 / (period + 1)
    values = s.values.astype(float)
    result = np.full(len(values), np.nan, dtype=float)
    first_pos = s.index.get_loc(first_valid)
    result[first_pos] = sma.iloc[first_pos]
    for i in range(first_pos + 1, len(values)):
        if not np.isnan(values[i]) and not np.isnan(result[i - 1]):
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return pd.Series(result, index=s.index, dtype=float)


def highest(series: pd.Series, period: int) -> pd.Series:
    """Rolling highest value over ``period`` bars."""
    if period <= 0:
        raise ValueError("period must be positive")
    return pd.to_numeric(series, errors="coerce").rolling(period, min_periods=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """Rolling lowest value over ``period`` bars."""
    if period <= 0:
        raise ValueError("period must be positive")
    return pd.to_numeric(series, errors="coerce").rolling(period, min_periods=period).min()
