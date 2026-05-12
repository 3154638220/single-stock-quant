"""Indicator helper functions."""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA with the first valid value available after ``period`` samples."""
    if period <= 0:
        raise ValueError("period must be positive")
    s = pd.to_numeric(series, errors="coerce")
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


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
