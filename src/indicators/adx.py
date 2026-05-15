"""Average Directional Index (ADX) — trend strength indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_adx(
    ohlcv: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Compute ADX, +DI, -DI for a DataFrame with high, low, close columns.

    Returns a DataFrame with columns: adx, plus_di, minus_di.
    ADX > 25 typically indicates a trending market; ADX < 20 indicates ranging.
    """
    high = pd.to_numeric(ohlcv["high"], errors="coerce")
    low = pd.to_numeric(ohlcv["low"], errors="coerce")
    close = pd.to_numeric(ohlcv["close"], errors="coerce")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = true_range.rolling(period, min_periods=period).mean()
    smooth_plus_dm = pd.Series(plus_dm, index=ohlcv.index).rolling(period, min_periods=period).mean()
    smooth_minus_dm = pd.Series(minus_dm, index=ohlcv.index).rolling(period, min_periods=period).mean()

    plus_di = 100.0 * smooth_plus_dm / atr
    minus_di = 100.0 * smooth_minus_dm / atr

    dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period, min_periods=period).mean()

    out = ohlcv[["trade_date"]].copy() if "trade_date" in ohlcv.columns else pd.DataFrame(index=ohlcv.index)
    out["adx"] = adx.values
    out["plus_di"] = plus_di.values
    out["minus_di"] = minus_di.values
    return out
