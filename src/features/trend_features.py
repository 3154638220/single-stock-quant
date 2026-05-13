"""Trend strength, volatility, and relative-strength features.

All features are point-in-time: no look-ahead bias by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trend_features(
    ohlcv: pd.DataFrame,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    ma_windows: tuple[int, ...] = (20, 60),
    donchian_window: int = 20,
    atr_window: int = 14,
    atr_rank_lookback: int = 120,
    rs_windows: tuple[int, ...] = (60, 120),
) -> pd.DataFrame:
    """Compute trend-strength features from daily OHLCV.

    Returns a DataFrame with the same index as *ohlcv*.
    """
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    high = pd.to_numeric(ohlcv["high"], errors="coerce")
    low = pd.to_numeric(ohlcv["low"], errors="coerce")
    volume = pd.to_numeric(ohlcv["volume"], errors="coerce")
    out = pd.DataFrame(index=ohlcv.index)

    # ── Moving average slopes ──
    for w in ma_windows:
        ma = close.rolling(w, min_periods=w).mean()
        out[f"ma{w}"] = ma
        out[f"ma{w}_slope"] = (ma / ma.shift(5) - 1.0) * 100
        out[f"close_above_ma{w}"] = close > ma

    # ── Donchian breakout ──
    donchian_high = high.rolling(donchian_window, min_periods=donchian_window).max()
    donchian_low = low.rolling(donchian_window, min_periods=donchian_window).min()
    out[f"donchian_high_{donchian_window}"] = donchian_high
    out[f"donchian_low_{donchian_window}"] = donchian_low
    out[f"donchian_breakout_{donchian_window}"] = close >= donchian_high.shift(1)

    # ── ATR-based volatility ──
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    out["atr_pct"] = atr / close * 100
    out["atr_pct_rank"] = (
        out["atr_pct"]
        .rolling(atr_rank_lookback, min_periods=atr_rank_lookback)
        .apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    )

    # ── Volume features ──
    avg_vol_20 = volume.rolling(20, min_periods=20).mean()
    out["volume_ratio_20"] = volume / avg_vol_20
    out["volume_ratio_5"] = volume / volume.rolling(5, min_periods=5).mean()

    # ── Relative strength vs index ──
    for w in rs_windows:
        stock_ret = close.pct_change(w)
        out[f"stock_ret_{w}"] = stock_ret
        if index_ohlcv is not None and not index_ohlcv.empty:
            idx_close = pd.to_numeric(index_ohlcv["close"], errors="coerce")
            idx_dates = pd.to_datetime(index_ohlcv["trade_date"]).dt.normalize()
            stock_dates = pd.to_datetime(ohlcv["trade_date"]).dt.normalize()
            idx_ret_map = dict(zip(idx_dates, idx_close.pct_change(w)))
            idx_ret_aligned = pd.Series(
                [idx_ret_map.get(d, np.nan) for d in stock_dates],
                index=ohlcv.index,
                dtype=np.float64,
            )
            out[f"index_ret_{w}"] = idx_ret_aligned
            out[f"rs_{w}"] = stock_ret - idx_ret_aligned
        else:
            out[f"index_ret_{w}"] = np.nan
            out[f"rs_{w}"] = np.nan

    return out
