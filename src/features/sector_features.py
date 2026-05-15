"""Sector/industry relative strength features.

Provides ``industry_rs_20``: the stock's 20-day excess return relative to
its industry or benchmark index.  When an industry-mapping CSV is provided it
looks up the stock's sector index; otherwise it falls back to the broad-market
benchmark (e.g. CSI 300 / 510300).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _map_index_close(
    stock_dates: pd.DatetimeIndex,
    index_ohlcv: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (aligned_close, aligned_ret20) arrays matching *stock_dates*."""
    idx_dates = pd.to_datetime(index_ohlcv["trade_date"]).dt.normalize()
    idx_close = pd.to_numeric(index_ohlcv["close"], errors="coerce")
    idx_ret20 = idx_close.pct_change(20)
    idx_map_close = dict(zip(idx_dates, idx_close))
    idx_map_ret20 = dict(zip(idx_dates, idx_ret20))
    aligned_close = np.full(len(stock_dates), np.nan, dtype=np.float64)
    aligned_ret20 = np.full(len(stock_dates), np.nan, dtype=np.float64)
    for i, d in enumerate(stock_dates):
        aligned_close[i] = idx_map_close.get(d, np.nan)
        aligned_ret20[i] = idx_map_ret20.get(d, np.nan)
    return aligned_close, aligned_ret20


def compute_industry_rs_20(
    ohlcv: pd.DataFrame,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    industry_index_ohlcv: pd.DataFrame | None = None,
) -> pd.Series:
    """20-day excess return of a stock relative to its industry or benchmark.

    Parameters
    ----------
    ohlcv:
        Stock OHLCV DataFrame (must have ``trade_date`` and ``close``).
    index_ohlcv:
        Broad-market benchmark (e.g. CSI 300). Used as fallback when
        *industry_index_ohlcv* is not available.
    industry_index_ohlcv:
        Industry/sector index for this specific stock. Takes precedence
        over *index_ohlcv* when both are provided.

    Returns
    -------
    pd.Series
        ``(stock_ret_20 - index_ret_20)`` aligned to *ohlcv* index.
    """
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    stock_ret_20 = close.pct_change(20)

    stock_dates = pd.to_datetime(ohlcv["trade_date"]).dt.normalize()

    # Prefer industry index, fall back to broad benchmark, then zero
    ref = industry_index_ohlcv if (industry_index_ohlcv is not None and not industry_index_ohlcv.empty) else index_ohlcv

    if ref is not None and not ref.empty:
        _, ref_ret20 = _map_index_close(stock_dates, ref)
        rs = stock_ret_20.to_numpy(dtype=np.float64) - ref_ret20
    else:
        rs = stock_ret_20.to_numpy(dtype=np.float64)

    result = pd.Series(rs, index=ohlcv.index, dtype=np.float64)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.fillna(0.0)
