"""Signal ranking for watchlist cross-sectional selection.

Scores every stock on every trading day so the allocator can pick the top N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_signals(
    daily_long: pd.DataFrame,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
    ma_windows: tuple[int, ...] = (20, 60),
    rs_window: int = 60,
    atr_rank_lookback: int = 120,
    volume_ma_days: int = 20,
) -> pd.DataFrame:
    """Produce a daily rank score for every (date, symbol) row.

    Returns a wide DataFrame (index=trade_date, columns=symbol) with scores
    in [0, 100].
    """
    df = daily_long.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    dates_all = sorted(df[date_col].unique())
    symbols = sorted(df[sym_col].unique())

    # Build wide tables for each component
    close_wide = df.pivot(index=date_col, columns=sym_col, values="close").sort_index().astype(np.float64)
    volume_wide = df.pivot(index=date_col, columns=sym_col, values="volume").sort_index().astype(np.float64)
    high_wide = df.pivot(index=date_col, columns=sym_col, values="high").sort_index().astype(np.float64)
    low_wide = df.pivot(index=date_col, columns=sym_col, values="low").sort_index().astype(np.float64)

    score = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)

    # ── 1. Trend strength (20%): MA20 slope ──
    for sym in symbols:
        c = close_wide[sym].dropna()
        if len(c) < 22:
            continue
        ma20 = c.rolling(20, min_periods=20).mean()
        slope = (ma20 / ma20.shift(5) - 1.0) * 100
        score[sym] += 0.20 * np.clip(slope * 5, -20, 20)

    # ── 2. Relative strength vs index (25%) ──
    score_rs = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    if index_ohlcv is not None and not index_ohlcv.empty:
        idx_close = pd.to_numeric(index_ohlcv["close"], errors="coerce")
        idx_dates = pd.to_datetime(index_ohlcv["trade_date"]).dt.normalize()
        idx_map = dict(zip(idx_dates, idx_close))
        aligned = pd.Series(
            [idx_map.get(d, np.nan) for d in close_wide.index],
            index=close_wide.index, dtype=np.float64,
        ).dropna()
        if len(aligned) > rs_window:
            idx_ret = aligned.pct_change(rs_window)
            for sym in symbols:
                stock_ret = close_wide[sym].pct_change(rs_window)
                rs = stock_ret - idx_ret.reindex(stock_ret.index)
                score_rs[sym] = 0.25 * np.clip(rs * 5, -20, 20)
    else:
        for sym in symbols:
            stock_ret = close_wide[sym].pct_change(rs_window)
            score_rs[sym] = 0.25 * np.clip(stock_ret * 5, -20, 20)
    score += score_rs

    # ── 3. Signal quality proxy (30%): MA trend alignment ──
    for sym in symbols:
        c = close_wide[sym].dropna()
        if len(c) < 62:
            continue
        ma20 = c.rolling(20, min_periods=20).mean()
        ma60 = c.rolling(60, min_periods=60).mean()
        above_ma = (c > ma20).astype(float) * 10
        ma_aligned = (ma20 > ma60).astype(float) * 10
        close_to_ma = np.clip((c / ma20 - 1.0) * 100, -5, 5) * 2
        score[sym] += 0.30 * (above_ma + ma_aligned + close_to_ma)

    # ── 4. Volatility penalty (-20%): high ATR rank → penalise ──
    for sym in symbols:
        c = close_wide[sym].dropna()
        h = high_wide[sym].dropna()
        l = low_wide[sym].dropna()
        if len(c) < 15:
            continue
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        atr_pct = atr / c * 100
        atr_rank = atr_pct.rolling(atr_rank_lookback, min_periods=60).apply(
            lambda x: (x.iloc[-1] >= x).mean(), raw=False,
        )
        score[sym] -= 0.20 * atr_rank * 20  # penalty up to -4

    # ── 5. Market regime (15%): index above MA60 ──
    if index_ohlcv is not None and not index_ohlcv.empty:
        idx_close = pd.to_numeric(index_ohlcv["close"], errors="coerce")
        idx_dates = pd.to_datetime(index_ohlcv["trade_date"]).dt.normalize()
        idx_map_c = dict(zip(idx_dates, idx_close))
        aligned_c = pd.Series(
            [idx_map_c.get(d, np.nan) for d in close_wide.index],
            index=close_wide.index, dtype=np.float64,
        ).dropna()
        idx_ma60 = aligned_c.rolling(60, min_periods=60).mean()
        regime_bull = (aligned_c > idx_ma60).astype(float)
        for sym in symbols:
            score[sym] += 0.15 * regime_bull * 10

    # ── 6. Liquidity (10%): volume ratio vs own 20d average ──
    for sym in symbols:
        v = volume_wide[sym].dropna()
        if len(v) < volume_ma_days + 1:
            continue
        v_ma = v.rolling(volume_ma_days, min_periods=volume_ma_days).mean()
        vol_ratio = v / v_ma
        score[sym] += 0.10 * np.clip(vol_ratio - 0.5, 0, 3) * 5

    return score.clip(lower=0.0, upper=100.0)
