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
    meta_label_scores: pd.DataFrame | None = None,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
    ma_windows: tuple[int, ...] = (20, 60),
    rs_window: int = 60,
    atr_rank_lookback: int = 120,
    volume_ma_days: int = 20,
    require_above_ma120: bool = False,
    require_positive_rs60: bool = False,
    min_meta_score: float | None = None,
) -> pd.DataFrame:
    """Produce a daily rank score for every (date, symbol) row.

    Returns a wide DataFrame (index=trade_date, columns=symbol) with scores
    in [0, 100].
    """
    df = daily_long.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    symbols = sorted(df[sym_col].unique())

    # Build wide tables for each component
    close_wide = df.pivot(index=date_col, columns=sym_col, values="close").sort_index().astype(np.float64)
    volume_wide = df.pivot(index=date_col, columns=sym_col, values="volume").sort_index().astype(np.float64)
    high_wide = df.pivot(index=date_col, columns=sym_col, values="high").sort_index().astype(np.float64)
    low_wide = df.pivot(index=date_col, columns=sym_col, values="low").sort_index().astype(np.float64)

    score = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)

    if meta_label_scores is not None:
        meta = meta_label_scores.copy()
        meta.index = pd.to_datetime(meta.index).normalize()
        meta.columns = meta.columns.astype(str).str.zfill(6)
        meta = meta.reindex(index=score.index, columns=score.columns).astype(np.float64)
        score += 0.25 * (meta.clip(0.0, 1.0) * 100.0).fillna(50.0)
    else:
        meta = None

    above_ma120 = pd.DataFrame(False, index=close_wide.index, columns=close_wide.columns)
    rs_filter = pd.DataFrame(True, index=close_wide.index, columns=close_wide.columns)
    stock_ret_by_sym: dict[str, pd.Series] = {}

    # ── 1. Trend strength (15%): MA20 slope ──
    for sym in symbols:
        c = close_wide[sym].dropna()
        if len(c) < 22:
            continue
        ma20 = c.rolling(20, min_periods=20).mean()
        slope = (ma20 / ma20.shift(5) - 1.0) * 100
        score[sym] += 0.15 * np.clip(50.0 + slope * 10.0, 0.0, 100.0)

    # ── 2. Relative strength vs index (20%) ──
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
            idx_ret = aligned.pct_change(rs_window, fill_method=None)
            for sym in symbols:
                stock_ret = close_wide[sym].pct_change(rs_window, fill_method=None)
                stock_ret_by_sym[sym] = stock_ret
                rs = stock_ret - idx_ret.reindex(stock_ret.index)
                score_rs[sym] = 0.20 * np.clip(50.0 + rs * 250.0, 0.0, 100.0)
                rs_filter[sym] = rs > 0
    else:
        for sym in symbols:
            stock_ret = close_wide[sym].pct_change(rs_window, fill_method=None)
            stock_ret_by_sym[sym] = stock_ret
            score_rs[sym] = 0.20 * np.clip(50.0 + stock_ret * 250.0, 0.0, 100.0)
            rs_filter[sym] = stock_ret > 0
    score += score_rs

    # ── 3. MA120 position (15%) and medium-term momentum (10%) ──
    for sym in symbols:
        c = close_wide[sym].dropna()
        if len(c) < 62:
            continue
        ma20 = c.rolling(20, min_periods=20).mean()
        ma60 = c.rolling(60, min_periods=60).mean()
        ma120 = c.rolling(120, min_periods=120).mean()
        above_ma120[sym] = (c > ma120).reindex(score.index, fill_value=False).astype(bool)
        ma_state = ((c > ma20).astype(float) + (ma20 > ma60).astype(float)) * 25.0
        ma120_state = (c > ma120).astype(float) * 50.0
        score[sym] += 0.15 * (ma_state + ma120_state)
        stock_ret = stock_ret_by_sym.get(sym, c.pct_change(rs_window, fill_method=None))
        score[sym] += 0.10 * np.clip(50.0 + stock_ret * 250.0, 0.0, 100.0)

    # ── 4. Donchian breakout (10%) and volatility penalty ──
    for sym in symbols:
        c = close_wide[sym].dropna()
        h = high_wide[sym].dropna()
        l = low_wide[sym].dropna()
        if len(c) < 15:
            continue
        donchian_high = h.rolling(20, min_periods=20).max()
        score[sym] += 0.10 * (c >= donchian_high.shift(1)).astype(float) * 100.0
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        atr_pct = atr / c * 100
        atr_rank = atr_pct.rolling(atr_rank_lookback, min_periods=60).apply(
            lambda x: (x.iloc[-1] >= x).mean(), raw=False,
        )
        score[sym] -= 0.05 * atr_rank * 100.0

    # ── 5. Market regime bonus: index above MA60 ──
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
            score[sym] += 0.05 * regime_bull * 100.0

    # ── 6. Liquidity (5%): volume ratio vs own 20d average ──
    for sym in symbols:
        v = volume_wide[sym].dropna()
        if len(v) < volume_ma_days + 1:
            continue
        v_ma = v.rolling(volume_ma_days, min_periods=volume_ma_days).mean()
        vol_ratio = v / v_ma
        score[sym] += 0.05 * np.clip((vol_ratio - 0.5) / 2.0, 0.0, 1.0) * 100.0

    if require_above_ma120:
        score = score.where(above_ma120, 0.0)
    if require_positive_rs60:
        score = score.where(rs_filter, 0.0)
    if meta is not None and min_meta_score is not None:
        score = score.where(meta >= float(min_meta_score), 0.0)

    return score.clip(lower=0.0, upper=100.0)
