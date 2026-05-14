"""Signal ranking for watchlist cross-sectional selection.

Scores every stock on every trading day so the allocator can pick the top N.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import DKTrendParams, compute_dktrend
from src.portfolio.attribution import calibrate_scores_by_forward_returns

_RANKING_PROFILES = {
    "balanced": {
        "meta": 0.25,
        "trend": 0.15,
        "rs": 0.20,
        "ma": 0.15,
        "momentum": 0.10,
        "donchian": 0.10,
        "volatility_penalty": 0.05,
        "regime": 0.05,
        "liquidity": 0.05,
        "require_dk_red": False,
    },
    "meta_priority": {
        "meta": 0.55,
        "trend": 0.08,
        "rs": 0.08,
        "ma": 0.08,
        "momentum": 0.05,
        "donchian": 0.06,
        "volatility_penalty": 0.05,
        "regime": 0.03,
        "liquidity": 0.04,
        "require_dk_red": False,
    },
    "dk_meta": {
        "meta": 0.65,
        "trend": 0.07,
        "rs": 0.06,
        "ma": 0.06,
        "momentum": 0.04,
        "donchian": 0.05,
        "volatility_penalty": 0.04,
        "regime": 0.03,
        "liquidity": 0.04,
        "require_dk_red": True,
        "max_dk_run_len": None,
    },
    "dk_fresh_meta": {
        "meta": 0.65,
        "trend": 0.07,
        "rs": 0.06,
        "ma": 0.06,
        "momentum": 0.04,
        "donchian": 0.05,
        "volatility_penalty": 0.04,
        "regime": 0.03,
        "liquidity": 0.04,
        "require_dk_red": True,
        "max_dk_run_len": 20,
    },
    "dk_calibrated_meta": {
        "meta": 0.65,
        "trend": 0.07,
        "rs": 0.06,
        "ma": 0.06,
        "momentum": 0.04,
        "donchian": 0.05,
        "volatility_penalty": 0.04,
        "regime": 0.03,
        "liquidity": 0.04,
        "require_dk_red": True,
        "max_dk_run_len": None,
        "calibrate_forward_returns": True,
        "calibration_horizon": 5,
        "calibration_lookback_days": 252,
        "calibration_quantiles": 5,
        "calibration_min_observations": 100,
        "calibration_strength": 0.70,
    },
    "dk_rolling_greylist": {
        "meta": 0.65,
        "trend": 0.07,
        "rs": 0.06,
        "ma": 0.06,
        "momentum": 0.04,
        "donchian": 0.05,
        "volatility_penalty": 0.04,
        "regime": 0.03,
        "liquidity": 0.04,
        "require_dk_red": True,
        "max_dk_run_len": None,
        "rolling_greylist_lookback": 126,
        "rolling_greylist_horizon": 5,
        "rolling_greylist_threshold": -0.01,
        "rolling_greylist_scale": 0.0,
        "rolling_greylist_min_samples": 5,
    },
}


def rank_signals(
    daily_long: pd.DataFrame,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    meta_label_scores: pd.DataFrame | None = None,
    ranking_profile: str = "balanced",
    dk_params: DKTrendParams | None = None,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
    ma_windows: tuple[int, ...] = (20, 60),
    rs_window: int = 60,
    atr_rank_lookback: int = 120,
    volume_ma_days: int = 20,
    require_above_ma120: bool = False,
    require_positive_rs60: bool = False,
    min_meta_score: float | None = None,
    exclude_symbols: list[str] | tuple[str, ...] | set[str] | None = None,
    greylist_symbols: list[str] | tuple[str, ...] | set[str] | None = None,
    greylist_score_scale: float = 0.50,
    rolling_greylist_lookback: int = 0,
    rolling_greylist_horizon: int = 5,
    rolling_greylist_threshold: float = -0.01,
    rolling_greylist_scale: float = 0.0,
    rolling_greylist_min_samples: int = 5,
) -> pd.DataFrame:
    """Produce a daily rank score for every (date, symbol) row.

    Returns a wide DataFrame (index=trade_date, columns=symbol) with scores
    in [0, 100].

    ``ranking_profile`` controls the cross-sectional experiment:
    ``balanced`` keeps the stage-10 weights, ``meta_priority`` makes p_win the
    dominant rank input, ``dk_meta`` requires a DK red trend state,
    ``dk_fresh_meta`` further excludes red trends older than 20 trading days,
    ``dk_calibrated_meta`` applies rolling forward-return bucket calibration,
    and ``dk_rolling_greylist`` adds point-in-time rolling symbol greylisting
    on top of ``dk_meta``.

    ``exclude_symbols`` and ``greylist_symbols`` are static structural
    guardrails for symbol-level experiments.

    ``rolling_greylist_*`` enables point-in-time dynamic greylisting: for each
    date, symbols whose realized forward returns over the lookback window fall
    below the threshold are suppressed (score zeroed or scaled). Only past
    signals whose forward returns are fully observable by the current date are
    used, avoiding look-ahead bias.
    """
    weights = _ranking_profile_weights(ranking_profile)
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
        score += weights["meta"] * (meta.clip(0.0, 1.0) * 100.0).fillna(50.0)
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
        score[sym] += weights["trend"] * np.clip(50.0 + slope * 10.0, 0.0, 100.0)

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
                score_rs[sym] = weights["rs"] * np.clip(50.0 + rs * 250.0, 0.0, 100.0)
                rs_filter[sym] = rs > 0
    else:
        for sym in symbols:
            stock_ret = close_wide[sym].pct_change(rs_window, fill_method=None)
            stock_ret_by_sym[sym] = stock_ret
            score_rs[sym] = weights["rs"] * np.clip(50.0 + stock_ret * 250.0, 0.0, 100.0)
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
        score[sym] += weights["ma"] * (ma_state + ma120_state)
        stock_ret = stock_ret_by_sym.get(sym, c.pct_change(rs_window, fill_method=None))
        score[sym] += weights["momentum"] * np.clip(50.0 + stock_ret * 250.0, 0.0, 100.0)

    # ── 4. Donchian breakout (10%) and volatility penalty ──
    for sym in symbols:
        c = close_wide[sym].dropna()
        h = high_wide[sym].dropna()
        low = low_wide[sym].dropna()
        if len(c) < 15:
            continue
        donchian_high = h.rolling(20, min_periods=20).max()
        score[sym] += weights["donchian"] * (c >= donchian_high.shift(1)).astype(float) * 100.0
        prev_c = c.shift(1)
        tr = pd.concat([h - low, (h - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        atr_pct = atr / c * 100
        atr_rank = atr_pct.rolling(atr_rank_lookback, min_periods=60).apply(
            lambda x: (x.iloc[-1] >= x).mean(), raw=False,
        )
        score[sym] -= weights["volatility_penalty"] * atr_rank * 100.0

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
            score[sym] += weights["regime"] * regime_bull * 100.0

    # ── 6. Liquidity (5%): volume ratio vs own 20d average ──
    for sym in symbols:
        v = volume_wide[sym].dropna()
        if len(v) < volume_ma_days + 1:
            continue
        v_ma = v.rolling(volume_ma_days, min_periods=volume_ma_days).mean()
        vol_ratio = v / v_ma
        score[sym] += weights["liquidity"] * np.clip((vol_ratio - 0.5) / 2.0, 0.0, 1.0) * 100.0

    if require_above_ma120:
        score = score.where(above_ma120, 0.0)
    if require_positive_rs60:
        score = score.where(rs_filter, 0.0)
    if meta is not None and min_meta_score is not None:
        score = score.where(meta >= float(min_meta_score), 0.0)
    if weights["require_dk_red"]:
        dk_candidate = _compute_dk_candidate_mask(
            df,
            index=score.index,
            columns=score.columns,
            date_col=date_col,
            sym_col=sym_col,
            dk_params=dk_params,
            max_run_len=weights.get("max_dk_run_len"),
        )
        score = score.where(dk_candidate, 0.0)

    if weights.get("calibrate_forward_returns"):
        score = calibrate_scores_by_forward_returns(
            df,
            score,
            horizon=int(weights.get("calibration_horizon", 5)),
            lookback_days=int(weights.get("calibration_lookback_days", 252)),
            n_quantiles=int(weights.get("calibration_quantiles", 5)),
            min_observations=int(weights.get("calibration_min_observations", 100)),
            calibration_strength=float(weights.get("calibration_strength", 0.70)),
            min_score=0.0,
            date_col=date_col,
            sym_col=sym_col,
        )

    excluded = _normalize_symbol_set(exclude_symbols)
    if excluded:
        blocked_cols = [sym for sym in score.columns if sym in excluded]
        if blocked_cols:
            score.loc[:, blocked_cols] = 0.0

    greylisted = _normalize_symbol_set(greylist_symbols) - excluded
    if greylisted:
        scale = float(np.clip(greylist_score_scale, 0.0, 1.0))
        scaled_cols = [sym for sym in score.columns if sym in greylisted]
        if scaled_cols:
            score.loc[:, scaled_cols] *= scale

    # ── E21 Rolling greylist: point-in-time dynamic symbol suppression ──
    rl_lookback = max(int(rolling_greylist_lookback), int(weights.get("rolling_greylist_lookback", 0)))
    if rl_lookback > 0:
        rl_horizon = int(rolling_greylist_horizon) if rolling_greylist_horizon != 5 else int(weights.get("rolling_greylist_horizon", 5))
        rl_threshold = float(rolling_greylist_threshold) if rolling_greylist_threshold != -0.01 else float(weights.get("rolling_greylist_threshold", -0.01))
        rl_scale = float(rolling_greylist_scale) if rolling_greylist_scale != 0.0 else float(weights.get("rolling_greylist_scale", 0.0))
        rl_min_samples = int(rolling_greylist_min_samples) if rolling_greylist_min_samples != 5 else int(weights.get("rolling_greylist_min_samples", 5))
        greylist_mask = _compute_rolling_greylist_mask(
            df,
            score,
            lookback=rl_lookback,
            horizon=rl_horizon,
            threshold=rl_threshold,
            min_samples=rl_min_samples,
            date_col=date_col,
            sym_col=sym_col,
        )
        if rl_scale <= 0.0:
            score = score.where(~greylist_mask, 0.0)
        else:
            discount = score * float(np.clip(rl_scale, 0.0, 1.0))
            score = score.where(~greylist_mask, discount)

    return score.clip(lower=0.0, upper=100.0)


def _ranking_profile_weights(profile: str) -> dict[str, Any]:
    key = str(profile or "balanced").strip().lower().replace("-", "_")
    if key not in _RANKING_PROFILES:
        valid = ", ".join(sorted(_RANKING_PROFILES))
        raise ValueError(f"unknown ranking_profile {profile!r}; expected one of: {valid}")
    return dict(_RANKING_PROFILES[key])


def _normalize_symbol_set(symbols: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    if not symbols:
        return set()
    return {str(sym).strip().zfill(6) for sym in symbols if str(sym).strip()}


def _compute_dk_candidate_mask(
    daily_long: pd.DataFrame,
    *,
    index: pd.Index,
    columns: pd.Index,
    date_col: str,
    sym_col: str,
    dk_params: DKTrendParams | None,
    max_run_len: Any = None,
) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=index, columns=columns)
    params = dk_params or DKTrendParams()
    max_run = None if max_run_len is None else max(int(max_run_len), 1)
    for sym, sdf in daily_long.groupby(sym_col, sort=False):
        code = str(sym).zfill(6)
        if code not in mask.columns:
            continue
        trend = compute_dktrend(sdf.sort_values(date_col), params)
        dates = pd.to_datetime(trend[date_col]).dt.normalize()
        candidate = trend["dk_color"].astype(str).eq("red")
        if max_run is not None:
            run_len = pd.to_numeric(trend["dk_run_len"], errors="coerce").fillna(0)
            candidate &= run_len.le(max_run)
        mask.loc[dates, code] = candidate.to_numpy()
    return mask


def _compute_rolling_greylist_mask(
    daily_long: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    lookback: int,
    horizon: int,
    threshold: float,
    min_samples: int,
    date_col: str,
    sym_col: str,
) -> pd.DataFrame:
    """Point-in-time rolling greylist mask based on realized forward returns.

    For each date *t*, only past signals whose *horizon*-day forward return
    is fully observable by *t* are used: a signal on date *s* enters at
    *open[s+1]* and its return is known once *open[s+horizon+1]* exists,
    i.e. at date *s+horizon+1*.  Therefore the training window for date *t*
    ends at *t - horizon - 1*.

    Returns a boolean DataFrame (True = greylisted) aligned with *scores*.
    """
    h = max(int(horizon), 1)
    lb = max(int(lookback), h + 2)
    thresh = float(threshold)
    min_obs = max(int(min_samples), 1)

    df = daily_long.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    open_wide = df.pivot(index=date_col, columns=sym_col, values="open").sort_index().astype(np.float64)
    open_wide = open_wide.reindex(index=scores.index, columns=scores.columns).astype(np.float64)

    score_arr = scores.to_numpy(dtype=np.float64)
    open_arr = open_wide.to_numpy(dtype=np.float64)
    n_dates, n_syms = score_arr.shape

    mask = np.zeros((n_dates, n_syms), dtype=bool)

    for i in range(n_dates):
        train_end = i - h - 1
        train_start = max(0, i - h - lb)
        if train_end < train_start or train_end < 0:
            continue
        n_train = train_end - train_start + 1
        if n_train < min_obs:
            continue

        score_window = score_arr[train_start : train_end + 1, :]
        # Forward return for signal at position j: open[j+h+1] / open[j+1] - 1
        entry = open_arr[train_start + 1 : train_end + 2, :]
        exit_price = open_arr[train_start + h + 1 : train_end + h + 2, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd_ret = np.where(
                (entry > 0) & (exit_price > 0),
                exit_price / entry - 1.0,
                np.nan,
            )

        active = (score_window > 0.0) & ~np.isnan(fwd_ret)
        n_active = active.sum(axis=0)
        with np.errstate(invalid="ignore"):
            sum_ret = np.nansum(np.where(active, fwd_ret, 0.0), axis=0)
            mean_ret = np.where(n_active >= min_obs, sum_ret / n_active, np.nan)

        mask[i, :] = ~np.isnan(mean_ret) & (mean_ret < thresh)

    return pd.DataFrame(mask, index=scores.index, columns=scores.columns)
