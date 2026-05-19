"""Single-stock DK trend backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.performance_panel import annualized_return_cagr, compute_performance_panel
from src.backtest.risk_metrics import risk_off_multiplier_from_index
from src.backtest.transaction_costs import TransactionCostParams, cost_params_dict_for_logging
from src.features.weekly_trend import compute_weekly_trend_state
from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.indicators.adx import compute_adx
from src.indicators.donchian import compute_donchian_trend
from src.market.tradability import (
    is_open_limit_down_unsellable,
    is_open_limit_up_unbuyable,
    is_row_suspended_like,
    is_tradable_open,
    next_buy_index,
)
from src.models.meta_label import FEATURE_COLUMNS, build_signal_features
from src.signals.consensus import compute_consensus_trend
from src.signals.generator import apply_volume_confirmation, compute_signal_quality


@dataclass(frozen=True)
class SingleStockBacktestResult:
    symbol: str
    stock_name: str
    period: str
    n_trades: int
    win_rate: float
    avg_hold_days: float
    avg_return_per_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_return: float
    annualized_return: float
    buy_hold_return: float
    buy_hold_annualized_return: float
    excess_annualized_return: float
    information_ratio: float
    beta_to_benchmark: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    stop_loss_exits: int
    trailing_stop_exits: int
    atr_stop_exits: int = 0
    atr_trailing_exits: int = 0
    profit_lock_exits: int = 0
    market_exit_exits: int = 0
    time_stop_exits: int = 0
    dk_fade_exits: int = 0
    intrapos_dd_exits: int = 0
    avg_position_fraction: float = 1.0
    cost_model: dict | None = None
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _prepare_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    need = {"open", "close", "high", "low"}
    missing = need - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv missing required columns: {sorted(missing)}")
    df = ohlcv.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    else:
        df["trade_date"] = pd.to_datetime(df.index).normalize()
    df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    for col in ["open", "close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("ohlcv must contain at least two valid bars")
    return df


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    atr.name = "atr"
    return atr


def _ewma_volatility(returns: pd.Series, span: int = 20) -> pd.Series:
    """Annualised EWMA volatility, using only historical returns."""
    span = max(int(span), 2)
    vol = pd.to_numeric(returns, errors="coerce").ewm(span=span, min_periods=max(5, span // 2)).std() * np.sqrt(252)
    vol.name = "ewma_volatility"
    return vol


def _next_sell_index(df: pd.DataFrame, start_idx: int, symbol: str = "") -> int | None:
    for j in range(start_idx, len(df)):
        if not is_tradable_open(df, j):
            continue
        prev_close = float(df.loc[j - 1, "close"]) if j > 0 else np.nan
        open_px = float(df.loc[j, "open"])
        if symbol and is_open_limit_down_unsellable(open_px, prev_close, symbol):
            continue
        return j
    return None


def _max_consecutive(values: list[bool], target: bool) -> int:
    best = cur = 0
    for v in values:
        if v is target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _future_exit_index(actions: dict[int, str], start_idx: int) -> int | None:
    exits = [
        idx
        for idx, action in actions.items()
        if idx >= start_idx
        and action
        in {
            "sell",
            "stop_loss",
            "trailing_stop",
            "atr_stop",
            "atr_trailing_stop",
            "profit_lock",
            "market_exit",
            "sector_exit",
            "time_stop",
            "dk_fade_exit",
            "intrapos_dd_stop",
        }
    ]
    return min(exits) if exits else None


def _index_allows_new_position(
    index_ohlcv: pd.DataFrame | None,
    *,
    benchmark_symbol: str,
    asof: pd.Timestamp,
    lookback_days: int,
    drop_threshold: float,
    risk_off_factor: float,
) -> bool:
    if index_ohlcv is None or index_ohlcv.empty:
        return True
    multiplier, _, _ = risk_off_multiplier_from_index(
        index_ohlcv,
        benchmark_symbol=str(benchmark_symbol).zfill(6),
        asof=asof,
        lookback_trading_days=int(lookback_days),
        drop_threshold=float(drop_threshold),
        risk_off_factor=float(risk_off_factor),
    )
    return multiplier > 0.0


def _align_index_macd_hist(index_ohlcv: pd.DataFrame | None, stock_dates: pd.Series) -> np.ndarray | None:
    """Align benchmark MACD histogram values to stock bar dates."""
    if index_ohlcv is None or index_ohlcv.empty or "close" not in index_ohlcv.columns:
        return None
    idx_df = index_ohlcv.copy()
    if "trade_date" in idx_df.columns:
        idx_df["trade_date"] = pd.to_datetime(idx_df["trade_date"]).dt.normalize()
    else:
        idx_df["trade_date"] = pd.to_datetime(idx_df.index).normalize()
    idx_df = idx_df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    idx_close = pd.to_numeric(idx_df["close"], errors="coerce")
    ema_fast = idx_close.ewm(span=12, adjust=False).mean()
    ema_slow = idx_close.ewm(span=26, adjust=False).mean()
    diff = ema_fast - ema_slow
    signal = diff.ewm(span=9, adjust=False).mean()
    hist = diff - signal
    date_to_hist = dict(zip(pd.to_datetime(idx_df["trade_date"]).dt.normalize(), hist))
    aligned = np.full(len(stock_dates), np.nan, dtype=np.float64)
    for j, d in enumerate(pd.to_datetime(stock_dates).dt.normalize()):
        value = date_to_hist.get(d)
        if value is not None and np.isfinite(value):
            aligned[j] = float(value)
    return aligned


def _close_to_returns(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=np.float64, name=name)
    work = df.copy()
    if "trade_date" in work.columns:
        dates = pd.to_datetime(work["trade_date"]).dt.normalize()
    else:
        dates = pd.to_datetime(work.index).normalize()
    close = pd.to_numeric(work["close"], errors="coerce")
    out = pd.Series(close.to_numpy(dtype=np.float64), index=dates, name="close")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.pct_change().fillna(0.0).rename(name)


def _information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: float = 252.0) -> float:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    sd = float(active.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def _beta_to_benchmark(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    strategy = aligned.iloc[:, 0].to_numpy(dtype=np.float64)
    benchmark = aligned.iloc[:, 1].to_numpy(dtype=np.float64)
    var = float(np.var(benchmark, ddof=1))
    if var <= 0 or not np.isfinite(var):
        return float("nan")
    cov = float(np.cov(strategy, benchmark, ddof=1)[0, 1])
    return float(cov / var)


def _extract_signal_features_at(features: pd.DataFrame, idx: int) -> np.ndarray:
    """Return one model-ready feature row for a signal position."""
    if idx < 0 or idx >= len(features):
        raise IndexError(f"signal idx out of range: {idx}")
    row = features.iloc[idx]
    values = np.array([row.get(c, np.nan) for c in FEATURE_COLUMNS], dtype=np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _predict_meta_p_win(meta_model: object, features: pd.DataFrame, idx: int) -> float:
    """Predict p_win from either this project's model or a sklearn-like model."""
    x = _extract_signal_features_at(features, idx).reshape(1, -1)
    raw = meta_model.predict_proba(x)  # type: ignore[attr-defined]
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 2:
        p = arr[0, 1] if arr.shape[1] > 1 else arr[0, 0]
    else:
        p = arr.reshape(-1)[0]
    return float(p) if np.isfinite(p) else float("nan")


def _align_sector_exit_flags(
    sector_index_ohlcv: pd.DataFrame | None,
    stock_dates: pd.Series,
    *,
    drop_threshold: float,
    ma_period: int,
) -> np.ndarray | None:
    """Align sector-index stress flags to stock bars."""
    if sector_index_ohlcv is None or sector_index_ohlcv.empty or "close" not in sector_index_ohlcv.columns:
        return None
    idx_df = sector_index_ohlcv.copy()
    if "trade_date" in idx_df.columns:
        idx_df["trade_date"] = pd.to_datetime(idx_df["trade_date"]).dt.normalize()
    else:
        idx_df["trade_date"] = pd.to_datetime(idx_df.index).normalize()
    idx_df = idx_df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    idx_close = pd.to_numeric(idx_df["close"], errors="coerce")
    lookback = max(int(ma_period), 1)
    idx_drop = idx_close / idx_close.shift(lookback) - 1.0
    idx_ma = idx_close.rolling(lookback, min_periods=lookback).mean()
    sector_stressed = (idx_drop < -abs(float(drop_threshold))) | (idx_close < idx_ma)
    date_to_flag = dict(zip(pd.to_datetime(idx_df["trade_date"]).dt.normalize(), sector_stressed.fillna(False)))
    flags = np.zeros(len(stock_dates), dtype=bool)
    for j, d in enumerate(pd.to_datetime(stock_dates).dt.normalize()):
        flags[j] = bool(date_to_flag.get(d, False))
    return flags


def run_single_stock_backtest(
    symbol: str,
    ohlcv: pd.DataFrame,
    params: DKTrendParams,
    *,
    cost_bps: float = 15.0,
    cost_params: TransactionCostParams | None = None,
    initial_capital: float = 100_000.0,
    stock_name: str = "",
    volume_confirm: bool = False,
    volume_lookback: int = 20,
    volume_ratio_min: float = 1.0,
    consensus_n_agree: int | None = None,
    enable_index_filter: bool = False,
    index_ohlcv: pd.DataFrame | None = None,
    benchmark_symbol: str = "510300",
    extreme_lookback_days: int = 10,
    extreme_drop_threshold: float = 0.05,
    risk_off_factor: float = 0.0,
    stop_loss_pct: float = 0.0,
    trailing_stop_pct: float = 0.0,
    atr_stop_multiplier: float = 0.0,
    atr_stop_period: int = 14,
    atr_trailing_mult: float = 0.0,
    atr_trailing_min_gain: float = 0.0,
    risk_per_trade_pct: float = 0.0,
    position_size_cap: float = 1.0,
    stop_reentry_enabled: bool = False,
    stop_reentry_cooldown: int = 3,
    stop_reentry_min_run: int = 2,
    trend_override: pd.DataFrame | None = None,
    min_quality_score: float = 0.0,
    quality_score_mode: str = "hard",
    quality_score_floor: float = 0.3,
    meta_model: object | None = None,
    meta_label_threshold: float = 0.50,
    meta_label_mode: str = "off",
    require_above_ma120: bool = False,
    require_positive_rs60: bool = False,
    require_index_trend_bullish: bool = False,
    require_weekly_bullish: bool = False,
    weekly_ma_fast: int = 5,
    weekly_ma_slow: int = 13,
    # Phase 4.1 — exit optimisation
    time_stop_days: int = 0,
    time_stop_min_return: float = 0.0,
    profit_lock_trigger: float = 0.0,
    profit_lock_trailing: float = 0.0,
    profit_lock_trigger_hq: float = 0.0,
    profit_lock_trailing_hq: float = 0.0,
    quality_hq_threshold: float = 70.0,
    market_exit_mode: str = "off",
    sector_index_ohlcv: pd.DataFrame | None = None,
    sector_drop_threshold: float = 0.10,
    sector_ma_period: int = 20,
    # Phase 4.2 — volatility-target position sizing
    volatility_target_ann: float = 0.0,
    volatility_lookback: int = 20,
    volatility_high_vol_multiple: float = 1.5,
    volatility_high_vol_scale: float = 0.5,
    # Phase 4.3 — drawdown throttle
    drawdown_throttle_enabled: bool = False,
    # S2 — exit engine redesign
    dk_fade_exit_n: int = 0,
    intrapos_dd_limit: float = 0.0,
    # S3 — entry quality gate
    require_price_breakout: bool = False,
    breakout_lookback: int = 20,
    require_adx_min: float = 0.0,
    adx_period: int = 14,
    require_pullback_entry: bool = False,
    pullback_wait_days: int = 5,
    # S3.4 — index MA20 position scaling
    enable_index_ma20_filter: bool = False,
) -> SingleStockBacktestResult:
    """Backtest one stock with T+1 open execution and single-position long/flat state.

    If *trend_override* is provided it must be a DataFrame with at minimum a
    ``dk_signal`` column aligned to the same length as *ohlcv*; it completely
    replaces the on-the-fly DK-trend computation (useful for permutation tests).
    """
    code = str(symbol).strip().zfill(6)
    df = _prepare_ohlcv(ohlcv)
    if trend_override is not None:
        trend = trend_override.reset_index(drop=True)
    elif consensus_n_agree is not None and int(consensus_n_agree) > 1:
        trend = compute_consensus_trend(
            df,
            base_params=params,
            n_agree=int(consensus_n_agree),
            volume_confirm=volume_confirm,
            volume_lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    elif params.mode == TrendMode.DONCHIAN_BREAKOUT:
        trend = compute_donchian_trend(
            df,
            entry_window=params.donchian_entry_window,
            exit_window=params.donchian_exit_window,
            min_run_len=params.min_run_len,
        ).reset_index(drop=True)
        trend = apply_volume_confirmation(
            trend,
            enabled=volume_confirm,
            lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    else:
        trend = compute_dktrend(df, params).reset_index(drop=True)
        trend = apply_volume_confirmation(
            trend,
            enabled=volume_confirm,
            lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    if cost_params is not None:
        buy_frac = cost_params.buy_fraction()
        sell_frac = cost_params.sell_fraction()
        cost_model_dict = cost_params_dict_for_logging(cost_params)
    else:
        c = max(float(cost_bps), 0.0) * 1e-4
        buy_frac = c
        sell_frac = c
        cost_model_dict = {"type": "symmetric", "cost_bps": float(cost_bps)}

    # ── Trade attribution pre-computation ──
    _attr_quality = compute_signal_quality(trend).reset_index(drop=True)
    _attr_atr14 = _compute_atr(df, 14)
    _attr_close = pd.to_numeric(df["close"], errors="coerce")
    _attr_atr_pct = _attr_atr14 / _attr_close
    _attr_vol = pd.to_numeric(df["volume"], errors="coerce")
    _attr_avg_vol = _attr_vol.rolling(window=volume_lookback, min_periods=volume_lookback).mean()
    _attr_vol_ratio = _attr_vol / _attr_avg_vol
    _attr_stock_ret60 = _attr_close.pct_change(60)
    _attr_ma120 = _attr_close.rolling(120, min_periods=120).mean()
    _attr_weekly_trend = (
        compute_weekly_trend_state(df, ma_windows=(int(weekly_ma_fast), int(weekly_ma_slow))).reset_index(drop=True)
        if bool(require_weekly_bullish)
        else pd.Series("neutral", index=df.index, dtype=object)
    )
    # S3 — ADX and price breakout pre-computation
    _s3_adx_enabled = float(require_adx_min) > 0.0
    _s3_adx_series = None
    if _s3_adx_enabled:
        _s3_adx_df = compute_adx(df, period=int(adx_period))
        _s3_adx_series = pd.to_numeric(_s3_adx_df["adx"], errors="coerce").to_numpy(dtype=np.float64)
    _s3_breakout_enabled = bool(require_price_breakout)
    _s3_breakout_n = max(int(breakout_lookback), 1)
    _s3_breakout_high = _attr_close.rolling(_s3_breakout_n, min_periods=_s3_breakout_n).max().shift(1) if _s3_breakout_enabled else None
    _attr_regime = pd.Series("unknown", index=df.index, dtype=object)
    _attr_rs_60 = pd.Series(np.nan, index=df.index, dtype=np.float64)
    _attr_idx_ret60 = pd.Series(np.nan, index=df.index, dtype=np.float64)
    if index_ohlcv is not None and not index_ohlcv.empty:
        idx_dates = pd.to_datetime(index_ohlcv["trade_date"]).dt.normalize()
        idx_close = pd.to_numeric(index_ohlcv["close"], errors="coerce")
        idx_ret = idx_close.pct_change(60)
        idx_map = dict(zip(idx_dates, idx_ret))
        stock_dates = pd.to_datetime(df["trade_date"]).dt.normalize()
        for _j, _d in enumerate(stock_dates):
            _v = idx_map.get(_d)
            if _v is not None and np.isfinite(_v):
                _attr_idx_ret60.iloc[_j] = _v
        _attr_regime = _attr_idx_ret60.apply(
            lambda x: "bull" if x > 0.10 else ("bear" if x < -0.10 else "ranging") if pd.notna(x) else "unknown"
        )
        _attr_rs_60 = _attr_stock_ret60 - _attr_idx_ret60
    else:
        _attr_rs_60 = _attr_stock_ret60

    _index_trend_bullish_enabled = bool(require_index_trend_bullish)
    _index_macd_hist = (
        _align_index_macd_hist(index_ohlcv, df["trade_date"])
        if _index_trend_bullish_enabled
        else None
    )

    # S3.4 — index MA20 for position scaling (half-position when index below MA20)
    _s3_idx_ma20_enabled = bool(enable_index_ma20_filter)
    _s3_idx_ma20_aligned = None
    if _s3_idx_ma20_enabled and index_ohlcv is not None and not index_ohlcv.empty:
        _idx_close_for_ma20 = pd.to_numeric(index_ohlcv["close"], errors="coerce")
        _idx_ma20 = _idx_close_for_ma20.rolling(20, min_periods=20).mean()
        _idx_dates_for_ma20 = pd.to_datetime(index_ohlcv["trade_date"]).dt.normalize()
        _idx_ma20_map = dict(zip(_idx_dates_for_ma20, _idx_ma20))
        _idx_close_map = dict(zip(_idx_dates_for_ma20, _idx_close_for_ma20))
        _stock_dates_for_ma20 = pd.to_datetime(df["trade_date"]).dt.normalize()
        _s3_idx_ma20_aligned = np.full(len(df), np.nan, dtype=np.float64)
        for _j, _d in enumerate(_stock_dates_for_ma20):
            _ma20_v = _idx_ma20_map.get(_d)
            _close_v = _idx_close_map.get(_d)
            if _ma20_v is not None and _close_v is not None and np.isfinite(_ma20_v) and np.isfinite(_close_v) and _ma20_v > 0:
                _s3_idx_ma20_aligned[_j] = float(_close_v < _ma20_v)  # 1.0 if below MA20, 0.0 if above

    meta_mode = str(meta_label_mode).lower()
    meta_threshold = float(meta_label_threshold)
    _meta_features = (
        build_signal_features(df, index_ohlcv=index_ohlcv).reset_index(drop=True)
        if meta_model is not None and meta_mode != "off"
        else None
    )

    actions: dict[int, str] = {}
    entry_meta: dict[int, dict] = {}
    planned_long = False
    pending_buy_idx: int | None = None
    for i, row in trend.iterrows():
        if pending_buy_idx is not None and pending_buy_idx <= i:
            planned_long = True
            pending_buy_idx = None
        sig = str(row.get("dk_signal", ""))
        if sig == "buy" and not planned_long and pending_buy_idx is None and i + 1 < len(df):
            if enable_index_filter and not _index_allows_new_position(
                index_ohlcv,
                benchmark_symbol=benchmark_symbol,
                asof=pd.Timestamp(df.loc[i, "trade_date"]),
                lookback_days=extreme_lookback_days,
                drop_threshold=extreme_drop_threshold,
                risk_off_factor=risk_off_factor,
            ):
                continue
            _q = float(_attr_quality.iloc[i]) if i < len(_attr_quality) else 0.0
            if min_quality_score > 0 and str(quality_score_mode).lower() in {"hard", "scale"} and _q < min_quality_score:
                continue
            if bool(require_above_ma120):
                _ma120 = float(_attr_ma120.iloc[i]) if i < len(_attr_ma120) else float("nan")
                _close_i = float(_attr_close.iloc[i]) if i < len(_attr_close) else float("nan")
                if not (np.isfinite(_ma120) and np.isfinite(_close_i) and _close_i > _ma120):
                    continue
            if bool(require_positive_rs60):
                _rs60 = float(_attr_rs_60.iloc[i]) if i < len(_attr_rs_60) else float("nan")
                if not (np.isfinite(_rs60) and _rs60 > 0):
                    continue
            if _index_trend_bullish_enabled:
                _idx_hist = (
                    float(_index_macd_hist[i])
                    if _index_macd_hist is not None and i < len(_index_macd_hist)
                    else float("nan")
                )
                if not (np.isfinite(_idx_hist) and _idx_hist > 0):
                    continue
            if bool(require_weekly_bullish):
                _weekly_state = str(_attr_weekly_trend.iloc[i]) if i < len(_attr_weekly_trend) else "neutral"
                if _weekly_state != "bullish":
                    continue
            # S3.1 — price breakout confirmation
            if _s3_breakout_enabled and _s3_breakout_high is not None:
                _breakout_high = float(_s3_breakout_high.iloc[i]) if i < len(_s3_breakout_high) else float("nan")
                _close_i_s3 = float(_attr_close.iloc[i]) if i < len(_attr_close) else float("nan")
                if np.isfinite(_breakout_high) and np.isfinite(_close_i_s3) and _close_i_s3 <= _breakout_high:
                    continue
            # S3.2 — ADX trend strength filter
            if _s3_adx_enabled and _s3_adx_series is not None:
                _adx_val = float(_s3_adx_series[i]) if i < len(_s3_adx_series) else 0.0
                if not np.isfinite(_adx_val) or _adx_val < float(require_adx_min):
                    continue
            # S3.3 — pullback entry: wait for a down day after signal before buying
            _pullback_enabled = bool(require_pullback_entry)
            _pullback_wait = max(int(pullback_wait_days), 1) if _pullback_enabled else 0
            _pullback_offset = 0
            if _pullback_enabled:
                _found_pullback = False
                for _pb_day in range(1, _pullback_wait + 1):
                    _pb_idx = i + _pb_day
                    if _pb_idx >= len(df):
                        break
                    _pb_close = float(_attr_close.iloc[_pb_idx]) if _pb_idx < len(_attr_close) else float("nan")
                    _pb_prev_close = float(_attr_close.iloc[_pb_idx - 1]) if (_pb_idx - 1) < len(_attr_close) else float("nan")
                    if np.isfinite(_pb_close) and np.isfinite(_pb_prev_close) and _pb_close < _pb_prev_close:
                        _pullback_offset = _pb_day
                        _found_pullback = True
                        break
                if not _found_pullback:
                    _pullback_offset = _pullback_wait
            _meta_p_win = float("nan")
            if meta_model is not None and _meta_features is not None and meta_mode in {"hard", "scale"}:
                _meta_p_win = _predict_meta_p_win(meta_model, _meta_features, i)
                if meta_mode == "hard" and (not np.isfinite(_meta_p_win) or _meta_p_win < meta_threshold):
                    continue
            j = next_buy_index(df, code, i + 1 + _pullback_offset)
            if j is None:
                continue
            actions.setdefault(j, "buy")
            pending_buy_idx = j
            entry_meta[j] = {
                "quality": float(_attr_quality.iloc[i]) if i < len(_attr_quality) else 0.0,
                "vol_ratio": float(_attr_vol_ratio.iloc[i]) if i < len(_attr_vol_ratio) and np.isfinite(_attr_vol_ratio.iloc[i]) else float("nan"),
                "atr_pct": float(_attr_atr_pct.iloc[i]) if i < len(_attr_atr_pct) and np.isfinite(_attr_atr_pct.iloc[i]) else float("nan"),
                "regime": str(_attr_regime.iloc[i]) if i < len(_attr_regime) else "unknown",
                "rs_60": float(_attr_rs_60.iloc[i]) if i < len(_attr_rs_60) and np.isfinite(_attr_rs_60.iloc[i]) else float("nan"),
                "meta_p_win": _meta_p_win,
            }
        elif sig == "sell" and planned_long and i + 1 < len(df):
            j = _next_sell_index(df, i + 1, code)
            if j is None:
                continue
            actions.setdefault(j, "sell")
            planned_long = False
        elif sig == "sell" and pending_buy_idx is not None:
            actions.pop(pending_buy_idx, None)
            pending_buy_idx = None

    stop_loss = max(float(stop_loss_pct), 0.0)
    trailing_stop = max(float(trailing_stop_pct), 0.0)
    atr_stop_mult = max(float(atr_stop_multiplier), 0.0)
    atr_period = max(int(atr_stop_period), 1)
    atr_trailing_mult_val = max(float(atr_trailing_mult), 0.0)
    atr_trailing_min_gain_val = max(float(atr_trailing_min_gain), 0.0)
    atr_series = _compute_atr(df, atr_period) if max(atr_stop_mult, atr_trailing_mult_val) > 0 else pd.Series(dtype=float)
    risk_per_trade = max(float(risk_per_trade_pct), 0.0)
    pos_cap = max(min(float(position_size_cap), 1.0), 0.0)
    use_risk_sizing = risk_per_trade > 0 and (stop_loss > 0 or atr_stop_mult > 0)
    cash = float(initial_capital)
    total_equity = cash
    shares = 0.0
    in_pos = False
    entry_date = pd.NaT
    entry_price = float("nan")
    entry_cash = float("nan")
    highest_close = float("nan")
    atr_stop_price = float("nan")
    trades: list[dict] = []
    equity = np.zeros(len(df), dtype=np.float64)
    position_fractions: list[float] = []
    reentry_cooldown = max(int(stop_reentry_cooldown), 0)
    reentry_min_run = max(int(stop_reentry_min_run), 1)
    reentry_enabled = bool(stop_reentry_enabled)
    cooldown_remaining = 0

    # Phase 4 — exit optimisation & position management state
    time_stop_enabled = max(int(time_stop_days), 0) > 0
    profit_lock_enabled = (
        max(float(profit_lock_trigger), 0.0) > 0 and max(float(profit_lock_trailing), 0.0) > 0
    ) or (
        max(float(profit_lock_trigger_hq), 0.0) > 0 and max(float(profit_lock_trailing_hq), 0.0) > 0
    )
    profit_lock_active = False
    profit_lock_high = float("nan")
    market_exit_mode_norm = str(market_exit_mode).lower()
    market_exit = market_exit_mode_norm in {"exit", "reduce", "on"}
    sector_exit = market_exit_mode_norm == "sector"
    vol_target_enabled = max(float(volatility_target_ann), 0.0) > 0
    vol_lookback = max(int(volatility_lookback), 5)
    close_returns = pd.to_numeric(df["close"], errors="coerce").pct_change()
    ewma_vol = _ewma_volatility(close_returns, span=vol_lookback) if vol_target_enabled else pd.Series(dtype=float)
    ewma_vol_median = ewma_vol.expanding(min_periods=max(10, vol_lookback)).median() if vol_target_enabled else pd.Series(dtype=float)
    high_vol_multiple = max(float(volatility_high_vol_multiple), 0.0)
    high_vol_scale = max(min(float(volatility_high_vol_scale), 1.0), 0.0)
    dd_throttle = bool(drawdown_throttle_enabled)
    peak_equity = float(initial_capital)
    # S2 — exit engine state
    dk_fade_enabled = int(dk_fade_exit_n) > 0
    dk_fade_n = max(int(dk_fade_exit_n), 1)
    intrapos_dd_enabled = float(intrapos_dd_limit) > 0.0
    intrapos_dd_limit_val = max(float(intrapos_dd_limit), 0.0)
    # Precompute dk_value series for fade exit
    _dk_value_series = pd.to_numeric(trend["dk_value"], errors="coerce").to_numpy(dtype=np.float64) if "dk_value" in trend.columns else np.full(len(df), np.nan)
    # Precompute index MA60 for market-exit signals
    _idx_ma60: pd.Series | None = None
    _idx_ma60_below: np.ndarray | None = None
    _idx_drop20: np.ndarray | None = None
    if market_exit and index_ohlcv is not None and not index_ohlcv.empty:
        idx_df = index_ohlcv.copy()
        if "trade_date" in idx_df.columns:
            idx_df["trade_date"] = pd.to_datetime(idx_df["trade_date"]).dt.normalize()
        idx_df = idx_df.sort_values("trade_date").reset_index(drop=True)
        idx_close = pd.to_numeric(idx_df["close"], errors="coerce")
        _idx_ma60 = idx_close.rolling(60, min_periods=60).mean()
        _idx_ma60_below_arr = np.zeros(len(df), dtype=bool)
        _idx_drop20_arr = np.zeros(len(df), dtype=bool)
        idx_dates = pd.to_datetime(idx_df["trade_date"]).dt.normalize()
        idx_date_to_i = {d: j for j, d in enumerate(idx_dates)}
        for _i in range(len(df)):
            _d = pd.Timestamp(df.loc[_i, "trade_date"])
            _j = idx_date_to_i.get(_d)
            if _j is not None and _j < len(idx_close) and _j >= 60:
                _idx_ma60_below_arr[_i] = bool(idx_close.iloc[_j] < _idx_ma60.iloc[_j])
                if _j >= 20:
                    _drop = idx_close.iloc[_j] / idx_close.iloc[_j - 20] - 1.0
                    _idx_drop20_arr[_i] = bool(_drop < -0.08)
        _idx_ma60_below = _idx_ma60_below_arr
        _idx_drop20 = _idx_drop20_arr
    _sector_exit_flags = _align_sector_exit_flags(
        sector_index_ohlcv,
        df["trade_date"],
        drop_threshold=float(sector_drop_threshold),
        ma_period=int(sector_ma_period),
    ) if sector_exit else None

    for i in range(len(df)):
        action = actions.get(i)
        if action == "buy" and not in_pos:
            price = float(df.loc[i, "open"])
            if price > 0 and np.isfinite(price):
                total_equity = cash  # cash is all equity when flat
                position_value = total_equity * pos_cap * (1.0 - buy_frac)
                if use_risk_sizing:
                    stop_dist_pct = stop_loss
                    if atr_stop_mult > 0 and not atr_series.empty:
                        atr_val = float(atr_series.iloc[i])
                        if np.isfinite(atr_val) and atr_val > 0:
                            atr_stop_price = price - atr_stop_mult * atr_val
                            atr_dist_pct = (price - atr_stop_price) / price
                            if np.isfinite(atr_dist_pct) and atr_dist_pct > 0:
                                stop_dist_pct = max(stop_dist_pct, atr_dist_pct) if stop_loss > 0 else atr_dist_pct
                    if stop_dist_pct > 0:
                        max_risk_amount = total_equity * risk_per_trade
                        risk_capped_value = max_risk_amount / stop_dist_pct
                        position_value = min(position_value, risk_capped_value * (1.0 - buy_frac))
                # Phase 14 — EWMA volatility-target scaling and high-volatility haircut
                if vol_target_enabled and i < len(ewma_vol):
                    _real_vol = float(ewma_vol.iloc[i])
                    if np.isfinite(_real_vol) and _real_vol > 0:
                        _vol_scale = min(1.0, float(volatility_target_ann) / _real_vol)
                        _median_vol = float(ewma_vol_median.iloc[i]) if i < len(ewma_vol_median) else float("nan")
                        if (
                            high_vol_multiple > 0
                            and high_vol_scale < 1.0
                            and np.isfinite(_median_vol)
                            and _median_vol > 0
                            and _real_vol > high_vol_multiple * _median_vol
                        ):
                            _vol_scale = min(_vol_scale, high_vol_scale)
                        position_value *= _vol_scale
                # Phase 4.3 — drawdown throttle
                if dd_throttle:
                    _dd = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0
                    if _dd < 0.05:
                        _dd_mult = 1.0
                    elif _dd < 0.10:
                        _dd_mult = 0.7
                    elif _dd < 0.15:
                        _dd_mult = 0.5
                    else:
                        _dd_mult = 0.3  # floor to avoid zero-size positions
                    position_value *= _dd_mult
                # Phase 2.1 — quality-score position scaling
                if str(quality_score_mode).lower() == "scale" and i in entry_meta:
                    _q_for_scale = float(entry_meta[i].get("quality", 50.0))
                    if np.isfinite(_q_for_scale) and _q_for_scale > 0:
                        _q_mult = max(float(quality_score_floor), _q_for_scale / 100.0)
                        position_value *= _q_mult
                # Phase 8 — meta-label position scaling
                if meta_model is not None and meta_mode == "scale" and i in entry_meta:
                    _p_win_for_scale = float(entry_meta[i].get("meta_p_win", float("nan")))
                    if np.isfinite(_p_win_for_scale):
                        _meta_mult = min(1.0, max(0.3, (_p_win_for_scale - 0.40) / 0.40))
                        position_value *= _meta_mult
                # S3.4 — index MA20 half-position filter
                if _s3_idx_ma20_enabled and _s3_idx_ma20_aligned is not None:
                    _idx_below_ma20 = float(_s3_idx_ma20_aligned[i]) if i < len(_s3_idx_ma20_aligned) else 0.0
                    if np.isfinite(_idx_below_ma20) and _idx_below_ma20 > 0.5:
                        position_value *= 0.5
                shares = position_value / price
                entry_cash = total_equity
                cash = total_equity - position_value
                position_fractions.append(position_value / total_equity if total_equity > 0 else 1.0)
                in_pos = True
                entry_date = pd.Timestamp(df.loc[i, "trade_date"])
                entry_price = price
                highest_close = float(df.loc[i, "close"])
                _meta = entry_meta.get(i, {})
                entry_quality = float(_meta.get("quality", 0.0))
                entry_vol_ratio = float(_meta.get("vol_ratio", float("nan")))
                entry_atr_pct_val = float(_meta.get("atr_pct", float("nan")))
                entry_regime = str(_meta.get("regime", "unknown"))
                entry_rs_60_val = float(_meta.get("rs_60", float("nan")))
                entry_meta_p_win = float(_meta.get("meta_p_win", float("nan")))
                entry_profit_lock_trigger = max(float(profit_lock_trigger), 0.0)
                entry_profit_lock_trailing = max(float(profit_lock_trailing), 0.0)
                if (
                    max(float(profit_lock_trigger_hq), 0.0) > 0
                    and entry_quality >= float(quality_hq_threshold)
                ):
                    entry_profit_lock_trigger = max(float(profit_lock_trigger_hq), 0.0)
                    entry_profit_lock_trailing = max(
                        float(profit_lock_trailing_hq),
                        entry_profit_lock_trailing,
                    )
                profit_lock_active = False
                profit_lock_high = float("nan")
                mae = 0.0
                mfe = 0.0
                if atr_stop_mult > 0 and not atr_series.empty:
                    atr_val2 = float(atr_series.iloc[i])
                    if np.isfinite(atr_val2) and atr_val2 > 0:
                        atr_stop_price = price - atr_stop_mult * atr_val2
                    else:
                        atr_stop_price = float("nan")
                elif atr_stop_mult <= 0:
                    atr_stop_price = float("nan")
        elif action in {
            "sell",
            "stop_loss",
            "trailing_stop",
            "atr_stop",
            "atr_trailing_stop",
            "profit_lock",
            "market_exit",
            "sector_exit",
            "time_stop",
            "dk_fade_exit",
            "intrapos_dd_stop",
        } and in_pos:
            price = float(df.loc[i, "open"])
            if price <= 0 or not np.isfinite(price):
                continue
            cash += shares * price * (1.0 - sell_frac)
            shares = 0.0
            exit_date = pd.Timestamp(df.loc[i, "trade_date"])
            ret = cash / entry_cash - 1.0
            trades.append(
                {
                    "buy_date": entry_date,
                    "sell_date": exit_date,
                    "buy_price": entry_price,
                    "sell_price": price,
                    "hold_days": int((exit_date - entry_date).days),
                    "return": ret,
                    "exit_reason": action if action != "sell" else "signal",
                    "entry_quality_score": entry_quality,
                    "entry_volume_ratio": entry_vol_ratio,
                    "entry_atr_pct": entry_atr_pct_val,
                    "entry_market_regime": entry_regime,
                    "entry_rs_60": entry_rs_60_val,
                    "entry_meta_p_win": entry_meta_p_win,
                    "mae": mae,
                    "mfe": mfe,
                }
            )
            in_pos = False
            highest_close = float("nan")
            if reentry_enabled and action in {
                "stop_loss",
                "trailing_stop",
                "atr_stop",
                "atr_trailing_stop",
                "profit_lock",
                "market_exit",
                "time_stop",
                "dk_fade_exit",
                "intrapos_dd_stop",
            }:
                cooldown_remaining = reentry_cooldown
        equity[i] = shares * float(df.loc[i, "close"]) if in_pos else cash
        peak_equity = max(peak_equity, equity[i])
        if in_pos:
            close_px = float(df.loc[i, "close"])
            if np.isfinite(close_px) and np.isfinite(entry_price) and entry_price > 0:
                pnl = close_px / entry_price - 1.0
                mae = min(mae, pnl)
                mfe = max(mfe, pnl)
            if np.isfinite(close_px):
                highest_close = close_px if not np.isfinite(highest_close) else max(highest_close, close_px)
            forced_reason = ""
            if stop_loss > 0 and np.isfinite(close_px) and np.isfinite(entry_price) and close_px / entry_price - 1.0 < -stop_loss:
                forced_reason = "stop_loss"
            elif atr_stop_mult > 0 and np.isfinite(close_px) and np.isfinite(atr_stop_price) and close_px < atr_stop_price:
                forced_reason = "atr_stop"
            elif trailing_stop > 0 and np.isfinite(close_px) and np.isfinite(highest_close) and close_px / highest_close - 1.0 < -trailing_stop:
                forced_reason = "trailing_stop"
            elif (
                atr_trailing_mult_val > 0
                and np.isfinite(close_px)
                and np.isfinite(highest_close)
                and np.isfinite(entry_price)
                and entry_price > 0
                and close_px / entry_price - 1.0 >= atr_trailing_min_gain_val
                and i < len(atr_series)
            ):
                atr_now = float(atr_series.iloc[i])
                if np.isfinite(atr_now) and atr_now > 0 and close_px < highest_close - atr_trailing_mult_val * atr_now:
                    forced_reason = "atr_trailing_stop"
            # S2.1 — DK momentum fade exit: dk_value declining N consecutive days
            if not forced_reason and dk_fade_enabled and i >= dk_fade_n:
                dk_window = _dk_value_series[i - dk_fade_n + 1:i + 1]
                if len(dk_window) >= dk_fade_n and all(np.isfinite(dk_window)):
                    if all(dk_window[j] > dk_window[j + 1] for j in range(len(dk_window) - 1)):
                        forced_reason = "dk_fade_exit"
            # S2.2 — Intra-position drawdown stop from highest close
            if not forced_reason and intrapos_dd_enabled and np.isfinite(highest_close) and highest_close > 0:
                dd_from_peak = (highest_close - close_px) / highest_close
                if dd_from_peak > intrapos_dd_limit_val:
                    forced_reason = "intrapos_dd_stop"
            # Phase 4.1 — profit lock: activate tighter trailing after trigger
            if (
                not forced_reason
                and profit_lock_enabled
                and entry_profit_lock_trigger > 0
                and entry_profit_lock_trailing > 0
                and np.isfinite(close_px)
                and np.isfinite(entry_price)
            ):
                if not profit_lock_active and close_px / entry_price - 1.0 >= entry_profit_lock_trigger:
                    profit_lock_active = True
                    profit_lock_high = close_px
                if profit_lock_active:
                    profit_lock_high = max(profit_lock_high, close_px)
                    if close_px / profit_lock_high - 1.0 < -entry_profit_lock_trailing:
                        forced_reason = "profit_lock"
            # Phase 4.1 — market exit: index below MA60 or sharp drop
            if not forced_reason and market_exit and _idx_ma60_below is not None and i < len(_idx_ma60_below):
                if _idx_ma60_below[i] or (_idx_drop20 is not None and i < len(_idx_drop20) and _idx_drop20[i]):
                    forced_reason = "market_exit"
            if not forced_reason and sector_exit and _sector_exit_flags is not None and i < len(_sector_exit_flags):
                if bool(_sector_exit_flags[i]):
                    forced_reason = "sector_exit"
            # Phase 4.1 — time stop: exit after N days if return below threshold
            if not forced_reason and time_stop_enabled and np.isfinite(close_px) and np.isfinite(entry_price):
                _hold = (pd.Timestamp(df.loc[i, "trade_date"]) - entry_date).days
                if _hold >= time_stop_days and close_px / entry_price - 1.0 < time_stop_min_return:
                    forced_reason = "time_stop"
            if forced_reason and i + 1 < len(df):
                j = _next_sell_index(df, i + 1, code)
                existing = _future_exit_index(actions, i + 1)
                if j is not None and (existing is None or j < existing):
                    actions[j] = forced_reason

        if reentry_enabled and not in_pos and cooldown_remaining == 0 and i + 1 < len(df):
            color_now = str(trend.loc[i, "dk_color"]) if i < len(trend) else ""
            run_now = int(trend.loc[i, "dk_run_len"]) if i < len(trend) else 0
            if color_now == "red" and run_now >= reentry_min_run:
                if not is_tradable_open(df, i + 1):
                    pass
                elif enable_index_filter and not _index_allows_new_position(
                    index_ohlcv,
                    benchmark_symbol=benchmark_symbol,
                    asof=pd.Timestamp(df.loc[i, "trade_date"]),
                    lookback_days=extreme_lookback_days,
                    drop_threshold=extreme_drop_threshold,
                    risk_off_factor=risk_off_factor,
                ):
                    pass
                else:
                    j = next_buy_index(df, code, i + 1)
                    if j is not None:
                        actions[j] = "buy"
                        planned_long = True

        if cooldown_remaining > 0 and not in_pos:
            cooldown_remaining -= 1

    last_idx = len(df) - 1
    if in_pos:
        price = float(df.loc[last_idx, "close"])
        cash += shares * price * (1.0 - sell_frac)
        shares = 0.0
        exit_date = pd.Timestamp(df.loc[last_idx, "trade_date"])
        ret = cash / entry_cash - 1.0
        trades.append(
            {
                "buy_date": entry_date,
                "sell_date": exit_date,
                "buy_price": entry_price,
                "sell_price": price,
                "hold_days": int((exit_date - entry_date).days),
                "return": ret,
                "exit_reason": "end",
                "entry_quality_score": entry_quality,
                "entry_volume_ratio": entry_vol_ratio,
                "entry_atr_pct": entry_atr_pct_val,
                "entry_market_regime": entry_regime,
                "entry_rs_60": entry_rs_60_val,
                "entry_meta_p_win": entry_meta_p_win,
                "mae": mae,
                "mfe": mfe,
            }
        )
        in_pos = False
        highest_close = float("nan")
    equity[last_idx] = cash
    for k in range(1, len(equity)):
        if equity[k] == 0:
            equity[k] = equity[k - 1]
    daily_returns = pd.Series(equity, index=pd.to_datetime(df["trade_date"]), name="equity").pct_change().fillna(0.0)
    daily_returns.name = "strategy_ret"
    panel = compute_performance_panel(daily_returns.to_numpy(dtype=np.float64))
    trade_log = pd.DataFrame(trades)
    trade_returns = trade_log["return"].to_numpy(dtype=np.float64) if not trade_log.empty else np.array([])
    wins = [bool(x > 0) for x in trade_returns]
    stop_loss_exits = int((trade_log["exit_reason"] == "stop_loss").sum()) if not trade_log.empty else 0
    trailing_stop_exits = int((trade_log["exit_reason"] == "trailing_stop").sum()) if not trade_log.empty else 0
    atr_stop_exits = int((trade_log["exit_reason"] == "atr_stop").sum()) if not trade_log.empty else 0
    atr_trailing_exits = int((trade_log["exit_reason"] == "atr_trailing_stop").sum()) if not trade_log.empty else 0
    profit_lock_exits = int((trade_log["exit_reason"] == "profit_lock").sum()) if not trade_log.empty else 0
    market_exit_exits = int((trade_log["exit_reason"] == "market_exit").sum()) if not trade_log.empty else 0
    if not trade_log.empty:
        market_exit_exits += int((trade_log["exit_reason"] == "sector_exit").sum())
    time_stop_exits = int((trade_log["exit_reason"] == "time_stop").sum()) if not trade_log.empty else 0
    dk_fade_exits = int((trade_log["exit_reason"] == "dk_fade_exit").sum()) if not trade_log.empty else 0
    intrapos_dd_exits = int((trade_log["exit_reason"] == "intrapos_dd_stop").sum()) if not trade_log.empty else 0
    avg_position_fraction = float(np.mean(position_fractions)) if position_fractions else 1.0
    buy_hold_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)
    buy_hold_returns = _close_to_returns(df, "buy_hold_ret")
    buy_hold_annualized_return = annualized_return_cagr(buy_hold_returns.to_numpy(dtype=np.float64))
    excess_annualized_return = (
        float(panel.annualized_return - buy_hold_annualized_return)
        if np.isfinite(panel.annualized_return) and np.isfinite(buy_hold_annualized_return)
        else float("nan")
    )
    information_ratio = _information_ratio(daily_returns, buy_hold_returns)
    benchmark_returns = _close_to_returns(index_ohlcv, "benchmark_ret") if index_ohlcv is not None else pd.Series(dtype=np.float64, name="benchmark_ret")
    beta_to_benchmark = _beta_to_benchmark(daily_returns, benchmark_returns)
    start_s = pd.Timestamp(df["trade_date"].iloc[0]).date().isoformat()
    end_s = pd.Timestamp(df["trade_date"].iloc[-1]).date().isoformat()
    return SingleStockBacktestResult(
        symbol=code,
        stock_name=stock_name or code,
        period=f"{start_s} ~ {end_s}",
        n_trades=int(len(trade_log)),
        win_rate=float(np.mean(trade_returns > 0)) if trade_returns.size else float("nan"),
        avg_hold_days=float(trade_log["hold_days"].mean()) if not trade_log.empty else float("nan"),
        avg_return_per_trade=float(np.mean(trade_returns)) if trade_returns.size else float("nan"),
        max_consecutive_wins=_max_consecutive(wins, True),
        max_consecutive_losses=_max_consecutive(wins, False),
        total_return=panel.total_return,
        annualized_return=panel.annualized_return,
        buy_hold_return=buy_hold_return,
        buy_hold_annualized_return=buy_hold_annualized_return,
        excess_annualized_return=excess_annualized_return,
        information_ratio=information_ratio,
        beta_to_benchmark=beta_to_benchmark,
        sharpe_ratio=panel.sharpe_ratio,
        max_drawdown=panel.max_drawdown,
        calmar_ratio=panel.calmar_ratio,
        stop_loss_exits=stop_loss_exits,
        trailing_stop_exits=trailing_stop_exits,
        atr_stop_exits=atr_stop_exits,
        atr_trailing_exits=atr_trailing_exits,
        profit_lock_exits=profit_lock_exits,
        market_exit_exits=market_exit_exits,
        time_stop_exits=time_stop_exits,
        dk_fade_exits=dk_fade_exits,
        intrapos_dd_exits=intrapos_dd_exits,
        avg_position_fraction=avg_position_fraction,
        cost_model=cost_model_dict,
        trade_log=trade_log,
        daily_returns=daily_returns,
    )
