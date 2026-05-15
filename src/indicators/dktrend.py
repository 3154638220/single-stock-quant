"""Approximate Eastmoney-style long/short trend indicator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .utils import ema, ema_seeded


class TrendMode(str, Enum):
    MACD_CROSS = "macd_cross"
    MA_CROSS = "ma_cross"
    BOLL_TREND = "boll_trend"
    DONCHIAN_BREAKOUT = "donchian_breakout"
    LONG_MA_TREND = "long_ma_trend"
    DUAL_MA_CROSS = "dual_ma_cross"
    TREND_SCORE = "trend_score"


@dataclass(frozen=True)
class DKTrendParams:
    mode: TrendMode | str = TrendMode.MACD_CROSS
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ma_fast: int = 5
    ma_slow: int = 20
    ma_smooth: int = 3
    boll_window: int = 20
    min_run_len: int = 1
    donchian_entry_window: int = 20
    donchian_exit_window: int = 10
    # Long MA trend params
    trend_ma_period: int = 250
    trend_ma_type: str = "ema"
    slope_lookback: int = 10
    require_positive_slope: bool = True
    # Dual MA cross params
    dual_ma_fast: int = 30
    dual_ma_slow: int = 120
    # Trend score params
    trend_score_ma_long: int = 250
    trend_score_ma_fast: int = 30
    trend_score_ma_slow: int = 120
    # Signal confirmation
    min_breakout_days: int = 3

    @classmethod
    def from_mapping(cls, data: dict | None) -> "DKTrendParams":
        d = dict(data or {})
        raw_mode = d.get("mode", TrendMode.MACD_CROSS.value)
        return cls(
            mode=raw_mode if isinstance(raw_mode, TrendMode) else TrendMode(str(raw_mode)),
            macd_fast=int(d.get("macd_fast", 12)),
            macd_slow=int(d.get("macd_slow", 26)),
            macd_signal=int(d.get("macd_signal", 9)),
            ma_fast=int(d.get("ma_fast", 5)),
            ma_slow=int(d.get("ma_slow", 20)),
            ma_smooth=int(d.get("ma_smooth", 3)),
            boll_window=int(d.get("boll_window", 20)),
            min_run_len=int(d.get("min_run_len", 1)),
            donchian_entry_window=int(d.get("donchian_entry_window", 20)),
            donchian_exit_window=int(d.get("donchian_exit_window", 10)),
            trend_ma_period=int(d.get("trend_ma_period", 250)),
            trend_ma_type=str(d.get("trend_ma_type", "ema")),
            slope_lookback=int(d.get("slope_lookback", 10)),
            require_positive_slope=bool(d.get("require_positive_slope", True)),
            dual_ma_fast=int(d.get("dual_ma_fast", 30)),
            dual_ma_slow=int(d.get("dual_ma_slow", 120)),
            trend_score_ma_long=int(d.get("trend_score_ma_long", 250)),
            trend_score_ma_fast=int(d.get("trend_score_ma_fast", 30)),
            trend_score_ma_slow=int(d.get("trend_score_ma_slow", 120)),
            min_breakout_days=int(d.get("min_breakout_days", 3)),
        )


def _trend_mode(value: TrendMode | str) -> TrendMode:
    if isinstance(value, TrendMode):
        return value
    return TrendMode(str(value))


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("ohlcv is empty")
    missing = {"close"} - set(df.columns)
    if missing:
        raise ValueError(f"ohlcv missing required columns: {sorted(missing)}")
    out = df.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
        out = out.sort_values("trade_date").set_index("trade_date", drop=False)
    else:
        out.index = pd.to_datetime(out.index).normalize()
        out = out.sort_index()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out


def _indicator_value(df: pd.DataFrame, params: DKTrendParams) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce")
    mode = _trend_mode(params.mode)
    if mode == TrendMode.MACD_CROSS:
        fast = ema(close, params.macd_fast)
        slow = ema(close, params.macd_slow)
        diff = fast - slow
        dea = ema(diff, params.macd_signal)
        return diff - dea
    if mode == TrendMode.MA_CROSS:
        ma_fast = close.rolling(params.ma_fast, min_periods=params.ma_fast).mean()
        ma_slow = close.rolling(params.ma_slow, min_periods=params.ma_slow).mean()
        return ema(ma_fast - ma_slow, params.ma_smooth)
    if mode == TrendMode.BOLL_TREND:
        middle = close.rolling(params.boll_window, min_periods=params.boll_window).mean()
        return close - middle
    if mode == TrendMode.LONG_MA_TREND:
        trend_line = _compute_ma(close, params.trend_ma_period, params.trend_ma_type)
        slope = trend_line.diff(params.slope_lookback)
        raw = close - trend_line
        if params.require_positive_slope:
            raw = raw.where(slope > 0, -raw.abs())
        return raw
    if mode == TrendMode.DUAL_MA_CROSS:
        fast = _compute_ma(close, params.dual_ma_fast, params.trend_ma_type)
        slow = _compute_ma(close, params.dual_ma_slow, params.trend_ma_type)
        return fast - slow
    if mode == TrendMode.TREND_SCORE:
        return _trend_score(df, close, params)
    raise ValueError(f"unsupported trend mode: {params.mode}")


def _compute_ma(series: pd.Series, period: int, ma_type: str) -> pd.Series:
    if ma_type == "ema":
        return ema_seeded(series, period)
    if ma_type == "wma":
        return series.rolling(period, min_periods=period).apply(
            lambda x: ((x * (np.arange(len(x)) + 1)) / (np.arange(len(x)) + 1).sum()).sum()
        )
    if ma_type == "hull":
        wma_half = _wma(series, period // 2)
        wma_full = _wma(series, period)
        hma_raw = 2 * wma_half - wma_full
        return _wma(hma_raw, int(np.sqrt(period)))
    return ema(series, period)


def _wma(series: pd.Series, period: int) -> pd.Series:
    p = max(1, int(period))
    return series.rolling(p, min_periods=p).apply(
        lambda x: ((x * (np.arange(len(x)) + 1)) / (np.arange(len(x)) + 1).sum()).sum(),
        raw=True,
    )


def _trend_score(df: pd.DataFrame, close: pd.Series, params: DKTrendParams) -> pd.Series:
    ma_long = _compute_ma(close, params.trend_score_ma_long, params.trend_ma_type)
    ma_fast = _compute_ma(close, params.trend_score_ma_fast, params.trend_ma_type)
    ma_slow = _compute_ma(close, params.trend_score_ma_slow, params.trend_ma_type)
    cond_price_above_long = (close > ma_long).astype(float)
    cond_fast_above_slow = (ma_fast > ma_slow).astype(float)
    score = 0.5 * cond_price_above_long + 0.5 * cond_fast_above_slow
    return (score - 0.5) * 2


def _run_lengths(colors: pd.Series) -> pd.Series:
    run = []
    prev = None
    n = 0
    for color in colors:
        if color not in ("red", "green"):
            run.append(0)
            prev = None
            n = 0
            continue
        if color == prev:
            n += 1
        else:
            n = 1
            prev = color
        run.append(n)
    return pd.Series(run, index=colors.index, dtype="int64")


def compute_dktrend(df: pd.DataFrame, params: DKTrendParams | None = None) -> pd.DataFrame:
    """
    Add ``dk_value``, ``dk_color``, ``dk_signal`` and ``dk_run_len`` columns.

    ``dk_color`` is red when the selected trend value is positive, otherwise green.
    Signals are emitted only on red/green transitions after the initial warmup span.
    """
    p = params or DKTrendParams()
    out = _validate_ohlcv(df)
    value = _indicator_value(out, p)
    valid = value.notna()
    color = pd.Series("", index=out.index, dtype="object")
    color.loc[valid & (value > 0)] = "red"
    color.loc[valid & (value <= 0)] = "green"

    run_len = _run_lengths(color)
    prev_color = color.shift(1)
    min_run = max(int(p.min_run_len), 1)
    signal = pd.Series("", index=out.index, dtype="object")
    if min_run <= 1:
        signal.loc[(color == "red") & (prev_color == "green")] = "buy"
        signal.loc[(color == "green") & (prev_color == "red")] = "sell"
    else:
        prev_run_len = run_len.shift(1).fillna(0).astype("int64")
        signal.loc[(color == "red") & (run_len >= min_run) & (prev_run_len < min_run)] = "buy"
        signal.loc[(color == "green") & (run_len >= min_run) & (prev_run_len < min_run)] = "sell"

    out["dk_value"] = value.astype(float)
    out["dk_color"] = color
    out["dk_signal"] = signal
    out["dk_run_len"] = run_len
    return out.replace({np.nan: np.nan})
