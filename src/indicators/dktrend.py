"""Approximate Eastmoney-style long/short trend indicator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .utils import ema


class TrendMode(str, Enum):
    MACD_CROSS = "macd_cross"
    MA_CROSS = "ma_cross"
    BOLL_TREND = "boll_trend"


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
    raise ValueError(f"unsupported trend mode: {params.mode}")


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
