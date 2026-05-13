"""Consensus signal generation across DK trend modes."""

from __future__ import annotations

import pandas as pd

from src.indicators import DKTrendParams, TrendMode, compute_dktrend

from .generator import SignalRecord, apply_volume_confirmation
from .types import Position, Signal


def _run_lengths(colors: pd.Series) -> pd.Series:
    run: list[int] = []
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


def compute_consensus_trend(
    ohlcv: pd.DataFrame,
    *,
    base_params: DKTrendParams | None = None,
    n_agree: int = 2,
    modes: list[TrendMode | str] | None = None,
    volume_confirm: bool = False,
    volume_lookback: int = 20,
    volume_ratio_min: float = 1.0,
) -> pd.DataFrame:
    """Return DK-like trend columns where at least ``n_agree`` modes agree."""
    params = base_params or DKTrendParams()
    selected_modes = modes or [TrendMode.MACD_CROSS, TrendMode.MA_CROSS, TrendMode.BOLL_TREND]
    threshold = max(1, min(int(n_agree), len(selected_modes)))

    frames = []
    for mode in selected_modes:
        p = DKTrendParams(
            mode=mode,
            macd_fast=params.macd_fast,
            macd_slow=params.macd_slow,
            macd_signal=params.macd_signal,
            ma_fast=params.ma_fast,
            ma_slow=params.ma_slow,
            ma_smooth=params.ma_smooth,
            boll_window=params.boll_window,
        )
        frames.append(compute_dktrend(ohlcv, p))

    out = frames[0].copy()
    color_frame = pd.concat([f["dk_color"].rename(str(mode)) for f, mode in zip(frames, selected_modes)], axis=1)
    red_count = (color_frame == "red").sum(axis=1)
    green_count = (color_frame == "green").sum(axis=1)

    color = pd.Series("", index=out.index, dtype="object")
    color.loc[red_count >= threshold] = "red"
    color.loc[green_count >= threshold] = "green"
    prev = color.shift(1)
    signal = pd.Series("", index=out.index, dtype="object")
    signal.loc[(color == "red") & (prev != "red")] = "buy"
    signal.loc[(color == "green") & (prev != "green")] = "sell"

    out["dk_value"] = red_count - green_count
    out["dk_color"] = color
    out["dk_signal"] = signal
    out["dk_run_len"] = _run_lengths(color)
    out["consensus_red_count"] = red_count.astype("int64")
    out["consensus_green_count"] = green_count.astype("int64")
    out["consensus_n_agree"] = threshold
    return apply_volume_confirmation(
        out,
        enabled=volume_confirm,
        lookback=volume_lookback,
        volume_ratio_min=volume_ratio_min,
    )


def generate_consensus_signals(
    ohlcv: pd.DataFrame,
    *,
    base_params: DKTrendParams | None = None,
    n_agree: int = 2,
    volume_confirm: bool = False,
    volume_lookback: int = 20,
    volume_ratio_min: float = 1.0,
) -> list[SignalRecord]:
    """Generate signal records from multi-mode DK consensus."""
    trend = compute_consensus_trend(
        ohlcv,
        base_params=base_params,
        n_agree=n_agree,
        volume_confirm=volume_confirm,
        volume_lookback=volume_lookback,
        volume_ratio_min=volume_ratio_min,
    )
    records: list[SignalRecord] = []
    position = Position.FLAT
    for idx, row in trend.iterrows():
        color = str(row.get("dk_color", ""))
        if color not in ("red", "green"):
            continue
        raw = str(row.get("dk_signal", ""))
        sig = Signal.BUY if raw == "buy" else Signal.SELL if raw == "sell" else Signal.HOLD
        if sig == Signal.BUY:
            position = Position.LONG
        elif sig == Signal.SELL:
            position = Position.FLAT
        records.append(
            SignalRecord(
                trade_date=pd.Timestamp(row.get("trade_date", idx)).normalize(),
                signal=sig,
                close=float(row["close"]),
                dk_color=color,
                dk_run_len=int(row["dk_run_len"]),
                position_after=position,
            )
        )
    return records
