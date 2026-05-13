"""Generate trading signals from DK trend colors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, compute_dktrend

from .types import Position, Signal


@dataclass(frozen=True)
class SignalRecord:
    trade_date: pd.Timestamp
    signal: Signal
    close: float
    dk_color: str
    dk_run_len: int
    position_after: Position
    quality_score: float = 0.0


def _signal_from_raw(raw: str) -> Signal:
    if raw == "buy":
        return Signal.BUY
    if raw == "sell":
        return Signal.SELL
    return Signal.HOLD


def apply_volume_confirmation(
    trend: pd.DataFrame,
    *,
    enabled: bool = False,
    lookback: int = 20,
    volume_ratio_min: float = 1.0,
) -> pd.DataFrame:
    """Downgrade BUY signals until volume confirms the red trend transition."""
    if not enabled:
        return trend
    if "volume" not in trend.columns:
        out = trend.copy()
        out["dk_signal"] = out["dk_signal"].mask(out["dk_signal"] == "buy", "")
        return out

    out = trend.copy()
    vol = pd.to_numeric(out["volume"], errors="coerce")
    avg = vol.rolling(max(int(lookback), 1), min_periods=max(int(lookback), 1)).mean()
    confirmed = vol >= avg * max(float(volume_ratio_min), 0.0)

    filtered_signal: list[str] = []
    in_position = False
    pending_buy = False
    for raw_sig, color, ok in zip(out["dk_signal"], out["dk_color"], confirmed):
        sig = ""
        raw = str(raw_sig)
        c = str(color)
        if c == "green":
            pending_buy = False
            if raw == "sell" and in_position:
                sig = "sell"
                in_position = False
        elif c == "red":
            if raw == "buy" and not in_position:
                pending_buy = True
            if pending_buy and not in_position and bool(ok):
                sig = "buy"
                in_position = True
                pending_buy = False
        filtered_signal.append(sig)

    out["dk_signal"] = filtered_signal
    return out


def compute_signal_quality(
    trend: pd.DataFrame,
    *,
    volume_ratio_min: float = 1.5,
    volume_lookback: int = 20,
    atr_lookback: int = 60,
    market_returns: pd.Series | None = None,
    market_lookback: int = 10,
) -> pd.Series:
    """Score each BUY signal (0-100) based on confirming evidence.

    Points:
    - +20: color run_len >= 2
    - +20: volume >= volume_ratio_min * avg volume
    - +20: resonance count == 3 (if available)
    - +20: market return > 0 over market_lookback
    - +20: current ATR < median ATR over atr_lookback
    """
    scores = pd.Series(0.0, index=trend.index, dtype="float64")
    buy_mask = trend["dk_signal"] == "buy"

    # run_len >= 2
    run_len = trend.get("dk_run_len", pd.Series(0, index=trend.index))
    scores.loc[buy_mask & (run_len >= 2)] += 20

    # volume confirmation
    if "volume" in trend.columns:
        vol = pd.to_numeric(trend["volume"], errors="coerce")
        avg_vol = vol.rolling(volume_lookback, min_periods=volume_lookback).mean()
        scores.loc[buy_mask & (vol >= avg_vol * volume_ratio_min)] += 20

    # resonance
    if "consensus_red_count" in trend.columns:
        scores.loc[buy_mask & (trend["consensus_red_count"] >= 3)] += 20

    # market regime
    if market_returns is not None and not market_returns.empty:
        aligned = market_returns.reindex(trend.index).fillna(0.0)
        market_cum = (1.0 + aligned).rolling(market_lookback, min_periods=market_lookback).apply(
            lambda x: (x + 1.0).prod() - 1.0, raw=True
        )
        scores.loc[buy_mask & (market_cum > 0)] += 20

    # low-volatility entry (ATR below median)
    if all(c in trend.columns for c in ("high", "low", "close")):
        high = pd.to_numeric(trend["high"], errors="coerce")
        low = pd.to_numeric(trend["low"], errors="coerce")
        close = pd.to_numeric(trend["close"], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        atr_median = atr.rolling(atr_lookback, min_periods=atr_lookback).median()
        scores.loc[buy_mask & (atr < atr_median)] += 20

    return scores.clip(upper=100.0)


def generate_signals(
    ohlcv: pd.DataFrame,
    params: DKTrendParams | None = None,
    *,
    volume_confirm: bool = False,
    volume_lookback: int = 20,
    volume_ratio_min: float = 1.0,
) -> list[SignalRecord]:
    """Compute DK trend and return one record per valid bar."""
    trend = compute_dktrend(ohlcv, params or DKTrendParams())
    trend = apply_volume_confirmation(
        trend,
        enabled=volume_confirm,
        lookback=volume_lookback,
        volume_ratio_min=volume_ratio_min,
    )
    trend_reset = trend.reset_index(drop=True)
    quality = compute_signal_quality(trend_reset).reset_index(drop=True)
    records: list[SignalRecord] = []
    position = Position.FLAT
    for pos_idx, row in trend_reset.iterrows():
        color = str(row.get("dk_color", ""))
        if color not in ("red", "green"):
            continue
        sig = _signal_from_raw(str(row.get("dk_signal", "")))
        if sig == Signal.BUY:
            position = Position.LONG
        elif sig == Signal.SELL:
            position = Position.FLAT
        records.append(
            SignalRecord(
                trade_date=pd.Timestamp(row.get("trade_date", row.name)).normalize(),
                signal=sig,
                close=float(row["close"]),
                dk_color=color,
                dk_run_len=int(row["dk_run_len"]),
                position_after=position,
                quality_score=float(quality.iloc[pos_idx]) if sig == Signal.BUY else 0.0,
            )
        )
    return records


def get_current_signal(
    symbol: str,
    db_path: str | Path,
    params: DKTrendParams | None = None,
) -> SignalRecord:
    """Read recent daily bars from DuckDB and return the latest trend state."""
    code = str(symbol).strip().zfill(6)
    with DuckDBManager(duckdb_path=str(db_path)) as db:
        df = db.read_daily_frame(symbols=[code])
    if df.empty:
        raise ValueError(f"no daily data found for symbol {code}")
    records = generate_signals(df, params or DKTrendParams())
    if not records:
        raise ValueError(f"not enough daily data to compute DK trend for {code}")
    return records[-1]
