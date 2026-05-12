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


def _signal_from_raw(raw: str) -> Signal:
    if raw == "buy":
        return Signal.BUY
    if raw == "sell":
        return Signal.SELL
    return Signal.HOLD


def generate_signals(ohlcv: pd.DataFrame, params: DKTrendParams | None = None) -> list[SignalRecord]:
    """Compute DK trend and return one record per valid bar."""
    trend = compute_dktrend(ohlcv, params or DKTrendParams())
    records: list[SignalRecord] = []
    position = Position.FLAT
    for idx, row in trend.iterrows():
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
                trade_date=pd.Timestamp(row.get("trade_date", idx)).normalize(),
                signal=sig,
                close=float(row["close"]),
                dk_color=color,
                dk_run_len=int(row["dk_run_len"]),
                position_after=position,
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
