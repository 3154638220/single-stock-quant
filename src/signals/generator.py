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
    trend_strength_lookback: int = 5,
) -> pd.Series:
    """Score each BUY signal (0-100) based on confirming evidence.

    Points (max 100):
    - +10: close > close.shift(3) (short-term price momentum)
    - +15: volume >= volume_ratio_min * avg volume
    - +8: resonance count >= 2 (if available)
    - +10: market return > 0 over market_lookback
    - +10: ATR percentile rank in the middle range (0.3-0.8)
    - +10: close > MA20 (trend direction)
    - +10: MA20 slope > 0 (trend momentum)
    - +12: DK value above its 5-day mean (momentum acceleration)
    - +10: close at least 20% above 52-week low
    - +15: RS_60 > 0 (relative strength vs index)
    """
    scores = pd.Series(0.0, index=trend.index, dtype="float64")
    buy_mask = trend["dk_signal"] == "buy"

    # volume confirmation
    if "volume" in trend.columns:
        vol = pd.to_numeric(trend["volume"], errors="coerce")
        avg_vol = vol.rolling(volume_lookback, min_periods=volume_lookback).mean()
        scores.loc[buy_mask & (vol >= avg_vol * volume_ratio_min)] += 15

    # resonance
    if "consensus_red_count" in trend.columns:
        scores.loc[buy_mask & (trend["consensus_red_count"] >= 2)] += 8

    # market regime
    if market_returns is not None and not market_returns.empty:
        aligned = market_returns.reindex(trend.index).fillna(0.0)
        market_cum = (1.0 + aligned).rolling(market_lookback, min_periods=market_lookback).apply(
            lambda x: (x + 1.0).prod() - 1.0, raw=True
        )
        scores.loc[buy_mask & (market_cum > 0)] += 10

    # Extract close for trend-checks (used by ATR + trend-strength blocks below)
    close = pd.to_numeric(trend["close"], errors="coerce") if "close" in trend.columns else pd.Series(dtype=float)

    # Medium-volatility entry: avoid both dead volatility and panic volatility.
    if all(c in trend.columns for c in ("high", "low", "close")):
        high = pd.to_numeric(trend["high"], errors="coerce")
        low = pd.to_numeric(trend["low"], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        atr_rank = atr.rolling(atr_lookback, min_periods=atr_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        )
        scores.loc[buy_mask & (atr_rank > 0.3) & (atr_rank < 0.8)] += 10

    # trend direction: close > MA20
    if close.notna().any():
        # Short-term price momentum replaces the old dk_run_len rule, which is
        # uninformative for first-day BUY transitions.
        scores.loc[buy_mask & (close > close.shift(3))] += 10

        ma20 = close.rolling(20, min_periods=20).mean()
        scores.loc[buy_mask & (close > ma20)] += 10

        # trend momentum: MA20 slope > 0
        ma20_slope = ma20 - ma20.shift(trend_strength_lookback)
        scores.loc[buy_mask & (ma20_slope > 0)] += 10

        if "dk_value" in trend.columns:
            dk_value = pd.to_numeric(trend["dk_value"], errors="coerce")
            dk_mean = dk_value.rolling(5, min_periods=5).mean()
            scores.loc[buy_mask & (dk_value > dk_mean)] += 12

        low_52w = close.rolling(252, min_periods=120).min()
        dist_from_low = close / low_52w - 1.0
        scores.loc[buy_mask & (dist_from_low > 0.20)] += 10

        # relative strength: stock has outperformed index over 60 days
        if market_returns is not None and not market_returns.empty:
            stock_ret_60 = close.pct_change(60)
            aligned_idx = market_returns.reindex(trend.index).fillna(0.0)
            idx_cum_60 = (1.0 + aligned_idx).rolling(60, min_periods=60).apply(
                lambda x: (x + 1.0).prod() - 1.0, raw=True
            )
            scores.loc[buy_mask & (stock_ret_60 > idx_cum_60)] += 15
        else:
            # Without index, reward positive 60-day return as a proxy
            stock_ret_60 = close.pct_change(60)
            scores.loc[buy_mask & (stock_ret_60 > 0)] += 15

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
