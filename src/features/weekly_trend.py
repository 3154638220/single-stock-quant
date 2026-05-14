"""Weekly trend state derived from daily OHLCV bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_daily(daily_ohlcv: pd.DataFrame) -> pd.DataFrame:
    need = {"open", "high", "low", "close"}
    missing = need - set(daily_ohlcv.columns)
    if missing:
        raise ValueError(f"daily_ohlcv missing required columns: {sorted(missing)}")

    df = daily_ohlcv.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    else:
        df["trade_date"] = pd.to_datetime(df.index).normalize()
    df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def compute_weekly_trend_state(
    daily_ohlcv: pd.DataFrame,
    *,
    ma_windows: tuple[int, int] = (5, 13),
) -> pd.Series:
    """Return each daily bar's last known weekly trend state.

    Daily bars are aggregated by exchange week ending Friday, labelled by the
    last actual trading day in that week. This avoids looking ahead from Monday
    through Thursday to a Friday close that was not yet known.
    """
    fast_window, slow_window = sorted((int(ma_windows[0]), int(ma_windows[1])))
    fast_window = max(fast_window, 1)
    slow_window = max(slow_window, fast_window + 1)

    df = _prepare_daily(daily_ohlcv)
    if df.empty:
        return pd.Series(dtype=object, name="weekly_trend_state")

    week_key = df["trade_date"].dt.to_period("W-FRI")
    weekly = (
        df.assign(_week=week_key)
        .groupby("_week", sort=True)
        .agg(
            trade_date=("trade_date", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum") if "volume" in df.columns else ("close", "size"),
        )
        .reset_index(drop=True)
    )

    close = pd.to_numeric(weekly["close"], errors="coerce")
    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    fast_slope = fast.diff()

    state = pd.Series("neutral", index=weekly.index, dtype=object)
    state[(fast > slow) & (fast_slope > 0)] = "bullish"
    state[(fast < slow) & (fast_slope < 0)] = "bearish"
    state[~np.isfinite(fast) | ~np.isfinite(slow)] = "neutral"

    weekly_state = pd.DataFrame({"trade_date": weekly["trade_date"], "weekly_trend_state": state})
    mapped = pd.merge_asof(
        df[["trade_date"]].sort_values("trade_date"),
        weekly_state.sort_values("trade_date"),
        on="trade_date",
        direction="backward",
    )["weekly_trend_state"].fillna("neutral")
    mapped.index = df.index
    mapped.name = "weekly_trend_state"
    return mapped
