"""Donchian channel breakout trend signal.

Buy when close breaks above the entry-period high; sell when close breaks below
the exit-period low.  The output schema is identical to ``compute_dktrend`` so it
drops into the existing backtest pipeline.
"""

from __future__ import annotations

import pandas as pd


def compute_donchian_trend(
    ohlcv: pd.DataFrame,
    *,
    entry_window: int = 20,
    exit_window: int = 10,
    min_run_len: int = 1,
) -> pd.DataFrame:
    """Return a DataFrame with ``dk_value``, ``dk_color``, ``dk_signal``, ``dk_run_len``."""
    if ohlcv.empty:
        raise ValueError("ohlcv is empty")
    missing = {"close", "high", "low"} - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv missing required columns: {sorted(missing)}")

    out = ohlcv.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
        out = out.sort_values("trade_date").set_index("trade_date", drop=False)
    else:
        out.index = pd.to_datetime(out.index).normalize()
        out = out.sort_index()

    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")

    ew = max(int(entry_window), 2)
    xw = max(int(exit_window), 2)

    donchian_high = high.rolling(ew, min_periods=ew).max().shift(1)
    donchian_low = low.rolling(xw, min_periods=xw).min().shift(1)

    # dk_value: position within donchian range [-0.5, +0.5], positive = upper half
    denom = donchian_high - donchian_low
    value = pd.Series(0.0, index=out.index, dtype=float)
    valid = denom.notna() & (denom > 0)
    value.loc[valid] = ((close.loc[valid] - donchian_low.loc[valid]) / denom.loc[valid]) - 0.5

    color = pd.Series("", index=out.index, dtype="object")
    color.loc[value.notna() & (value > 0)] = "red"
    color.loc[value.notna() & (value <= 0)] = "green"

    # Run lengths
    run = []
    prev = None
    n = 0
    for c in color:
        if c not in ("red", "green"):
            run.append(0)
            prev = None
            n = 0
            continue
        if c == prev:
            n += 1
        else:
            n = 1
            prev = c
        run.append(n)
    run_len = pd.Series(run, index=out.index, dtype="int64")

    prev_color = color.shift(1)
    min_run = max(int(min_run_len), 1)
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
    return out
