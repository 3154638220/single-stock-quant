"""Market regime gate for single-stock trend strategy.

Controls new position entry permission based on CSI300 index trend.
Does NOT force-exit existing positions — only gates new entries.

Design philosophy: More permissive than H-dual+dd (prevents 2019 whipsaw),
but more conservative than "always allow" (prevents 2022 frequent losses).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeGate:
    """Market regime gate using CSI300 MA crossover.

    Allows new entries only when BOTH:
    1. CSI300 close > MA(fast)
    2. CSI300 MA(fast) > MA(slow)

    Fails open (allows entry) when index data is missing.
    """

    def __init__(self, ma_fast: int = 20, ma_slow: int = 60) -> None:
        if ma_fast >= ma_slow:
            raise ValueError(f"ma_fast ({ma_fast}) must be < ma_slow ({ma_slow})")
        self.ma_fast = int(ma_fast)
        self.ma_slow = int(ma_slow)

    def is_entry_allowed(
        self,
        date: pd.Timestamp,
        index_df: pd.DataFrame,
    ) -> bool:
        """Return True if a new long entry is permitted on *date*.

        Args:
            date: The trade date to check.
            index_df: CSI300 daily OHLCV with ``trade_date`` and ``close`` columns.

        Returns:
            True if regime allows entry, False otherwise.
            Returns True (fail-open) when index data is insufficient.
        """
        if index_df is None or index_df.empty:
            return True

        df = index_df.copy()
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        else:
            df["trade_date"] = pd.to_datetime(df.index).normalize()

        df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        close = pd.to_numeric(df["close"], errors="coerce")

        target_date = pd.Timestamp(date).normalize()
        idx = df["trade_date"].searchsorted(target_date, side="right") - 1
        if idx < max(self.ma_fast, self.ma_slow):
            return True  # fail-open: not enough history

        close_window = close.iloc[: idx + 1]
        if len(close_window) < self.ma_slow:
            return True

        ma_fast_val = close_window.iloc[-self.ma_fast :].mean()
        ma_slow_val = close_window.iloc[-self.ma_slow :].mean()
        current_close = close_window.iloc[-1]

        if not (np.isfinite(ma_fast_val) and np.isfinite(ma_slow_val) and np.isfinite(current_close)):
            return True

        return bool(current_close > ma_fast_val and ma_fast_val > ma_slow_val)

    def regime_state(
        self,
        date: pd.Timestamp,
        index_df: pd.DataFrame,
    ) -> str:
        """Return a human-readable regime label for diagnostics."""
        if index_df is None or index_df.empty:
            return "no_data"

        df = index_df.copy()
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        else:
            df["trade_date"] = pd.to_datetime(df.index).normalize()

        df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        close = pd.to_numeric(df["close"], errors="coerce")

        target_date = pd.Timestamp(date).normalize()
        idx = df["trade_date"].searchsorted(target_date, side="right") - 1
        if idx < self.ma_slow:
            return "insufficient_data"

        close_window = close.iloc[: idx + 1]
        ma_fast_val = close_window.iloc[-self.ma_fast :].mean()
        ma_slow_val = close_window.iloc[-self.ma_slow :].mean()
        current_close = close_window.iloc[-1]

        if not (np.isfinite(ma_fast_val) and np.isfinite(ma_slow_val) and np.isfinite(current_close)):
            return "invalid_data"

        above_fast = current_close > ma_fast_val
        fast_above_slow = ma_fast_val > ma_slow_val

        if above_fast and fast_above_slow:
            return "bullish"
        elif not above_fast and not fast_above_slow:
            return "bearish"
        elif above_fast and not fast_above_slow:
            return "transitioning_up"
        else:
            return "transitioning_down"
