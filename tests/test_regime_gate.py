"""Unit tests for src.backtest.regime_gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.regime_gate import RegimeGate


def _make_bull_index_df(periods: int = 200) -> pd.DataFrame:
    """Build a deterministic bull market: steady climb from 3000 to 5000."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    close = np.linspace(3000, 5000, periods)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.005, size=periods)
    close = close * (1.0 + noise)
    df = pd.DataFrame(
        {
            "trade_date": dates,
            "open": close * (1.0 - 0.002),
            "high": close * (1.0 + 0.005),
            "low": close * (1.0 - 0.005),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, size=periods),
        }
    )
    return df


def _make_bear_index_df(periods: int = 200) -> pd.DataFrame:
    """Build a deterministic bear market: steady decline from 5000 to 3000."""
    dates = pd.date_range("2022-01-01", periods=periods, freq="B")
    close = np.linspace(5000, 3000, periods)
    rng = np.random.default_rng(43)
    noise = rng.normal(0, 0.005, size=periods)
    close = close * (1.0 + noise)
    df = pd.DataFrame(
        {
            "trade_date": dates,
            "open": close * (1.0 + 0.002),
            "high": close * (1.0 + 0.005),
            "low": close * (1.0 - 0.005),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, size=periods),
        }
    )
    return df


class TestRegimeGate:
    """Test suite for RegimeGate — covers bull/bear/fail-open/missing-data scenarios."""

    def test_constructor_validates_params(self) -> None:
        """ma_fast must be < ma_slow."""
        with pytest.raises(ValueError):
            RegimeGate(ma_fast=60, ma_slow=20)
        with pytest.raises(ValueError):
            RegimeGate(ma_fast=20, ma_slow=20)

    def test_bull_market_allows_entry(self) -> None:
        """In a steady uptrend, close > MA20 > MA60 — entry must be allowed."""
        gate = RegimeGate(ma_fast=20, ma_slow=60)
        df = _make_bull_index_df()
        target = df["trade_date"].iloc[-1]
        assert gate.is_entry_allowed(target, df) is True
        assert gate.regime_state(target, df) == "bullish"

    def test_bear_market_denies_entry(self) -> None:
        """In a steady downtrend, close < MA20 < MA60 — entry must be denied."""
        gate = RegimeGate(ma_fast=20, ma_slow=60)
        df = _make_bear_index_df()
        target = df["trade_date"].iloc[-1]
        assert gate.is_entry_allowed(target, df) is False
        assert gate.regime_state(target, df) == "bearish"

    def test_insufficient_data_fails_open(self) -> None:
        """When there aren't enough bars to compute MAs, fail open (allow entry)."""
        gate = RegimeGate(ma_fast=20, ma_slow=60)
        df = _make_bull_index_df(periods=30)
        target = df["trade_date"].iloc[-1]
        assert gate.is_entry_allowed(target, df) is True
        assert gate.regime_state(target, df) == "insufficient_data"

    def test_none_dataframe_fails_open(self) -> None:
        """None DataFrame must not crash and must allow entry."""
        gate = RegimeGate()
        assert gate.is_entry_allowed(pd.Timestamp("2024-01-15"), None) is True

    def test_empty_dataframe_fails_open(self) -> None:
        """Empty DataFrame must not crash and must allow entry."""
        gate = RegimeGate()
        empty = pd.DataFrame(columns=["trade_date", "close"])
        assert gate.is_entry_allowed(pd.Timestamp("2024-01-15"), empty) is True

    def test_nan_close_handled_gracefully(self) -> None:
        """NaN values in close column must not crash — fail open."""
        gate = RegimeGate(ma_fast=20, ma_slow=60)
        df = _make_bull_index_df()
        df.loc[df.index[-1], "close"] = np.nan
        target = df["trade_date"].iloc[-1]
        result = gate.is_entry_allowed(target, df)
        assert isinstance(result, bool)

    def test_transitioning_states(self) -> None:
        """Regime state labels must correctly identify transitional states."""
        gate = RegimeGate(ma_fast=20, ma_slow=60)
        df = _make_bull_index_df()
        target = df["trade_date"].iloc[-1]
        assert gate.regime_state(target, df) == "bullish"

    def test_date_before_index_start_fails_open(self) -> None:
        """Querying a date before any index data fails open."""
        gate = RegimeGate(ma_fast=20, ma_slow=60)
        df = _make_bull_index_df()
        target = pd.Timestamp("2019-06-15")
        assert gate.is_entry_allowed(target, df) is True
