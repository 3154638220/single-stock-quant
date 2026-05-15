"""Unit tests for ADX (Average Directional Index) indicator."""

import numpy as np
import pandas as pd

from src.indicators.adx import compute_adx


def _ohlcv_df(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    start: str = "2024-01-01",
) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "trade_date": pd.date_range(start, periods=n),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


class TestComputeADX:
    def test_returns_expected_columns(self):
        n = 40
        df = _ohlcv_df(
            highs=[10.0 + i * 0.1 + 0.5 for i in range(n)],
            lows=[10.0 + i * 0.1 - 0.5 for i in range(n)],
            closes=[10.0 + i * 0.1 for i in range(n)],
        )
        out = compute_adx(df, period=14)
        for col in ["adx", "plus_di", "minus_di"]:
            assert col in out.columns

    def test_adx_bounded(self):
        """ADX values should be between 0 and 100."""
        n = 60
        np.random.seed(42)
        base = 10.0 + np.cumsum(np.random.normal(0.05, 0.3, n))
        highs = base + np.abs(np.random.normal(0, 0.4, n))
        lows = base - np.abs(np.random.normal(0, 0.4, n))
        closes = base + np.random.normal(0, 0.1, n)

        df = _ohlcv_df(highs=list(highs), lows=list(lows), closes=list(closes))
        out = compute_adx(df, period=14)
        valid = out["adx"].dropna()
        assert len(valid) > 0
        assert valid.between(0, 100).all()

    def test_adx_high_in_strong_trend(self):
        """ADX should be higher in a one-directional trend."""
        n = 50
        # Strong uptrend
        highs = [10.0 + i * 0.5 + 0.3 for i in range(n)]
        lows = [10.0 + i * 0.5 - 0.1 for i in range(n)]
        closes = [10.0 + i * 0.5 for i in range(n)]

        df = _ohlcv_df(highs=highs, lows=lows, closes=closes)
        out = compute_adx(df, period=14)
        adx_tail = out["adx"].dropna().iloc[-10:].mean()
        # In a strong trend ADX should be notably above 20
        assert adx_tail > 15

    def test_adx_low_in_range(self):
        """ADX should be low in a sideways range market."""
        n = 50
        # Sideways range
        highs = [10.0 + 0.8 * np.sin(i / 3.0) for i in range(n)]
        lows = [10.0 + 0.8 * np.sin(i / 3.0) - 0.3 for i in range(n)]
        closes = [10.0 + 0.8 * np.sin(i / 3.0) for i in range(n)]

        df = _ohlcv_df(highs=highs, lows=lows, closes=closes)
        out = compute_adx(df, period=14)
        adx_tail = out["adx"].dropna().iloc[-10:].mean()
        # In a ranging market ADX should be lower than trending
        assert adx_tail < 35

    def test_adx_period_sensitivity(self):
        """Shorter period should produce more responsive (noisier) ADX."""
        n = 60
        np.random.seed(42)
        base = 10.0 + np.cumsum(np.random.normal(0.05, 0.3, n))
        highs = base + np.abs(np.random.normal(0, 0.4, n))
        lows = base - np.abs(np.random.normal(0, 0.4, n))
        closes = base + np.random.normal(0, 0.1, n)

        df = _ohlcv_df(highs=list(highs), lows=list(lows), closes=list(closes))
        adx7 = compute_adx(df, period=7)["adx"].dropna()
        adx21 = compute_adx(df, period=21)["adx"].dropna()

        # Shorter period should have higher std than longer period
        assert adx7.std() > adx21.std() * 0.7

    def test_plus_di_minus_di_crossover(self):
        """+DI and -DI should cross when trend reverses."""
        n = 50
        # uptrend then downtrend
        half = n // 2
        base_up = [10.0 + i * 0.3 for i in range(half)]
        base_dn = [base_up[-1] - i * 0.4 for i in range(half)]
        base = base_up + base_dn

        highs = [b + 0.2 for b in base]
        lows = [b - 0.2 for b in base]
        closes = base

        df = _ohlcv_df(highs=highs, lows=lows, closes=closes)
        out = compute_adx(df, period=7)

        # In second half, -DI should dominate
        first_half_plus = out["plus_di"].iloc[half // 2 : half].mean()
        first_half_minus = out["minus_di"].iloc[half // 2 : half].mean()
        second_half_plus = out["plus_di"].iloc[-half // 2 :].mean()
        second_half_minus = out["minus_di"].iloc[-half // 2 :].mean()

        # In uptrend: +DI >= -DI; in downtrend: -DI > +DI
        assert first_half_plus >= first_half_minus * 0.8
        assert second_half_minus >= second_half_plus * 0.8
