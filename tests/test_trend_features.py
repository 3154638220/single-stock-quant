import numpy as np
import pandas as pd
import pytest

from src.features.trend_features import compute_trend_features


def _flat_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [100] * len(closes),
        }
    )


class TestTrendFeatures:
    def test_returns_dataframe_same_index(self):
        df = _flat_df([10.0] * 50)
        feats = compute_trend_features(df)
        assert isinstance(feats, pd.DataFrame)
        assert len(feats) == len(df)

    def test_ma_columns_present(self):
        df = _flat_df([10.0] * 50)
        feats = compute_trend_features(df, ma_windows=(20, 60))
        assert "ma20" in feats.columns
        assert "ma60" in feats.columns
        assert "ma20_slope" in feats.columns
        assert "close_above_ma20" in feats.columns

    def test_ma_slope_positive_in_uptrend(self):
        """In a steady uptrend, MA slope should be positive after warmup."""
        closes = [10.0 + i * 0.1 for i in range(60)]
        df = _flat_df(closes)
        feats = compute_trend_features(df, ma_windows=(20,))
        # After warmup (20 bars), slope should be positive
        late_slope = feats["ma20_slope"].iloc[40:].dropna()
        assert (late_slope > 0).all()

    def test_ma_slope_negative_in_downtrend(self):
        closes = [10.0 - i * 0.1 for i in range(60)]
        df = _flat_df(closes)
        feats = compute_trend_features(df, ma_windows=(20,))
        late_slope = feats["ma20_slope"].iloc[40:].dropna()
        assert (late_slope < 0).all()

    def test_close_above_ma_detects_crossover(self):
        """When close crosses above MA, the flag should flip."""
        closes = [10.0] * 25 + [12.0] * 25
        df = _flat_df(closes)
        feats = compute_trend_features(df, ma_windows=(20,))
        # After the jump (bar 25), close > MA until MA catches up (~20 bars later)
        # At bar 25-43: close(12) > MA (which is rising from 10 toward 12)
        assert feats["close_above_ma20"].iloc[30:44].all()

    def test_donchian_breakout_detected(self):
        closes = [10.0] * 30 + [15.0]  # new high
        # Set high higher than previous donchian range
        highs = [10.1] * 30 + [15.5]
        lows = [9.9] * 30 + [14.9]
        df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
                "open": closes,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": [100] * len(closes),
            }
        )
        feats = compute_trend_features(df, donchian_window=20)
        # The last bar should be a breakout (close >= prev donchian high)
        assert feats["donchian_breakout_20"].iloc[-1]

    def test_atr_pct_positive(self):
        df = _flat_df([10.0 + np.random.default_rng(42).normal(0, 1) for _ in range(100)])
        feats = compute_trend_features(df)
        # ATR% should be >= 0
        assert (feats["atr_pct"].dropna() >= 0).all()

    def test_atr_pct_rank_between_0_and_1(self):
        df = _flat_df([10.0 + np.random.default_rng(42).normal(0, 1) for _ in range(200)])
        feats = compute_trend_features(df, atr_rank_lookback=120)
        ranks = feats["atr_pct_rank"].dropna()
        assert len(ranks) > 0
        assert (ranks >= 0).all() and (ranks <= 1).all()

    def test_volume_ratio_present(self):
        df = _flat_df([10.0] * 50)
        feats = compute_trend_features(df)
        assert "volume_ratio_20" in feats.columns
        assert "volume_ratio_5" in feats.columns

    def test_rs_60_without_index_is_nan(self):
        df = _flat_df([10.0] * 120)
        feats = compute_trend_features(df, index_ohlcv=None)
        assert feats["rs_60"].isna().all()

    def test_rs_60_with_index_is_computed(self):
        closes_stock = [10.0 + i * 0.1 for i in range(120)]
        closes_idx = [10.0 + i * 0.05 for i in range(120)]
        df_stock = _flat_df(closes_stock)
        df_idx = _flat_df(closes_idx)
        # Use same dates for both
        df_idx["trade_date"] = df_stock["trade_date"]
        feats = compute_trend_features(df_stock, index_ohlcv=df_idx)
        rs = feats["rs_60"].dropna()
        assert len(rs) > 0
        # Stock returns faster than index → RS should be positive
        assert (rs.iloc[-20:] > 0).all()

    def test_no_future_function_ma(self):
        """ma20 at bar t must not use bar t+1's close."""
        closes = list(range(1, 101))
        df = _flat_df([float(c) for c in closes])
        feats = compute_trend_features(df, ma_windows=(20,))
        # At bar 20 (0-indexed: 19), ma20 uses closes[0:20]
        ma20_val = feats["ma20"].iloc[19]  # first valid value
        expected = np.mean(closes[0:20])
        assert np.isclose(ma20_val, expected)
        # At bar 20 (idx 19), close[19]=20, and ma20 should NOT include close[20]
        assert not np.isclose(ma20_val, np.mean(closes[1:21]))

    def test_no_future_function_donchian(self):
        """Donchian breakout at bar t uses highs up to bar t-1."""
        closes = list(range(1, 60))
        highs = [c + 0.5 for c in closes]
        df = _flat_df(closes)
        df["high"] = highs
        feats = compute_trend_features(df, donchian_window=20)
        # First valid breakout signal at bar 20 (idx 19)
        breakout_idx = feats["donchian_breakout_20"].iloc[19]  # bool
        # Donchian high uses highs[0:20], check against close[19]
        expected_dh = max(highs[0:20])
        assert np.isclose(feats[f"donchian_high_20"].iloc[19], expected_dh)

    def test_insufficient_data_returns_nan_in_features(self):
        """Very short OHLCV should produce NaNs, not crash."""
        df = _flat_df([10.0, 10.5, 10.3])
        feats = compute_trend_features(df)
        # Should not crash, and some columns should exist
        assert "ma20" in feats.columns
        # All MA values should be NaN (need 20 bars)
        assert feats["ma20"].isna().all()

    def test_volume_ratio_handles_zero_volume(self):
        df = _flat_df([10.0] * 30)
        df["volume"] = 0.0
        feats = compute_trend_features(df)
        # Should produce inf or nan, not crash
        assert not feats.empty
