"""Tests for shared tradability functions used by rotation and single_stock."""

import numpy as np
import pandas as pd

from src.market.tradability import is_tradable_open, next_buy_index, is_open_limit_up_unbuyable


def _make_df(prices_open, prices_close, volumes):
    """Helper: create a minimal OHLCV DataFrame."""
    df = pd.DataFrame({
        "open": prices_open,
        "close": prices_close,
        "volume": volumes,
    })
    return df


class TestIsTradableOpen:
    def test_normal_day_is_tradable(self):
        df = _make_df([10.0, 11.0], [10.5, 11.5], [1000, 2000])
        assert is_tradable_open(df, 0)
        assert is_tradable_open(df, 1)

    def test_negative_index_returns_false(self):
        df = _make_df([10.0], [10.5], [1000])
        assert not is_tradable_open(df, -1)

    def test_out_of_bounds_returns_false(self):
        df = _make_df([10.0], [10.5], [1000])
        assert not is_tradable_open(df, 1)

    def test_zero_volume_is_not_tradable(self):
        df = _make_df([10.0], [10.5], [0])
        assert not is_tradable_open(df, 0)

    def test_missing_volume_column_defaults_to_tradable(self):
        df = pd.DataFrame({"open": [10.0], "close": [10.5]})
        assert is_tradable_open(df, 0)

    def test_nan_price_is_not_tradable(self):
        df = _make_df([np.nan], [10.5], [1000])
        assert not is_tradable_open(df, 0)


class TestNextBuyIndex:
    def test_next_buy_at_first_tradable_day(self):
        # bar 0 has no prev_close → limit-up check is conservative → skips to bar 1
        df = _make_df([10.0, 11.0, 12.0], [10.5, 11.5, 12.5], [1000, 2000, 3000])
        idx = next_buy_index(df, "600000", 0)
        assert idx == 1

    def test_skip_suspended_day(self):
        df = _make_df([10.0, 11.0, 12.0], [10.5, 11.5, 12.5], [0, 2000, 3000])
        idx = next_buy_index(df, "600000", 0)
        assert idx == 1  # skips suspended day 0

    def test_returns_none_when_all_untradable(self):
        df = _make_df([10.0, 11.0], [10.5, 11.5], [0, 0])
        idx = next_buy_index(df, "600000", 0)
        assert idx is None

    def test_limit_up_open_is_skipped_main_board(self):
        # prev_close=10.0, limit_up=11.0, open=11.0 -> unbuyable
        df = _make_df([11.0, 10.5, 10.8], [11.0, 10.5, 10.8], [1000, 2000, 3000])
        df.loc[0, "close"] = 10.0  # bar 0 close=10, so next bar's prev_close=10
        idx = next_buy_index(df, "600000", 1)
        assert idx == 1
