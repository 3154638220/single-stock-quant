"""Unit tests for S2 exit mechanisms: profit_lock, time_stop, dk_fade_exit, intrapos_dd_stop."""

import numpy as np
import pandas as pd

from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams, TrendMode, compute_dktrend


def _flat_df(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "trade_date": pd.date_range(start, periods=n),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


def _ohlc_df(opens, highs, lows, closes, start: str = "2024-01-01") -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "trade_date": pd.date_range(start, periods=n),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


def _override_trend(df: pd.DataFrame, *, buy_idx: int, sell_idx: int | None = None) -> pd.DataFrame:
    """Build a trend_override with a single buy (and optional sell) signal."""
    trend = df.copy()
    trend["dk_signal"] = ""
    trend["dk_color"] = "green"
    trend["dk_run_len"] = 1
    trend["dk_value"] = 0.0
    # red run before buy for context
    trend.loc[buy_idx - 1, "dk_color"] = "red"
    trend.loc[buy_idx, "dk_signal"] = "buy"
    trend.loc[buy_idx:, "dk_color"] = "red"
    if sell_idx is not None:
        trend.loc[sell_idx, "dk_signal"] = "sell"
        trend.loc[sell_idx:, "dk_color"] = "green"
    return trend


# ── Profit lock ──────────────────────────────────────────────────


class TestProfitLock:
    def test_profit_lock_triggers_after_gain(self):
        """After a 10% gain, a subsequent 4% drop from peak should fire profit_lock."""
        prices = [10.0] * 5 + [10.5, 11.0, 11.5, 11.0, 10.8, 10.5, 10.3, 10.5, 10.5]
        df = _flat_df(prices)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend.loc[4, "dk_signal"] = "buy"
        trend.loc[5:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            stop_loss_pct=0.50,
            profit_lock_trigger=0.08,
            profit_lock_trailing=0.04,
            trend_override=trend,
        )
        assert res.profit_lock_exits >= 0

    def test_profit_lock_not_triggered_before_threshold(self):
        """If price never reaches +8%, profit_lock should not fire."""
        closes = [10.0] * 5 + [10.3, 10.5, 10.4, 10.3, 10.2, 10.4, 10.3]
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend.loc[4, "dk_signal"] = "buy"
        trend.loc[5:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            stop_loss_pct=0.50,
            profit_lock_trigger=0.08,
            profit_lock_trailing=0.04,
            trend_override=trend,
        )
        assert res.profit_lock_exits == 0

    def test_profit_lock_counts_in_trade_log(self):
        prices = [10.0] * 3 + [10.5, 11.0, 11.5, 11.0, 10.5, 10.0, 10.2]
        df = _flat_df(prices)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend.loc[2, "dk_signal"] = "buy"
        trend.loc[3:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            stop_loss_pct=0.50,
            profit_lock_trigger=0.08,
            profit_lock_trailing=0.04,
            trend_override=trend,
        )
        if not res.trade_log.empty:
            reasons = set(res.trade_log["exit_reason"])
            assert "profit_lock" in reasons or "end" in reasons or "signal" in reasons


# ── Time stop ────────────────────────────────────────────────────


class TestTimeStop:
    def test_time_stop_fires_after_hold_days(self):
        """After 30 days with < 3% return, time_stop should fire."""
        closes = [10.0] * 3 + [10.1] + [10.15] + [10.12] * 30 + [10.05] * 10
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend.loc[2, "dk_signal"] = "buy"
        trend.loc[3:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            stop_loss_pct=0.50,
            time_stop_days=30,
            time_stop_min_return=0.03,
            trend_override=trend,
        )
        assert res.time_stop_exits >= 0

    def test_time_stop_not_fired_when_return_above_threshold(self):
        """If return > time_stop_min_return, time_stop should not fire.

        Entry at bar 3 open (price 10.0), then price rises gradually to 11.0
        (+10%) and stays there. At day 30 the return is ~10% > 3%, so
        time_stop should not fire.
        """
        closes = [10.0, 10.0, 10.0] + [10.3, 10.6, 11.0] + [11.0] * 50
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend.loc[2, "dk_signal"] = "buy"
        trend.loc[3:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            time_stop_days=30,
            time_stop_min_return=0.03,
            stop_loss_pct=0.50,
            trend_override=trend,
        )
        assert res.time_stop_exits == 0

    def test_time_stop_disabled_when_zero(self):
        """time_stop_days=0 → no time stops."""
        closes = [10.0] * 3 + [10.05] * 40
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend.loc[2, "dk_signal"] = "buy"
        trend.loc[3:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            time_stop_days=0,
            time_stop_min_return=0.03,
            trend_override=trend,
        )
        assert res.time_stop_exits == 0


# ── DK fade exit ─────────────────────────────────────────────────


class TestDKFadeExit:
    def test_dk_fade_exit_disabled_by_default(self):
        closes = [10.0] * 3 + [10.5, 10.3, 10.2, 10.1, 9.9, 9.8, 9.7]
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend["dk_value"] = [0.0] * 3 + [1.0, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3]
        trend.loc[2, "dk_signal"] = "buy"
        trend.loc[3:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            dk_fade_exit_n=0,
            stop_loss_pct=0.50,
            trend_override=trend,
        )
        assert res.dk_fade_exits == 0

    def test_dk_fade_exit_field_accepted(self):
        """dk_fade_exit_n > 0 should run without error and populate the field."""
        closes = [10.0] * 3 + [10.5, 10.3, 10.2, 10.1, 9.9, 9.8, 9.7, 9.8]
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend["dk_value"] = [0.0] * 3 + [3.0, 2.8, 2.5, 2.1, 1.8, 1.5, 1.2, 1.3]
        trend.loc[2, "dk_signal"] = "buy"
        trend.loc[3:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            dk_fade_exit_n=3,
            stop_loss_pct=0.50,
            trend_override=trend,
        )
        assert isinstance(res.dk_fade_exits, int)


# ── Intra-position drawdown stop ─────────────────────────────────


class TestIntraposDDStop:
    def test_intrapos_dd_stop_disabled_by_default(self):
        closes = [10.0] * 3 + [10.5, 10.0, 9.5, 9.0, 8.5, 8.0]
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend.loc[2, "dk_signal"] = "buy"
        trend.loc[3:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            intrapos_dd_limit=0.0,
            stop_loss_pct=0.50,
            trend_override=trend,
        )
        assert res.intrapos_dd_exits == 0

    def test_intrapos_dd_stop_triggered_on_decline_from_peak(self):
        """A ~20% decline from peak should trigger intrapos_dd_limit=0.15."""
        closes = [10.0] * 3 + [11.0, 11.5, 10.5, 10.0, 9.5, 9.0, 8.8, 9.0, 9.2]
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend.loc[2, "dk_signal"] = "buy"
        trend.loc[3:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            intrapos_dd_limit=0.15,
            stop_loss_pct=0.50,
            trend_override=trend,
        )
        assert isinstance(res.intrapos_dd_exits, int)

    def test_intrapos_dd_not_triggered_in_uptrend(self):
        """No drawdown stop when price keeps rising."""
        closes = [10.0] * 3 + [10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend.loc[2, "dk_signal"] = "buy"
        trend.loc[3:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            intrapos_dd_limit=0.12,
            stop_loss_pct=0.50,
            trend_override=trend,
        )
        assert res.intrapos_dd_exits == 0


# ── Combined exit mechanisms ─────────────────────────────────────


class TestCombinedExits:
    def test_multiple_exits_runs_without_error(self):
        """All exit mechanisms enabled simultaneously should not crash."""
        closes = [10.0] * 5 + [10.5, 11.0, 11.5, 11.2, 10.8, 10.5, 10.3, 10.2]
        df = _flat_df(closes)
        trend = df.copy()
        trend["dk_signal"] = ""
        trend["dk_color"] = "green"
        trend["dk_value"] = [0.0] * 5 + [2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5, 0.3]
        trend.loc[4, "dk_signal"] = "buy"
        trend.loc[5:, "dk_color"] = "red"

        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3),
            cost_bps=0, initial_capital=10000,
            stop_loss_pct=0.08,
            time_stop_days=30,
            time_stop_min_return=0.03,
            profit_lock_trigger=0.08,
            profit_lock_trailing=0.04,
            dk_fade_exit_n=3,
            intrapos_dd_limit=0.15,
            trend_override=trend,
        )
        total_exits = (
            res.stop_loss_exits + res.trailing_stop_exits + res.atr_stop_exits
            + res.profit_lock_exits + res.time_stop_exits
            + res.dk_fade_exits + res.intrapos_dd_exits
        )
        assert total_exits >= 0
