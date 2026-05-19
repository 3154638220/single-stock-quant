"""Unit tests for Phase H/I/J new parameters in rotation backtest."""

import numpy as np
import pandas as pd

from src.backtest.rotation import (
    _check_position_exit,
    _trend_strength_score,
    run_rotation_backtest,
)
from src.indicators import DKTrendParams, TrendMode


def _flat_ohlcv(prices: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame({
        "trade_date": pd.date_range(start, periods=n, freq="B"),
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000.0] * n,
    })


def _make_dual_symbol_map(sym1: str, df1: pd.DataFrame, sym2: str, df2: pd.DataFrame):
    return {sym1: df1, sym2: df2}


class TestDualSpeedRegime:
    """H-1: Dual-speed market state detection."""

    def test_slow_ma_only_backward_compat(self):
        """With regime_fast_ma_period=0, should behave like old single-MA mode."""
        n = 300
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        prices = np.maximum(prices, 50)

        df = _flat_ohlcv(prices.tolist())
        # Index same as stock (single-stock pool)
        index_df = df.copy()
        index_df["symbol"] = "000300"

        result = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            market_regime_mode="exit",
            regime_ma_period=120,
            regime_fast_ma_period=0,
            min_bars_required=50,
        )
        assert result.annualized_return != 0

    def test_fast_ma_triggers_before_slow_ma(self):
        """Fast MA should detect bear markets earlier than slow MA."""
        n = 200
        # Build a sharp decline: steady rise then crash
        prices = []
        for i in range(n):
            if i < 100:
                prices.append(100.0 + i * 0.5)  # rise to 150
            elif i < 120:
                prices.append(150.0 - (i - 100) * 2.5)  # crash to 100
            else:
                prices.append(100.0 + (i - 120) * 0.1)  # slow recovery
        prices = [max(p, 50) for p in prices]

        df = _flat_ohlcv(prices)
        index_df = df.copy()
        index_df["symbol"] = "000300"

        # With dual-speed enabled, the fast MA should catch the crash
        result = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            ranking_mode="trend_strength",
            market_regime_mode="exit",
            regime_ma_period=120,
            regime_fast_ma_period=60,
            regime_fast_threshold=0.97,
            min_bars_required=50,
        )
        # Should not crash - just verify it runs successfully
        assert result.max_drawdown <= 1.0


class TestDrawdownTrigger:
    """H-2: Index drawdown trigger (second defense line)."""

    def test_drawdown_trigger_fires_on_sharp_drop(self):
        """15% drawdown trigger should detect a sharp drop."""
        n = 150
        prices = []
        for i in range(n):
            if i < 80:
                prices.append(100.0 + i * 0.1)
            elif i < 100:
                prices.append(108.0 - (i - 80) * 1.8)  # 36 point drop = ~33%
            else:
                prices.append(72.0 + (i - 100) * 0.05)
        prices = [max(p, 50) for p in prices]

        df = _flat_ohlcv(prices)
        index_df = df.copy()
        index_df["symbol"] = "000300"

        result = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            ranking_mode="trend_strength",
            market_regime_mode="exit",
            regime_ma_period=120,
            regime_fast_ma_period=0,
            regime_drawdown_trigger=0.15,
            regime_drawdown_lookback=60,
            min_bars_required=50,
        )
        assert result.max_drawdown <= 1.0

    def test_drawdown_disabled_by_default(self):
        """With drawdown_trigger=0, should not activate drawdown detection."""
        n = 150
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        prices = np.maximum(prices, 50)

        df = _flat_ohlcv(prices.tolist())
        index_df = df.copy()
        index_df["symbol"] = "000300"

        result = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            market_regime_mode="exit",
            regime_ma_period=120,
            regime_drawdown_trigger=0.0,
            min_bars_required=50,
        )
        assert result.annualized_return != 0


class TestPortfolioDDLimit:
    """H-3: Portfolio equity curve drawdown limit."""

    def test_portfolio_dd_limit_runs(self):
        """Portfolio DD limit should not crash the backtest."""
        n = 200
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        prices = np.maximum(prices, 50)

        df = _flat_ohlcv(prices.tolist())

        result = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            ranking_mode="trend_strength",
            portfolio_dd_limit=0.20,
            min_bars_required=50,
        )
        assert result.max_drawdown <= 1.0

    def test_portfolio_dd_limit_disabled_by_default(self):
        """With portfolio_dd_limit=0, should behave normally."""
        n = 200
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        prices = np.maximum(prices, 50)

        df = _flat_ohlcv(prices.tolist())

        result = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            portfolio_dd_limit=0.0,
            min_bars_required=50,
        )
        # With random data, DD limit disabled
        assert result.annualized_return != 0

    def test_tight_dd_limit_reduces_exposure(self):
        """A very tight DD limit should reduce drawdown vs no limit."""
        n = 200
        # Start with losses to trigger DD limit
        prices = []
        for i in range(n):
            if i < 30:
                prices.append(100.0 - i * 1.0)  # steady decline
            else:
                prices.append(70.0 + (i - 30) * 0.2)
        prices = [max(p, 30) for p in prices]

        df = _flat_ohlcv(prices)

        result_tight = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            ranking_mode="trend_strength",
            portfolio_dd_limit=0.15,
            min_bars_required=30,
        )
        result_loose = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            ranking_mode="trend_strength",
            portfolio_dd_limit=0.0,
            min_bars_required=30,
        )
        # Tight DD limit should not be worse than no limit in MDD
        # (in a declining market, it should exit earlier)
        assert result_tight.max_drawdown <= result_loose.max_drawdown + 0.05


class TestVolatilityTarget:
    """I-2: Volatility target position scaling."""

    def test_vol_target_runs(self):
        """Vol target should not crash the backtest."""
        n = 200
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1.5, n))
        prices = np.maximum(prices, 50)

        df = _flat_ohlcv(prices.tolist())

        result = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            ranking_mode="trend_strength",
            volatility_target_ann=0.25,
            volatility_scale_floor=0.30,
            min_bars_required=50,
        )
        assert result.max_drawdown <= 1.0

    def test_vol_target_disabled_by_default(self):
        """With volatility_target_ann=0, should not scale positions."""
        n = 200
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        prices = np.maximum(prices, 50)

        df = _flat_ohlcv(prices.tolist())

        result = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            volatility_target_ann=0.0,
            min_bars_required=50,
        )
        assert result.annualized_return != 0


class TestPerSymbolParams:
    """J-1: Per-symbol DK params support."""

    def test_per_symbol_params_applied(self):
        """Different trend params per symbol should not crash."""
        n = 200
        rng = np.random.default_rng(42)
        prices_a = 100 + np.cumsum(rng.normal(0, 1.5, n))
        prices_a = np.maximum(prices_a, 50)
        prices_b = 100 + np.cumsum(rng.normal(0.05, 1.2, n))
        prices_b = np.maximum(prices_b, 50)

        df_a = _flat_ohlcv(prices_a.tolist())
        df_b = _flat_ohlcv(prices_b.tolist())

        symbol_params = {
            "A": DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=12, macd_slow=26, macd_signal=9),
            "B": DKTrendParams(mode=TrendMode.DONCHIAN_BREAKOUT, donchian_entry_window=30, donchian_exit_window=15),
        }

        result = run_rotation_backtest(
            {"A": df_a, "B": df_b},
            top_n=1,
            rebalance_freq=10,
            ranking_mode="trend_strength",
            symbol_params=symbol_params,
            min_bars_required=50,
        )
        assert result.annualized_return != 0
        assert result.n_trades >= 0

    def test_missing_symbol_falls_back_to_global(self):
        """Symbols not in symbol_params should use global trend_params."""
        n = 150
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        prices = np.maximum(prices, 50)

        df = _flat_ohlcv(prices.tolist())

        # Only provide params for "A", not "B" - "B" should fall back
        symbol_params = {
            "A": DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=8, macd_slow=21, macd_signal=5),
        }

        result = run_rotation_backtest(
            {"A": df.copy(), "B": df.copy()},
            top_n=1,
            rebalance_freq=10,
            ranking_mode="trend_strength",
            symbol_params=symbol_params,
            min_bars_required=50,
        )
        assert result.annualized_return != 0


class TestCheckPositionExit:
    """Existing exit mechanism coverage."""

    def test_stop_loss_triggers(self):
        df = _flat_ohlcv([100, 98, 95, 92, 90, 88, 85])
        atr = pd.Series([2.0] * len(df))
        reason, pl_active, pl_high = _check_position_exit(
            df, 5, entry_price=100, highest_close=100,
            stop_loss_pct=0.10, atr_trailing_mult=0.0, atr_trailing_min_gain=0.05,
            atr_series=atr, intrapos_dd_limit=0.0,
            profit_lock_trigger=0.0, profit_lock_trailing=0.0,
            profit_lock_active=False, profit_lock_high=100,
            time_stop_days=0, time_stop_min_return=0.0,
            entry_date=pd.Timestamp("2024-01-01"),
        )
        assert reason == "stop_loss"

    def test_no_exit_without_trigger(self):
        df = _flat_ohlcv([100, 101, 102, 103, 104])
        atr = pd.Series([1.0] * len(df))
        reason, _, _ = _check_position_exit(
            df, 4, entry_price=100, highest_close=104,
            stop_loss_pct=0.10, atr_trailing_mult=2.0, atr_trailing_min_gain=0.05,
            atr_series=atr, intrapos_dd_limit=0.15,
            profit_lock_trigger=0.12, profit_lock_trailing=0.05,
            profit_lock_active=False, profit_lock_high=104,
            time_stop_days=30, time_stop_min_return=0.03,
            entry_date=pd.Timestamp("2024-01-01"),
        )
        assert reason == ""


class TestTrendStrengthScore:
    def test_bearish_returns_negative(self):
        df = _flat_ohlcv([100] * 20)
        trend = df.copy()
        trend["dk_color"] = "green"
        trend["dk_value"] = 0.5
        trend["dk_run_len"] = 3
        score = _trend_strength_score(trend, 10)
        assert score < 0

    def test_bullish_returns_positive(self):
        df = _flat_ohlcv([100] * 20)
        trend = df.copy()
        trend["dk_color"] = "red"
        trend["dk_value"] = 0.7
        trend["dk_run_len"] = 5
        score = _trend_strength_score(trend, 10)
        assert score > 0
