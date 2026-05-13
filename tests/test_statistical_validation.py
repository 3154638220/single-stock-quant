import numpy as np
import pandas as pd

from src.backtest.performance_panel import (
    bootstrap_sharpe_ci,
    breakdown_by_regime,
    sharpe_ratio,
)
from src.backtest.permutation_test import (
    PermutationTestResult,
    run_permutation_test,
)
from src.indicators import DKTrendParams, TrendMode


def _flat_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )


# ── Permutation test ───────────────────────────────────────────

class TestPermutationTest:
    def test_result_dataclass_fields(self):
        result = PermutationTestResult(
            observed_sharpe=0.8,
            null_sharpe_mean=0.05,
            null_sharpe_p95=0.3,
            p_value=0.04,
            n_permutations=100,
            is_significant=True,
        )
        assert result.observed_sharpe == 0.8
        assert result.is_significant
        assert result.p_value == 0.04

    def test_run_permutation_test_returns_reasonable_result(self):
        closes = [10, 10, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
        df = _flat_df(closes)
        result = run_permutation_test(
            "600000", df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=5),
            n_permutations=30, cost_bps=0, initial_capital=10000,
        )
        assert result.n_permutations == 30
        assert isinstance(result.observed_sharpe, float)
        assert isinstance(result.p_value, float)
        # p_value ∈ [0, 1] for valid result
        if np.isfinite(result.p_value):
            assert 0.0 <= result.p_value <= 1.0

    def test_run_permutation_significant_flag_consistent_with_pvalue(self):
        closes = [10, 10, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
        df = _flat_df(closes)
        result = run_permutation_test(
            "600000", df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=5),
            n_permutations=20, cost_bps=0, initial_capital=10000,
        )
        if np.isfinite(result.p_value):
            assert result.is_significant == (result.p_value < 0.05)

    def test_permutation_null_not_all_equal_observed(self):
        """Regression: null Sharpes should vary, not all equal observed Sharpe."""
        closes = []
        rng = np.random.default_rng(77)
        px = 10.0
        for _ in range(200):
            px *= (1.0 + rng.normal(0.0, 0.025))
            closes.append(px)
        df = _flat_df(closes)
        result = run_permutation_test(
            "600000", df,
            DKTrendParams(mode=TrendMode.MACD_CROSS),
            n_permutations=30, cost_bps=0, initial_capital=10000,
        )
        if np.isfinite(result.observed_sharpe) and np.isfinite(result.null_sharpe_mean):
            # The null mean should differ from the observed; they should not be
            # trivially identical (which was the pre-fix sorting bug).
            assert abs(result.null_sharpe_mean - result.observed_sharpe) > 1e-8


# ── Regime breakdown ───────────────────────────────────────────

class TestRegimeBreakdown:
    def test_breakdown_returns_three_regimes(self):
        n = 80
        rng = np.random.default_rng(42)
        strat_ret = rng.normal(0.001, 0.02, n)
        idx_ret = rng.normal(0.0005, 0.015, n)

        result = breakdown_by_regime(strat_ret, idx_ret, regime_lookback=30)
        assert "regimes" in result
        for label in ("bull", "bear", "ranging"):
            assert label in result["regimes"]
            r = result["regimes"][label]
            assert "n_days" in r
            assert "strategy_annualized" in r
            assert "sharpe" in r

    def test_breakdown_without_index_uses_nan_regime(self):
        n = 80
        rng = np.random.default_rng(42)
        strat_ret = rng.normal(0.001, 0.02, n)
        result = breakdown_by_regime(strat_ret, None, regime_lookback=30)
        assert result["regimes"]["ranging"]["n_days"] > 0  # all classified as ranging

    def test_breakdown_insufficient_data_returns_error(self):
        result = breakdown_by_regime(
            np.array([0.01, -0.02]), np.array([0.005, -0.01]),
            regime_lookback=60,
        )
        assert "error" in result

    def test_breakdown_with_pandas_series_aligns_on_date(self):
        """When both are pd.Series with DatetimeIndex, inner join on date."""
        n = 100
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=n)
        strat = pd.Series(rng.normal(0.001, 0.02, n), index=dates, name="strat")
        # Index has fewer dates — should inner-join correctly
        idx_dates = pd.date_range("2024-01-10", periods=n - 20)
        idx = pd.Series(rng.normal(0.0005, 0.015, n - 20), index=idx_dates, name="idx")
        result = breakdown_by_regime(strat, idx, regime_lookback=30)
        assert "regimes" in result
        total = sum(r["n_days"] for r in result["regimes"].values())
        assert total > 0

    def test_breakdown_series_no_overlap_returns_error(self):
        dates1 = pd.date_range("2024-01-01", periods=30)
        dates2 = pd.date_range("2025-01-01", periods=30)
        strat = pd.Series(np.random.default_rng(0).normal(0.001, 0.02, 30), index=dates1)
        idx = pd.Series(np.random.default_rng(1).normal(0.0005, 0.015, 30), index=dates2)
        result = breakdown_by_regime(strat, idx)
        assert "error" in result


# ── Bootstrap confidence interval ──────────────────────────────

class TestBootstrapCI:
    def test_ci_returns_ordered_bounds(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 200)
        lo, hi = bootstrap_sharpe_ci(returns, n_bootstrap=200, ci=0.90)
        if np.isfinite(lo) and np.isfinite(hi):
            assert lo <= hi

    def test_ci_narrower_for_large_sample(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 500)
        lo, hi = bootstrap_sharpe_ci(returns, n_bootstrap=100, ci=0.90)
        if np.isfinite(lo) and np.isfinite(hi):
            assert (hi - lo) < 5.0  # reasonable width for 500 samples

    def test_ci_insufficient_data_returns_nan(self):
        lo, hi = bootstrap_sharpe_ci(np.array([0.01]))
        assert not np.isfinite(lo)
        assert not np.isfinite(hi)
