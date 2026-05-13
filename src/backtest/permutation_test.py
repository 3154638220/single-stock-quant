"""Monte Carlo permutation test for strategy Sharpe significance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.performance_panel import sharpe_ratio
from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams


@dataclass(frozen=True)
class PermutationTestResult:
    observed_sharpe: float
    null_sharpe_mean: float
    null_sharpe_p95: float
    p_value: float
    n_permutations: int
    is_significant: bool


def run_permutation_test(
    symbol: str,
    ohlcv: pd.DataFrame,
    params: DKTrendParams,
    *,
    n_permutations: int = 1000,
    **bt_kwargs,
) -> PermutationTestResult:
    """Test whether the strategy Sharpe is statistically significant.

    1. Run the real strategy to get observed_sharpe.
    2. Shuffle row order N times, re-run, collect null distribution.
    3. p_value = fraction of null Sharpes >= observed Sharpe.
    """
    real = run_single_stock_backtest(symbol, ohlcv, params, **bt_kwargs)
    observed = real.sharpe_ratio
    if not np.isfinite(observed):
        return PermutationTestResult(
            observed_sharpe=observed,
            null_sharpe_mean=float("nan"),
            null_sharpe_p95=float("nan"),
            p_value=float("nan"),
            n_permutations=n_permutations,
            is_significant=False,
        )

    n = len(ohlcv)
    n_iter = max(int(n_permutations), 1)
    null_sharpes = np.zeros(n_iter, dtype=np.float64)
    rng = np.random.default_rng(42)

    for i in range(n_iter):
        shuffled = ohlcv.sample(frac=1.0, random_state=rng.integers(0, 2**31 - 1)).reset_index(drop=True)
        try:
            res = run_single_stock_backtest(symbol, shuffled, params, **bt_kwargs)
            null_sharpes[i] = res.sharpe_ratio if np.isfinite(res.sharpe_ratio) else np.nan
        except Exception:
            null_sharpes[i] = np.nan

    valid = null_sharpes[np.isfinite(null_sharpes)]
    if len(valid) == 0:
        return PermutationTestResult(
            observed_sharpe=observed,
            null_sharpe_mean=float("nan"),
            null_sharpe_p95=float("nan"),
            p_value=float("nan"),
            n_permutations=n_iter,
            is_significant=False,
        )

    p_value = float(np.mean(valid >= observed))
    return PermutationTestResult(
        observed_sharpe=observed,
        null_sharpe_mean=float(np.mean(valid)),
        null_sharpe_p95=float(np.percentile(valid, 95)),
        p_value=p_value,
        n_permutations=n_iter,
        is_significant=bool(p_value < 0.05),
    )
