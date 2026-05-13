"""Monte Carlo permutation test for strategy Sharpe significance.

Permutes DK signal trigger dates while preserving real return paths and execution
rules, so the null distribution reflects "what if signals were randomly timed."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams, compute_dktrend
from src.signals.generator import apply_volume_confirmation


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
    2. For N iterations, randomly permute (circular shift) the DK signal column
       relative to prices, re-run, and collect the null Sharpe distribution.
    3. p_value = fraction of null Sharpes >= observed Sharpe.

    Circular shift preserves the signal auto-correlation structure while
    destroying any genuine timing edge.
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

    # Compute the real trend signals once.
    from src.backtest.single_stock import _prepare_ohlcv

    df = _prepare_ohlcv(ohlcv)
    volume_confirm = bt_kwargs.get("volume_confirm", False)
    volume_lookback = bt_kwargs.get("volume_lookback", 20)
    volume_ratio_min = bt_kwargs.get("volume_ratio_min", 1.0)

    trend = compute_dktrend(df, params).reset_index(drop=True)
    trend = apply_volume_confirmation(
        trend, enabled=volume_confirm, lookback=volume_lookback, volume_ratio_min=volume_ratio_min,
    ).reset_index(drop=True)

    n = len(df)
    n_iter = max(int(n_permutations), 1)
    null_sharpes = np.zeros(n_iter, dtype=np.float64)
    rng = np.random.default_rng(42)

    for i in range(n_iter):
        # Circular shift the signal column by a random offset (min 1 to avoid
        # the original alignment, max n-1).
        shift = int(rng.integers(1, max(n, 2)))
        permuted = trend.copy()
        # Only permute the signal-bearing columns; keep other columns aligned.
        for col in ["dk_signal", "dk_color", "dk_run_len"]:
            if col in permuted.columns:
                permuted[col] = np.roll(permuted[col].to_numpy(dtype=object), shift)

        try:
            res = run_single_stock_backtest(
                symbol, df, params, trend_override=permuted, **bt_kwargs,
            )
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
