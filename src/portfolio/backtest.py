"""Portfolio-level backtest using the compatibility engine."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from src.backtest.transaction_costs import TransactionCostParams

from .allocator import allocate_top_n, apply_constraints
from .signal_ranker import rank_signals


def run_portfolio_backtest(
    daily_long: pd.DataFrame,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
    n_top: int = 5,
    max_per_stock: float = 0.25,
    max_per_industry: float = 0.40,
    max_daily_turnover: float = 0.50,
    min_volume_amount: int = 100_000_000,
    industry_map: dict[str, str] | None = None,
    cost_params: TransactionCostParams | None = None,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """Run a portfolio-level backtest on the watchlist cross-section.

    Parameters
    ----------
    daily_long:
        Long-format OHLCV with columns ``trade_date``, ``symbol``, ``open``,
        ``high``, ``low``, ``close``, ``volume``.
    n_top:
        Maximum number of concurrent positions.
    max_per_stock:
        Maximum weight per stock.
    min_volume_amount:
        Minimum 20-day average daily amount (yuan) for a stock to be eligible.
    """
    del initial_capital

    df = daily_long.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)

    # Build volume-wide for liquidity filter
    volume_wide = df.pivot(index=date_col, columns=sym_col, values="volume").sort_index().astype(np.float64)

    # Rank signals
    scores = rank_signals(df, index_ohlcv=index_ohlcv, date_col=date_col, sym_col=sym_col)

    # Allocate
    weights = allocate_top_n(
        scores,
        n_top=n_top,
        max_per_stock=max_per_stock,
        min_score=0.0,
        daily_long=df,
        volume_wide=volume_wide,
        min_volume_rank=min_volume_amount,
        date_col=date_col,
        sym_col=sym_col,
    )

    weights = apply_constraints(
        weights,
        max_positions=n_top,
        max_per_stock=max_per_stock,
        max_per_industry=max_per_industry,
        industry_map=industry_map,
        max_daily_turnover=max_daily_turnover,
    )

    if weights.empty:
        return {
            "scores": scores,
            "weights": weights,
            "backtest": None,
            "summary": {"status": "no_valid_dates"},
        }

    # Build returns from the engine's open-to-open logic
    from src.backtest.engine import build_open_to_open_returns
    returns_wide = build_open_to_open_returns(
        df, date_col=date_col, sym_col=sym_col, zero_if_limit_up_open=True,
    )

    # Align weights to return dates
    common_dates = weights.index.intersection(returns_wide.index)
    weights_aligned = weights.reindex(common_dates)
    returns_aligned = returns_wide.reindex(common_dates)

    cfg = BacktestConfig(
        cost_params=cost_params,
        risk_free_daily=0.0,
        periods_per_year=252.0,
        max_gross_exposure=1.0,
        execution_mode="tplus1_open",
        execution_lag=0,
    )

    bt_result = run_backtest(returns_aligned, weights_aligned, config=cfg)

    return {
        "scores": scores,
        "weights": weights_aligned,
        "backtest": bt_result,
        "summary": _portfolio_summary(bt_result, weights_aligned),
    }


def _portfolio_summary(
    bt_result: BacktestResult,
    weights: pd.DataFrame,
) -> dict[str, Any]:
    panel = bt_result.panel
    n_dates = len(weights)
    avg_positions = float(weights.astype(bool).sum(axis=1).mean()) if n_dates > 0 else 0.0
    return {
        "total_return": panel.total_return,
        "annualized_return": panel.annualized_return,
        "sharpe_ratio": panel.sharpe_ratio,
        "max_drawdown": panel.max_drawdown,
        "calmar_ratio": panel.calmar_ratio,
        "n_rebalance_dates": n_dates,
        "avg_positions": avg_positions,
        "n_trading_days": int((~bt_result.daily_returns.isna()).sum()) if bt_result.daily_returns is not None else 0,
    }
