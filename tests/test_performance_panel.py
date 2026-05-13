import numpy as np

from src.backtest.performance_panel import compute_performance_panel, deflated_sharpe_ratio


def test_compute_performance_panel_populates_dsr():
    returns = np.array([0.01, -0.004, 0.006, 0.003, -0.002, 0.008], dtype=float)

    panel = compute_performance_panel(returns, n_concurrent_strategies=3)

    assert 0.0 <= panel.dsr <= 1.0
    assert 0.0 <= panel.dsr_pvalue <= 1.0


def test_deflated_sharpe_penalizes_multiple_trials():
    returns = np.array([0.01, -0.004, 0.006, 0.003, -0.002, 0.008], dtype=float)

    single, _ = deflated_sharpe_ratio(returns, n_trials=1)
    many, _ = deflated_sharpe_ratio(returns, n_trials=20)

    assert many < single
