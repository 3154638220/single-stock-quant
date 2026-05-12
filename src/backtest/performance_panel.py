"""Performance metrics: annual return, Sharpe, Calmar, drawdown and win rate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np

from src.backtest.risk_metrics import max_drawdown_from_returns


@dataclass(frozen=True)
class PerformancePanel:
    """Unified metric panel for backtests."""

    annualized_return: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    turnover_mean: float
    n_periods: int
    total_return: float
    periods_per_year: float
    dsr: float = float("nan")
    dsr_pvalue: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_returns(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64).ravel()
    return r[np.isfinite(r)]  # type: ignore[no-any-return]


def total_return_from_simple_returns(returns: np.ndarray) -> float:
    """简单收益序列的复利总收益：prod(1+r)-1。"""
    r = _finite_returns(returns)
    if r.size == 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)  # type: ignore[no-any-return]


def annualized_return_cagr(
    returns: np.ndarray,
    *,
    periods_per_year: float = 252.0,
) -> float:
    """
    由日（或 bar）简单收益序列推算年化复合收益：(1+R)^{PY/n}-1。
    """
    r = _finite_returns(returns)
    n = r.size
    if n == 0:
        return float("nan")
    cum = float(np.prod(1.0 + r))
    if cum <= 0:
        return float("nan")
    return float(cum ** (periods_per_year / n) - 1.0)


def sharpe_ratio(
    returns: np.ndarray,
    *,
    risk_free_daily: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """超额收益夏普（简单收益、样本标准差）。"""
    r = _finite_returns(returns)
    if r.size < 2:
        return float("nan")
    ex = r - float(risk_free_daily)
    mu = float(np.mean(ex))
    sd = float(np.std(ex, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(mu / sd * np.sqrt(periods_per_year))


def calmar_ratio(
    annualized_return: float,
    max_drawdown: float,
) -> float:
    """年化收益 / 最大回撤（回撤为正值）。"""
    if not np.isfinite(annualized_return) or not np.isfinite(max_drawdown):
        return float("nan")
    if max_drawdown <= 1e-15:
        return float("nan")
    return float(annualized_return / max_drawdown)


def win_rate(returns: np.ndarray) -> float:
    """正收益 bar 占比。"""
    r = _finite_returns(returns)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0.0))


def compute_performance_panel(
    returns: np.ndarray,
    *,
    turnover: Optional[np.ndarray] = None,
    risk_free_daily: float = 0.0,
    periods_per_year: float = 252.0,
    n_concurrent_strategies: int = 1,
) -> PerformancePanel:
    """
    由单条收益序列（通常为日收益）计算统一绩效面板。

    Parameters
    ----------
    returns
        简单收益序列（如日度）。
    turnover
        可选，与 ``returns`` 对齐或可聚合的换手序列；若提供则取 ``nanmean`` 为 turnover_mean，
        否则 turnover_mean 为 nan。
    n_concurrent_strategies
        Kept for API compatibility. No extra statistical test is computed.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r_fin = _finite_returns(r)
    n = int(r_fin.size)
    tot = total_return_from_simple_returns(r_fin)
    ann = annualized_return_cagr(r_fin, periods_per_year=periods_per_year)
    mdd = max_drawdown_from_returns(r_fin)
    sh = sharpe_ratio(r_fin, risk_free_daily=risk_free_daily, periods_per_year=periods_per_year)
    cal = calmar_ratio(ann, mdd)
    wr = win_rate(r_fin)

    if turnover is not None:
        t = np.asarray(turnover, dtype=np.float64).ravel()
        t = t[np.isfinite(t)]
        t_mean = float(np.nanmean(t)) if t.size else float("nan")
    else:
        t_mean = float("nan")

    dsr = float("nan")
    dsr_pvalue = float("nan")

    return PerformancePanel(
        annualized_return=ann,
        sharpe_ratio=sh,
        calmar_ratio=cal,
        max_drawdown=mdd,
        win_rate=wr,
        turnover_mean=t_mean,
        n_periods=n,
        total_return=tot,
        periods_per_year=float(periods_per_year),
        dsr=dsr,
        dsr_pvalue=dsr_pvalue,
    )


def aggregate_panels(
    panels: list[PerformancePanel],
    *,
    method: str = "mean",
) -> dict[str, Any]:
    """Aggregate multiple performance panels by mean or median."""
    if not panels:
        return {"n_folds": 0}
    keys = (
        "annualized_return",
        "sharpe_ratio",
        "calmar_ratio",
        "max_drawdown",
        "win_rate",
        "turnover_mean",
        "total_return",
    )
    method = str(method).lower()
    agg: dict[str, Any] = {"n_folds": len(panels), "method": method}
    for k in keys:
        vals = [getattr(p, k) for p in panels]
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            agg[f"{k}_agg"] = float("nan")
        elif method == "median":
            agg[f"{k}_agg"] = float(np.median(arr))
        else:
            agg[f"{k}_agg"] = float(np.mean(arr))
    return agg


def panel_from_mapping(m: Mapping[str, Any]) -> PerformancePanel:
    """从字典构造 PerformancePanel（便于序列化往返）。"""
    return PerformancePanel(
        annualized_return=float(m["annualized_return"]),
        sharpe_ratio=float(m["sharpe_ratio"]),
        calmar_ratio=float(m["calmar_ratio"]),
        max_drawdown=float(m["max_drawdown"]),
        win_rate=float(m["win_rate"]),
        turnover_mean=float(m["turnover_mean"]),
        n_periods=int(m["n_periods"]),
        total_return=float(m["total_return"]),
        periods_per_year=float(m["periods_per_year"]),
    )
