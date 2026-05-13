"""Small compatibility backtest engine for daily return series.

The production single-stock path lives in ``src.backtest.single_stock``. This
module remains as a lightweight helper for tests, notebooks, and any local code
that still wants to evaluate a precomputed daily return matrix with simple
long-only weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest.performance_panel import PerformancePanel, compute_performance_panel
from src.backtest.transaction_costs import TransactionCostParams, turnover_cost_drag


@dataclass
class TieredImpactConfig:
    """Deprecated placeholder kept for old imports."""

    large_cap_threshold: float = 500_000_000.0
    mid_cap_threshold: float = 100_000_000.0
    large_slippage_bps: float = 1.5
    large_impact_bps: float = 4.0
    mid_slippage_bps: float = 3.0
    mid_impact_bps: float = 8.0
    small_slippage_bps: float = 5.0
    small_impact_bps: float = 20.0

    def get_params(self, amount_20d: float) -> tuple[float, float]:
        a = float(amount_20d)
        if not np.isfinite(a) or a <= 0:
            return (self.small_slippage_bps, self.small_impact_bps)
        if a >= self.large_cap_threshold:
            return (self.large_slippage_bps, self.large_impact_bps)
        if a >= self.mid_cap_threshold:
            return (self.mid_slippage_bps, self.mid_impact_bps)
        return (self.small_slippage_bps, self.small_impact_bps)


@dataclass
class BacktestConfig:
    cost_params: Optional[TransactionCostParams] = None
    risk_free_daily: float = 0.0
    periods_per_year: float = 252.0
    max_gross_exposure: float = 1.0
    execution_mode: str = "close_to_close"
    execution_lag: int = 0
    rebalance_rule: str = ""
    risk_cfg: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    rebalance_turnover: pd.Series
    panel: PerformancePanel
    meta: Dict[str, Any] = field(default_factory=dict)


def _align_weights_columns(weights: pd.DataFrame, asset_cols: List[str]) -> pd.DataFrame:
    return weights.reindex(columns=asset_cols, fill_value=0.0).astype(np.float64)


def _normalize_long_only(row: np.ndarray, max_gross_exposure: float) -> np.ndarray:
    w = np.asarray(row, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= 0:
        return np.zeros_like(w, dtype=np.float64)
    cap = float(max_gross_exposure)
    if cap <= 0:
        return np.zeros_like(w, dtype=np.float64)
    return w / total * min(total, cap)


def build_daily_weights(
    trading_index: pd.DatetimeIndex,
    weights_rebalance: pd.DataFrame,
    *,
    max_gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Forward-fill rebalance weights over a full trading-day index."""
    if weights_rebalance.empty:
        raise ValueError("weights_rebalance is empty")
    ti = pd.DatetimeIndex(pd.to_datetime(trading_index).normalize())
    wr = weights_rebalance.sort_index().copy()
    wr.index = pd.to_datetime(wr.index).normalize()
    wr = wr[wr.index.isin(ti)]
    if wr.empty:
        raise ValueError("rebalance dates are not present in trading_index")

    out = pd.DataFrame(index=ti, columns=wr.columns, dtype=np.float64)
    current: np.ndarray | None = None
    for dt in ti:
        if dt in wr.index:
            row = wr.loc[dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            current = _normalize_long_only(row.to_numpy(dtype=np.float64), max_gross_exposure)
        if current is None:
            current = np.zeros(len(wr.columns), dtype=np.float64)
        out.loc[dt] = current
    return out


def build_limit_up_open_mask(
    daily_long: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
) -> pd.DataFrame:
    """Build a wide mask for open limit-up days from an OHLCV long table."""
    from src.market.tradability import is_open_limit_up_unbuyable

    if daily_long.empty:
        raise ValueError("daily_long is empty")
    missing = {date_col, sym_col, "open", "close"} - set(daily_long.columns)
    if missing:
        raise ValueError(f"daily_long missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    df = daily_long.copy()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    for sym, g in df.groupby(sym_col, sort=False):
        g = g.sort_values(date_col)
        prev_close = pd.to_numeric(g.get("pre_close", g["close"].shift(1)), errors="coerce")
        open_px = pd.to_numeric(g["open"], errors="coerce")
        for dt, o, pc in zip(g[date_col], open_px, prev_close):
            mask = bool(np.isfinite(o) and np.isfinite(pc) and is_open_limit_up_unbuyable(float(o), float(pc), sym))
            rows.append({date_col: dt, sym_col: sym, "_limit_up_open": mask})
    long_mask = pd.DataFrame(rows)
    return long_mask.pivot(index=date_col, columns=sym_col, values="_limit_up_open").fillna(False).astype(bool)


def build_open_to_open_returns(
    daily_long: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
    zero_if_limit_up_open: bool = False,
) -> pd.DataFrame:
    """Build open(t+1) / open(t) - 1 returns from an OHLCV long table."""
    if daily_long.empty:
        raise ValueError("daily_long is empty")
    missing = {date_col, sym_col, "open"} - set(daily_long.columns)
    if missing:
        raise ValueError(f"daily_long missing columns: {sorted(missing)}")

    df = daily_long.copy()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    limit_mask = build_limit_up_open_mask(df, date_col=date_col, sym_col=sym_col) if zero_if_limit_up_open else None
    rows: list[dict[str, Any]] = []
    for sym, g in df.groupby(sym_col, sort=False):
        g = g.sort_values(date_col)
        o = pd.to_numeric(g["open"], errors="coerce")
        ret = o.shift(-1) / o - 1.0
        if limit_mask is not None and sym in limit_mask.columns:
            mask = limit_mask.reindex(g[date_col])[sym].fillna(False).to_numpy(bool)
            ret = ret.mask(mask, 0.0)
        rows.append(pd.DataFrame({date_col: g[date_col].values, sym_col: sym, "_ret": ret.values}))
    long_ret = pd.concat(rows, ignore_index=True)
    return long_ret.pivot(index=date_col, columns=sym_col, values="_ret").sort_index().astype(np.float64)


def run_backtest(
    asset_returns: pd.DataFrame,
    weights_signal: pd.DataFrame,
    *,
    config: Optional[BacktestConfig] = None,
    rebalance_rule: Optional[str] = None,
    daily_amount: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Evaluate precomputed daily returns with forward-filled long-only weights."""
    del daily_amount
    cfg = config or BacktestConfig()
    exe = str(cfg.execution_mode).lower().strip()
    if exe not in ("close_to_close", "tplus1_open"):
        raise ValueError("execution_mode must be close_to_close or tplus1_open")
    lag = 0 if exe == "tplus1_open" else max(0, int(cfg.execution_lag))

    ar = asset_returns.sort_index().copy()
    ar.index = pd.to_datetime(ar.index).normalize()
    ar = ar.astype(np.float64)
    ws = weights_signal.sort_index().copy()
    ws.index = pd.to_datetime(ws.index).normalize()
    ws = _align_weights_columns(ws, list(ar.columns))

    rule = rebalance_rule if rebalance_rule is not None else cfg.rebalance_rule
    if rule:
        ws = ws.resample(rule).last().dropna(how="all")
        ws = ws[ws.index.isin(ar.index)]

    daily_w = build_daily_weights(ar.index, ws, max_gross_exposure=cfg.max_gross_exposure)
    r_mat = ar.fillna(0.0).to_numpy(dtype=np.float64)
    w_mat = daily_w.to_numpy(dtype=np.float64)
    n = len(ar)
    port = np.zeros(n, dtype=np.float64)
    turn = np.full(n, np.nan, dtype=np.float64)

    for i in range(1, n):
        j = i - 1 - lag
        if j >= 0:
            port[i] = float(np.dot(w_mat[j], r_mat[i]))
        half_l1 = 0.5 * float(np.sum(np.abs(w_mat[i] - w_mat[i - 1])))
        if half_l1 > 1e-15:
            turn[i] = half_l1
            if cfg.cost_params is not None:
                port[i] -= turnover_cost_drag(half_l1, cfg.cost_params)

    s = pd.Series(port, index=ar.index, name="strategy_ret")
    turnover = pd.Series(turn, index=ar.index, name="turnover_half_l1")
    panel = compute_performance_panel(
        s.to_numpy(dtype=np.float64),
        turnover=turnover.to_numpy(dtype=np.float64),
        risk_free_daily=cfg.risk_free_daily,
        periods_per_year=cfg.periods_per_year,
    )
    return BacktestResult(
        daily_returns=s,
        rebalance_turnover=turnover,
        panel=panel,
        meta={
            "n_rebalances": int(len(ws.index.intersection(ar.index))),
            "symbols": list(ar.columns),
            "execution_mode": exe,
            "execution_lag": lag,
            "compatibility_engine": True,
        },
    )


def result_to_dict(res: BacktestResult) -> Dict[str, Any]:
    return {
        "panel": res.panel.to_dict(),
        "meta": res.meta,
        "daily_returns": res.daily_returns,
        "rebalance_turnover": res.rebalance_turnover,
    }
