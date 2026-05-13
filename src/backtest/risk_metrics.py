"""Risk metrics for strategy return series."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    """
    简单收益序列上的最大回撤（非负）。

    注意：此函数仅适用于非杠杆收益序列（收益 > -1.0 的常规多头策略）。
    若策略含杠杆，equity 可能为负，回撤计算需额外保护。
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / np.maximum(peak, 1e-15)
    return float(np.max(dd))


def realized_volatility(
    returns: np.ndarray,
    *,
    periods_per_year: float = 252.0,
) -> float:
    """已实现波动率（年化）。"""
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def drawdown_alert(
    max_dd: float,
    threshold: float,
) -> Tuple[bool, str]:
    """是否触发最大回撤预警。"""
    if not np.isfinite(max_dd) or not np.isfinite(threshold):
        return False, ""
    if max_dd >= threshold:
        return True, f"max_drawdown {max_dd:.4f} >= alert {threshold:.4f}"
    return False, ""


def volatility_alert(
    vol_ann: float,
    max_vol_ann: float,
) -> Tuple[bool, str]:
    """是否触发波动率上限预警。"""
    if not np.isfinite(vol_ann) or not np.isfinite(max_vol_ann):
        return False, ""
    if vol_ann >= max_vol_ann:
        return True, f"vol_ann {vol_ann:.4f} >= cap {max_vol_ann:.4f}"
    return False, ""


def index_cumulative_return(
    daily_df: pd.DataFrame,
    *,
    symbol: str,
    end_date: pd.Timestamp,
    lookback_trading_days: int,
    price_col: str = "close",
    date_col: str = "trade_date",
    sym_col: str = "symbol",
) -> Optional[float]:
    """
    指数在 ``lookback_trading_days`` 根 **交易日** 上的累计简单收益（含当日收盘相对 lookback 前一日收盘）。

    数据不足返回 ``None``。
    """
    if lookback_trading_days < 1:
        return None
    sub = daily_df[daily_df[sym_col].astype(str) == str(symbol)].copy()
    if sub.empty:
        return None
    sub[date_col] = pd.to_datetime(sub[date_col]).dt.normalize()
    sub = sub.sort_values(date_col)
    t_end = pd.Timestamp(end_date).normalize()
    idx = sub[date_col].searchsorted(t_end, side="left")
    if idx >= len(sub) or sub.iloc[idx][date_col] != t_end:
        return None
    j0 = idx - lookback_trading_days
    if j0 < 0:
        return None
    p0 = float(sub.iloc[j0][price_col])
    p1 = float(sub.iloc[idx][price_col])
    if not np.isfinite(p0) or p0 == 0.0:
        return None
    return p1 / p0 - 1.0


def risk_off_multiplier_from_index(
    daily_df: pd.DataFrame,
    *,
    benchmark_symbol: str,
    asof: pd.Timestamp,
    lookback_trading_days: int,
    drop_threshold: float,
    risk_off_factor: float,
) -> Tuple[float, Optional[float], str]:
    """
    极端行情：若指数在 lookback 内累计收益低于 ``-drop_threshold``，
    将仓位乘子设为 ``risk_off_factor``（0 表示空仓规则，1 表示不调整）。

    Returns
    -------
    multiplier : float
    index_ret : float or None
    note : str
    """
    r = index_cumulative_return(
        daily_df,
        symbol=benchmark_symbol,
        end_date=asof,
        lookback_trading_days=lookback_trading_days,
    )
    if r is None:
        return 1.0, None, "no_benchmark_data"
    if r <= -float(drop_threshold):
        return float(risk_off_factor), r, f"extreme_drop index_ret={r:.6f}"
    return 1.0, r, "ok"


def risk_config_from_mapping(m: Mapping[str, Any]) -> Dict[str, Any]:
    d = dict(m) if isinstance(m, dict) else {}
    return {
        "max_drawdown_alert": float(d.get("max_drawdown_alert", 0.15)),
        "max_volatility_ann": float(d.get("max_volatility_ann", 0.55)),
        "benchmark_symbol": str(d.get("benchmark_symbol", "510300")),
        "extreme_lookback_days": int(d.get("extreme_lookback_days", 5)),
        "extreme_drop_threshold": float(d.get("extreme_drop_threshold", 0.05)),
        "risk_off_factor": float(d.get("risk_off_factor", 0.0)),
    }
