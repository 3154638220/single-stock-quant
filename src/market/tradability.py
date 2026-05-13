"""
A 股可交易性：涨跌停比例、次日开盘一字涨停（难以买入）、停牌近似。

单股信号回测用这些规则处理 T+1 买入遇到一字涨停时的顺延。
"""

from __future__ import annotations

import numpy as np


def limit_up_ratio(symbol: str) -> float:
    """
    普通股涨跌幅限制比例（不含 ST）。
    创业板/科创板 20%，北交所 30%，其余主板 10%。
    """
    s = str(symbol).zfill(6)
    if s.startswith(("300", "688")):
        return 0.20
    if s.startswith(("8", "4")):
        return 0.30
    return 0.10


def limit_up_px(prev_close: float, symbol: str) -> float:
    """涨停价（简化：未做 tick 舍入，与行情比较时用相对容差）。"""
    pc = float(prev_close)
    r = limit_up_ratio(symbol)
    return pc * (1.0 + r)


def is_open_limit_up_unbuyable(
    open_px: float,
    prev_close: float,
    symbol: str,
    *,
    rel_tol: float = 1e-4,
) -> bool:
    """一字涨停开盘：开盘价触及涨停价（约等于），散户无法按开盘价成交。"""
    if not np.isfinite(open_px) or not np.isfinite(prev_close) or prev_close <= 0:
        return True
    lim = limit_up_px(prev_close, symbol)
    return open_px >= lim * (1.0 - rel_tol)


def is_row_suspended_like(
    volume: float,
    open_px: float,
    close_px: float,
) -> bool:
    """停牌近似：无成交量或 OHLC 无效。"""
    if not np.isfinite(open_px) or not np.isfinite(close_px):
        return True
    if not np.isfinite(volume) or volume <= 0:
        return True
    return False


def limit_down_ratio(symbol: str) -> float:
    """跌停幅度比例（与涨停对称）。"""
    return limit_up_ratio(symbol)


def limit_down_px(prev_close: float, symbol: str) -> float:
    """跌停价。"""
    pc = float(prev_close)
    r = limit_down_ratio(symbol)
    return pc * (1.0 - r)


def is_open_limit_down_unsellable(
    open_px: float,
    prev_close: float,
    symbol: str,
    *,
    rel_tol: float = 1e-4,
) -> bool:
    """一字跌停开盘：开盘价触及跌停价，散户无法按开盘价卖出。"""
    if not np.isfinite(open_px) or not np.isfinite(prev_close) or prev_close <= 0:
        return True
    lim = limit_down_px(prev_close, symbol)
    return open_px <= lim * (1.0 + rel_tol)
