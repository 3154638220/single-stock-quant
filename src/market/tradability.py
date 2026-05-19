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


import pandas as pd


def is_tradable_open(df: pd.DataFrame, idx: int) -> bool:
    """Return True if the bar at idx is a normal trading day (not suspended)."""
    if idx < 0 or idx >= len(df):
        return False
    volume = float(df.loc[idx, "volume"]) if "volume" in df.columns else 1.0
    open_px = float(df.loc[idx, "open"])
    close_px = float(df.loc[idx, "close"])
    return not is_row_suspended_like(volume, open_px, close_px)


def next_buy_index(df: pd.DataFrame, symbol: str, start_idx: int) -> int | None:
    """Find the next tradable day from start_idx that is not limit-up at open."""
    for j in range(start_idx, len(df)):
        if not is_tradable_open(df, j):
            continue
        prev_close = float(df.loc[j - 1, "close"]) if j > 0 else float("nan")
        open_px = float(df.loc[j, "open"])
        if not is_open_limit_up_unbuyable(open_px, prev_close, symbol):
            return j
    return None


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
