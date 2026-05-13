"""Single-stock DK trend backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.performance_panel import compute_performance_panel
from src.indicators import DKTrendParams, compute_dktrend
from src.market.tradability import is_open_limit_up_unbuyable, is_row_suspended_like


@dataclass(frozen=True)
class SingleStockBacktestResult:
    symbol: str
    stock_name: str
    period: str
    n_trades: int
    win_rate: float
    avg_hold_days: float
    avg_return_per_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_return: float
    annualized_return: float
    buy_hold_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    trade_log: pd.DataFrame
    daily_returns: pd.Series


def _prepare_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    need = {"open", "close", "high", "low"}
    missing = need - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv missing required columns: {sorted(missing)}")
    df = ohlcv.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    else:
        df["trade_date"] = pd.to_datetime(df.index).normalize()
    df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    for col in ["open", "close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("ohlcv must contain at least two valid bars")
    return df


def _is_tradable_open(df: pd.DataFrame, idx: int) -> bool:
    volume = float(df.loc[idx, "volume"]) if "volume" in df.columns else 1.0
    open_px = float(df.loc[idx, "open"])
    close_px = float(df.loc[idx, "close"])
    return not is_row_suspended_like(volume, open_px, close_px)


def _next_buy_index(df: pd.DataFrame, symbol: str, start_idx: int) -> int | None:
    for j in range(start_idx, len(df)):
        if not _is_tradable_open(df, j):
            continue
        prev_close = float(df.loc[j - 1, "close"]) if j > 0 else np.nan
        open_px = float(df.loc[j, "open"])
        if not is_open_limit_up_unbuyable(open_px, prev_close, symbol):
            return j
    return None


def _next_sell_index(df: pd.DataFrame, start_idx: int) -> int | None:
    for j in range(start_idx, len(df)):
        if _is_tradable_open(df, j):
            return j
    return None


def _max_consecutive(values: list[bool], target: bool) -> int:
    best = cur = 0
    for v in values:
        if v is target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def run_single_stock_backtest(
    symbol: str,
    ohlcv: pd.DataFrame,
    params: DKTrendParams,
    *,
    cost_bps: float = 15.0,
    initial_capital: float = 100_000.0,
    stock_name: str = "",
) -> SingleStockBacktestResult:
    """Backtest one stock with T+1 open execution and single-position long/flat state."""
    code = str(symbol).strip().zfill(6)
    df = _prepare_ohlcv(ohlcv)
    trend = compute_dktrend(df, params).reset_index(drop=True)
    cost = max(float(cost_bps), 0.0) * 1e-4
    actions: dict[int, str] = {}
    planned_long = False
    pending_buy_idx: int | None = None
    for i, row in trend.iterrows():
        if pending_buy_idx is not None and pending_buy_idx <= i:
            planned_long = True
            pending_buy_idx = None
        sig = str(row.get("dk_signal", ""))
        if sig == "buy" and not planned_long and pending_buy_idx is None and i + 1 < len(df):
            j = _next_buy_index(df, code, i + 1)
            if j is None:
                continue
            actions.setdefault(j, "buy")
            pending_buy_idx = j
        elif sig == "sell" and planned_long and i + 1 < len(df):
            j = _next_sell_index(df, i + 1)
            if j is None:
                continue
            actions.setdefault(j, "sell")
            planned_long = False
        elif sig == "sell" and pending_buy_idx is not None:
            actions.pop(pending_buy_idx, None)
            pending_buy_idx = None

    cash = float(initial_capital)
    shares = 0.0
    in_pos = False
    entry_date = pd.NaT
    entry_price = float("nan")
    entry_cash = float("nan")
    trades: list[dict] = []
    equity = np.zeros(len(df), dtype=np.float64)

    for i in range(len(df)):
        action = actions.get(i)
        if action == "buy" and not in_pos:
            price = float(df.loc[i, "open"])
            if price > 0 and np.isfinite(price):
                shares = cash * (1.0 - cost) / price
                entry_cash = cash
                cash = 0.0
                in_pos = True
                entry_date = pd.Timestamp(df.loc[i, "trade_date"])
                entry_price = price
        elif action == "sell" and in_pos:
            price = float(df.loc[i, "open"])
            if price <= 0 or not np.isfinite(price):
                continue
            cash = shares * price * (1.0 - cost)
            shares = 0.0
            exit_date = pd.Timestamp(df.loc[i, "trade_date"])
            ret = cash / entry_cash - 1.0
            trades.append(
                {
                    "buy_date": entry_date,
                    "sell_date": exit_date,
                    "buy_price": entry_price,
                    "sell_price": price,
                    "hold_days": int((exit_date - entry_date).days),
                    "return": ret,
                    "exit_reason": "signal",
                }
            )
            in_pos = False
        equity[i] = shares * float(df.loc[i, "close"]) if in_pos else cash

    last_idx = len(df) - 1
    if in_pos:
        price = float(df.loc[last_idx, "close"])
        cash = shares * price * (1.0 - cost)
        shares = 0.0
        exit_date = pd.Timestamp(df.loc[last_idx, "trade_date"])
        ret = cash / entry_cash - 1.0
        trades.append(
            {
                "buy_date": entry_date,
                "sell_date": exit_date,
                "buy_price": entry_price,
                "sell_price": price,
                "hold_days": int((exit_date - entry_date).days),
                "return": ret,
                "exit_reason": "end",
            }
        )
        in_pos = False
    equity[last_idx] = cash
    for k in range(1, len(equity)):
        if equity[k] == 0:
            equity[k] = equity[k - 1]
    daily_returns = pd.Series(equity, index=pd.to_datetime(df["trade_date"]), name="equity").pct_change().fillna(0.0)
    daily_returns.name = "strategy_ret"
    panel = compute_performance_panel(daily_returns.to_numpy(dtype=np.float64))
    trade_log = pd.DataFrame(trades)
    trade_returns = trade_log["return"].to_numpy(dtype=np.float64) if not trade_log.empty else np.array([])
    wins = [bool(x > 0) for x in trade_returns]
    buy_hold_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)
    start_s = pd.Timestamp(df["trade_date"].iloc[0]).date().isoformat()
    end_s = pd.Timestamp(df["trade_date"].iloc[-1]).date().isoformat()
    return SingleStockBacktestResult(
        symbol=code,
        stock_name=stock_name or code,
        period=f"{start_s} ~ {end_s}",
        n_trades=int(len(trade_log)),
        win_rate=float(np.mean(trade_returns > 0)) if trade_returns.size else float("nan"),
        avg_hold_days=float(trade_log["hold_days"].mean()) if not trade_log.empty else float("nan"),
        avg_return_per_trade=float(np.mean(trade_returns)) if trade_returns.size else float("nan"),
        max_consecutive_wins=_max_consecutive(wins, True),
        max_consecutive_losses=_max_consecutive(wins, False),
        total_return=panel.total_return,
        annualized_return=panel.annualized_return,
        buy_hold_return=buy_hold_return,
        sharpe_ratio=panel.sharpe_ratio,
        max_drawdown=panel.max_drawdown,
        calmar_ratio=panel.calmar_ratio,
        trade_log=trade_log,
        daily_returns=daily_returns,
    )
