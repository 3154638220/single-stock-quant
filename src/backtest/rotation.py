"""Multi-stock rotation backtest (Section 7, X1).

Abandons the single-stock/single-parameter-set WFO framework in favour of
running N stocks simultaneously, holding the 1-2 strongest signals each week.
This increases the effective OOS trade count from 3-39 to 20-80 and lowers
single-stock concentration risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.performance_panel import annualized_return_cagr, compute_performance_panel
from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.indicators.donchian import compute_donchian_trend
from src.market.tradability import is_open_limit_up_unbuyable, is_row_suspended_like
from src.signals.generator import apply_volume_confirmation


@dataclass
class RotationResult:
    symbols: list[str]
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    n_trades: int
    n_rotations: int
    daily_returns: pd.Series
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    turnover_pct: float


def _is_tradable_open(df: pd.DataFrame, idx: int) -> bool:
    if idx < 0 or idx >= len(df):
        return False
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


def _compute_trend(
    df: pd.DataFrame,
    params: DKTrendParams,
    volume_confirm: bool = True,
    volume_lookback: int = 20,
    volume_ratio_min: float = 1.0,
) -> pd.DataFrame:
    """Compute trend signal for one stock's OHLCV DataFrame."""
    if params.mode == TrendMode.DONCHIAN_BREAKOUT:
        trend = compute_donchian_trend(
            df,
            entry_window=params.donchian_entry_window,
            exit_window=params.donchian_exit_window,
            min_run_len=params.min_run_len,
        ).reset_index(drop=True)
    else:
        trend = compute_dktrend(df, params).reset_index(drop=True)
    trend = apply_volume_confirmation(
        trend,
        enabled=volume_confirm,
        lookback=volume_lookback,
        volume_ratio_min=volume_ratio_min,
    ).reset_index(drop=True)
    return trend


def _trend_strength_score(trend: pd.DataFrame, idx: int) -> float:
    """Composite score for ranking: trend strength * persistence."""
    if idx < 0 or idx >= len(trend):
        return -999.0
    color = str(trend.loc[idx, "dk_color"])
    if color != "red":
        return -1.0  # bearish stocks get negative score
    dk_val = float(trend.loc[idx, "dk_value"]) if "dk_value" in trend.columns else 0.5
    run_len = int(trend.loc[idx, "dk_run_len"]) if "dk_run_len" in trend.columns else 1
    return dk_val * (1.0 + run_len / 10.0)


def _check_position_exit(
    df: pd.DataFrame,
    i: int,
    entry_price: float,
    highest_close: float,
    stop_loss_pct: float,
    atr_trailing_mult: float,
    atr_trailing_min_gain: float,
    atr_series: pd.Series,
    intrapos_dd_limit: float,
    profit_lock_trigger: float,
    profit_lock_trailing: float,
    profit_lock_active: bool,
    profit_lock_high: float,
    time_stop_days: int,
    time_stop_min_return: float,
    entry_date: pd.Timestamp,
) -> tuple[str, bool, float]:
    """Check all exit conditions for a position. Returns (reason, profit_lock_active, profit_lock_high)."""
    close_px = float(df.loc[i, "close"])
    if not np.isfinite(close_px) or not np.isfinite(entry_price) or entry_price <= 0:
        return ("", profit_lock_active, profit_lock_high)

    pnl = close_px / entry_price - 1.0

    # Stop loss
    if stop_loss_pct > 0 and pnl < -stop_loss_pct:
        return ("stop_loss", profit_lock_active, profit_lock_high)

    # ATR trailing stop
    if atr_trailing_mult > 0 and pnl >= atr_trailing_min_gain and i < len(atr_series):
        atr_now = float(atr_series.iloc[i])
        if np.isfinite(atr_now) and atr_now > 0:
            if close_px < highest_close - atr_trailing_mult * atr_now:
                return ("atr_trailing_stop", profit_lock_active, profit_lock_high)

    # Intra-position drawdown stop
    if intrapos_dd_limit > 0 and np.isfinite(highest_close) and highest_close > 0:
        dd = (highest_close - close_px) / highest_close
        if dd > intrapos_dd_limit:
            return ("intrapos_dd_stop", profit_lock_active, profit_lock_high)

    # Profit lock
    if profit_lock_trigger > 0 and profit_lock_trailing > 0:
        if not profit_lock_active and pnl >= profit_lock_trigger:
            profit_lock_active = True
            profit_lock_high = close_px
        if profit_lock_active:
            profit_lock_high = max(profit_lock_high, close_px)
            if close_px / profit_lock_high - 1.0 < -profit_lock_trailing:
                return ("profit_lock", profit_lock_active, profit_lock_high)

    # Time stop
    if time_stop_days > 0:
        hold_days = (pd.Timestamp(df.loc[i, "trade_date"]) - entry_date).days
        if hold_days >= time_stop_days and pnl < time_stop_min_return:
            return ("time_stop", profit_lock_active, profit_lock_high)

    return ("", profit_lock_active, profit_lock_high)


def run_rotation_backtest(
    ohlcv_map: dict[str, pd.DataFrame],
    *,
    trend_params: DKTrendParams | None = None,
    top_n: int = 2,
    rebalance_freq: int = 5,
    ranking_mode: str = "trend_strength",
    volume_confirm: bool = True,
    volume_lookback: int = 20,
    volume_ratio_min: float = 1.0,
    stop_loss_pct: float = 0.08,
    atr_trailing_mult: float = 2.0,
    atr_trailing_min_gain: float = 0.05,
    intrapos_dd_limit: float = 0.15,
    profit_lock_trigger: float = 0.12,
    profit_lock_trailing: float = 0.05,
    time_stop_days: int = 30,
    time_stop_min_return: float = 0.03,
    cost_bps: float = 15.0,
    initial_capital: float = 100_000.0,
    min_bars_required: int = 100,
) -> RotationResult:
    """Run multi-stock rotation backtest.

    Parameters
    ----------
    ohlcv_map:
        Dict mapping symbol string to OHLCV DataFrame (must contain trade_date,
        open, high, low, close, volume).
    top_n:
        Number of stocks to hold simultaneously.
    rebalance_freq:
        Rebalance every N trading days (5 = weekly).
    ranking_mode:
        How to rank stocks: "trend_strength" or "rs_momentum".
    """
    params = trend_params or DKTrendParams(mode=TrendMode.DONCHIAN_BREAKOUT)
    symbols = sorted(ohlcv_map.keys())
    if len(symbols) < top_n:
        raise ValueError(f"Need at least {top_n} symbols, got {len(symbols)}")

    # ---- align all DataFrames to a common date index ----
    aligned: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = ohlcv_map[sym].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
        if len(df) >= min_bars_required:
            aligned[sym] = df
    if len(aligned) < top_n:
        raise ValueError(f"Need at least {top_n} symbols with sufficient data")

    # Common date range: intersection of all symbol date ranges
    common_dates = None
    for sym in aligned:
        dates = set(aligned[sym]["trade_date"])
        common_dates = dates if common_dates is None else common_dates & dates
    if not common_dates:
        raise ValueError("No common trading dates across symbols")

    # Use the first symbol's DataFrame as the timeline reference
    ref_symbol = list(aligned.keys())[0]
    ref_df = aligned[ref_symbol]
    ref_df = ref_df[ref_df["trade_date"].isin(common_dates)].reset_index(drop=True)
    timeline_dates = ref_df["trade_date"].tolist()

    # ---- pre-compute trends and ATRs for each stock ----
    trends: dict[str, pd.DataFrame] = {}
    atrs: dict[str, pd.Series] = {}
    for sym in aligned:
        df = aligned[sym]
        trends[sym] = _compute_trend(
            df, params,
            volume_confirm=volume_confirm,
            volume_lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        )
        # ATR(14) as % of close
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        tr = pd.DataFrame({
            "h_l": high - low,
            "h_c": (high - close.shift(1)).abs(),
            "l_c": (low - close.shift(1)).abs(),
        }).max(axis=1)
        atr_val = tr.ewm(span=14, min_periods=5).mean()
        atrs[sym] = atr_val / close

    # ---- backtest state ----
    cash = initial_capital
    positions: dict[str, dict] = {}  # sym -> {entry_price, shares, highest_close, entry_date, profit_lock_active, profit_lock_high}
    equity_curve = pd.Series(np.nan, index=ref_df.index, dtype=float)
    daily_returns = pd.Series(0.0, index=ref_df.index, dtype=float)
    trades: list[dict] = []
    rotation_count = 0

    def _total_equity(i: int) -> float:
        val = cash
        for sym, pos in positions.items():
            close_px = float(ref_df.loc[i, "close"]) if sym == ref_symbol else float(
                aligned[sym][aligned[sym]["trade_date"] == ref_df.loc[i, "trade_date"]]["close"].iloc[0]
            ) if sym in aligned else 0.0
            val += pos["shares"] * close_px
        return val

    def _get_stock_data(sym: str, ref_idx: int):
        """Get the row index in the stock's own DataFrame for the given reference date."""
        ref_date = ref_df.loc[ref_idx, "trade_date"]
        df = aligned[sym]
        mask = df["trade_date"] == ref_date
        if mask.any():
            return mask.idxmax()
        return None

    # ---- main loop ----
    for i in range(len(ref_df)):
        ref_date = ref_df.loc[i, "trade_date"]

        # Process pending sells first
        sells_to_process = []
        for sym, pos in list(positions.items()):
            stock_idx = _get_stock_data(sym, i)
            if stock_idx is None:
                continue
            s_df = aligned[sym]
            reason, new_pl_active, new_pl_high = _check_position_exit(
                s_df, stock_idx,
                entry_price=pos["entry_price"],
                highest_close=pos["highest_close"],
                stop_loss_pct=stop_loss_pct,
                atr_trailing_mult=atr_trailing_mult,
                atr_trailing_min_gain=atr_trailing_min_gain,
                atr_series=atrs[sym],
                intrapos_dd_limit=intrapos_dd_limit,
                profit_lock_trigger=profit_lock_trigger,
                profit_lock_trailing=profit_lock_trailing,
                profit_lock_active=pos["profit_lock_active"],
                profit_lock_high=pos["profit_lock_high"],
                time_stop_days=time_stop_days,
                time_stop_min_return=time_stop_min_return,
                entry_date=pos["entry_date"],
            )
            pos["profit_lock_active"] = new_pl_active
            pos["profit_lock_high"] = new_pl_high
            # Update highest close
            close_px = float(s_df.loc[stock_idx, "close"])
            if np.isfinite(close_px):
                pos["highest_close"] = max(pos["highest_close"], close_px)

            if reason:
                sells_to_process.append((sym, reason))

        for sym, reason in sells_to_process:
            pos = positions.pop(sym)
            stock_idx = _get_stock_data(sym, i)
            if stock_idx is not None:
                s_df = aligned[sym]
                exit_px = float(s_df.loc[stock_idx, "close"])
                pnl = exit_px / pos["entry_price"] - 1.0
                # Simulate T+1 open execution for exit
                if stock_idx + 1 < len(s_df):
                    j = _next_buy_index(s_df, sym, stock_idx + 1)  # reuse buy index logic for next tradable
                    if j is not None:
                        exit_px = float(s_df.loc[j, "open"])
                exit_val = pos["shares"] * exit_px
                cost = exit_val * cost_bps / 10000.0
                cash += exit_val - cost
                trades.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": s_df.loc[stock_idx, "trade_date"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_px,
                    "return": pnl,
                    "exit_reason": reason,
                    "hold_days": (pd.Timestamp(s_df.loc[stock_idx, "trade_date"]) - pos["entry_date"]).days,
                })

        # Rebalance on schedule (or when a position was exited and we have capacity)
        is_rebalance_day = (i % rebalance_freq == 0)
        capacity_available = len(positions) < top_n

        if is_rebalance_day or (capacity_available and sells_to_process):
            # Compute ranking scores for all symbols at this date
            scores: list[tuple[str, float]] = []
            for sym in aligned:
                stock_idx = _get_stock_data(sym, i)
                if stock_idx is None:
                    continue
                if sym in positions:
                    continue  # already held
                score = _trend_strength_score(trends[sym], stock_idx)
                if score > 0:  # only consider bullish stocks
                    scores.append((sym, score))

            scores.sort(key=lambda x: x[1], reverse=True)

            # Determine which held stocks to keep
            held_scores: list[tuple[str, float]] = []
            for sym in list(positions.keys()):
                stock_idx = _get_stock_data(sym, i)
                if stock_idx is not None:
                    held_scores.append((sym, _trend_strength_score(trends[sym], stock_idx)))
            held_scores.sort(key=lambda x: x[1], reverse=True)

            # Build target portfolio: top_n by score
            all_ranked = held_scores + scores
            all_ranked.sort(key=lambda x: x[1], reverse=True)
            target = {sym for sym, score in all_ranked[:top_n] if score > 0}

            # Exit stocks not in target
            for sym in list(positions.keys()):
                if sym not in target:
                    pos = positions.pop(sym)
                    stock_idx = _get_stock_data(sym, i)
                    if stock_idx is not None:
                        s_df = aligned[sym]
                        exit_px = float(s_df.loc[stock_idx, "close"])
                        if stock_idx + 1 < len(s_df):
                            j = _next_buy_index(s_df, sym, stock_idx + 1)
                            if j is not None:
                                exit_px = float(s_df.loc[j, "open"])
                        pnl = exit_px / pos["entry_price"] - 1.0
                        exit_val = pos["shares"] * exit_px
                        cost = exit_val * cost_bps / 10000.0
                        cash += exit_val - cost
                        trades.append({
                            "symbol": sym,
                            "entry_date": pos["entry_date"],
                            "exit_date": s_df.loc[stock_idx, "trade_date"],
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_px,
                            "return": pnl,
                            "exit_reason": "rotation",
                            "hold_days": (pd.Timestamp(s_df.loc[stock_idx, "trade_date"]) - pos["entry_date"]).days,
                        })
                        rotation_count += 1

            # Enter stocks in target not held
            for sym in target:
                if sym in positions:
                    continue
                stock_idx = _get_stock_data(sym, i)
                if stock_idx is None:
                    continue
                s_df = aligned[sym]

                # T+1 open execution
                buy_idx = _next_buy_index(s_df, sym, stock_idx + 1)
                if buy_idx is None:
                    continue
                entry_px = float(s_df.loc[buy_idx, "open"])

                # Equal weight allocation
                n_positions = len(positions) + 1  # after adding this one
                allocation = (cash / (top_n - len(positions))) * 0.95  # leave buffer for costs
                if allocation <= 0:
                    continue

                shares = allocation / entry_px
                cost = allocation * cost_bps / 10000.0
                cash -= allocation + cost

                positions[sym] = {
                    "entry_price": entry_px,
                    "shares": shares,
                    "highest_close": entry_px,
                    "entry_date": pd.Timestamp(s_df.loc[buy_idx, "trade_date"]),
                    "profit_lock_active": False,
                    "profit_lock_high": entry_px,
                }

        # Record equity
        eq = _total_equity(i)
        equity_curve.iloc[i] = eq
        if i > 0:
            prev_eq = equity_curve.iloc[i - 1]
            if np.isfinite(prev_eq) and prev_eq > 0:
                daily_returns.iloc[i] = eq / prev_eq - 1.0

    # ---- close any remaining positions at end ----
    for sym, pos in list(positions.items()):
        s_df = aligned[sym]
        last_close = float(s_df.loc[len(s_df) - 1, "close"])
        exit_val = pos["shares"] * last_close
        cost = exit_val * cost_bps / 10000.0
        cash += exit_val - cost
        trades.append({
            "symbol": sym,
            "entry_date": pos["entry_date"],
            "exit_date": s_df.loc[len(s_df) - 1, "trade_date"],
            "entry_price": pos["entry_price"],
            "exit_price": last_close,
            "return": last_close / pos["entry_price"] - 1.0,
            "exit_reason": "end_of_period",
            "hold_days": (pd.Timestamp(s_df.loc[len(s_df) - 1, "trade_date"]) - pos["entry_date"]).days,
        })
    positions.clear()

    # ---- compute performance metrics ----
    daily_ret_clean = daily_returns.dropna()
    if len(daily_ret_clean) == 0:
        return RotationResult(
            symbols=list(aligned.keys()), total_return=0.0, annualized_return=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
            n_trades=len(trades), n_rotations=rotation_count,
            daily_returns=daily_returns, equity_curve=equity_curve,
            trade_log=pd.DataFrame(trades), turnover_pct=0.0,
        )

    panel = compute_performance_panel(
        daily_ret_clean.to_numpy(dtype=np.float64),
        n_concurrent_strategies=top_n,
    )
    total_ret = equity_curve.dropna().iloc[-1] / initial_capital - 1.0
    years = len(ref_df) / 252.0
    ann_ret = annualized_return_cagr(daily_ret_clean.to_numpy(dtype=np.float64), periods_per_year=252.0)
    n_trades = len(trades)
    trade_log = pd.DataFrame(trades) if trades else pd.DataFrame()
    turnover = float(trade_log["exit_price"].sum() * cost_bps / 10000.0 / initial_capital) if len(trade_log) > 0 else 0.0

    return RotationResult(
        symbols=list(aligned.keys()),
        total_return=total_ret,
        annualized_return=ann_ret,
        sharpe_ratio=panel.sharpe_ratio,
        max_drawdown=panel.max_drawdown,
        calmar_ratio=panel.calmar_ratio,
        n_trades=n_trades,
        n_rotations=rotation_count,
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        trade_log=trade_log,
        turnover_pct=turnover,
    )
