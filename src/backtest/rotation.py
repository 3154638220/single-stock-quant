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


def _rs_momentum_score(df: pd.DataFrame, idx: int) -> float:
    """Relative strength momentum: excess return over 20 and 60 days.

    Returns a score where higher = stronger relative momentum.
    Negative scores indicate the stock is underperforming.
    """
    if idx < 20:
        return -1.0
    close = pd.to_numeric(df["close"], errors="coerce")
    if idx >= 60:
        rs_20 = close.iloc[idx] / close.iloc[idx - 20] - 1.0
        rs_60 = close.iloc[idx] / close.iloc[idx - 60] - 1.0
        return float(rs_20 * 0.6 + rs_60 * 0.4)
    rs_20 = close.iloc[idx] / close.iloc[idx - 20] - 1.0
    return float(rs_20)


def _multi_factor_score(
    trend: pd.DataFrame,
    df: pd.DataFrame,
    idx: int,
    *,
    w_trend: float = 0.85,
    w_vol_adj: float = 0.15,
) -> float:
    """IC-verified multi-factor ranking score.

    Weights calibrated from P1-B IC analysis (2026-05-19):
    - dk_value ICIR=+0.123 (dominant, only robust factor)
    - vol_adj ICIR=+0.018 (weak positive tilt)
    - rs_20/rs_60/above_ma120/run_len all negative ICIR → excluded
    """
    if idx < 0 or idx >= len(trend):
        return -999.0

    # Trend factor: dk_value only — run_len bonus removed (ICIR=-0.049)
    color = str(trend.loc[idx, "dk_color"])
    if color != "red":
        return -1.0
    dk_val = float(trend.loc[idx, "dk_value"]) if "dk_value" in trend.columns else 0.5

    # Volatility adjustment (inverse vol — lower vol preferred, weak ICIR=+0.018)
    close = pd.to_numeric(df["close"], errors="coerce")
    ret_1d = close.pct_change()
    vol_20 = ret_1d.iloc[max(0, idx - 20):idx + 1].std() * np.sqrt(252)
    vol_score = 1.0 / (vol_20 + 0.05) if np.isfinite(vol_20) and vol_20 > 0 else 0.5

    return float(w_trend * dk_val + w_vol_adj * vol_score)


def _ranking_score(trend: pd.DataFrame, df: pd.DataFrame, idx: int, mode: str) -> float:
    """Dispatch to the appropriate ranking function based on mode."""
    if mode == "rs_momentum":
        return _rs_momentum_score(df, idx)
    if mode == "multi_factor":
        return _multi_factor_score(trend, df, idx)
    return _trend_strength_score(trend, idx)


def _apply_sector_constraint(
    ranked: list[tuple[str, float]],
    sector_map: dict[str, str],
    top_n: int,
) -> set[str]:
    """Limit to one stock per sector, keeping highest-ranked within each sector."""
    seen_sectors: set[str] = set()
    final: set[str] = set()
    for sym, score in ranked:
        sector = sector_map.get(sym, sym)
        if sector not in seen_sectors:
            seen_sectors.add(sector)
            final.add(sym)
        if len(final) == top_n:
            break
    return final


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
    position_sizing: str = "equal",
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
    sector_map: dict[str, str] | None = None,
    index_ohlcv: pd.DataFrame | None = None,
    market_regime_mode: str = "off",
    regime_ma_period: int = 120,
    regime_reduce_top_n: int = 1,
    regime_fast_ma_period: int = 0,
    regime_fast_threshold: float = 0.97,
    regime_drawdown_trigger: float = 0.0,
    regime_drawdown_lookback: int = 60,
    portfolio_dd_limit: float = 0.0,
    volatility_target_ann: float = 0.0,
    volatility_scale_floor: float = 0.30,
    symbol_params: dict[str, DKTrendParams] | None = None,
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
        effective_params = (symbol_params or {}).get(sym, params)
        trends[sym] = _compute_trend(
            df, effective_params,
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

    # ---- market regime detection (pre-compute for all dates) ----
    # H-1/H-2: Dual-speed MA + drawdown trigger detection
    market_regime_bearish: pd.Series | None = None
    if market_regime_mode != "off":
        if index_ohlcv is not None:
            idx_df = index_ohlcv.copy()
            idx_df["trade_date"] = pd.to_datetime(idx_df["trade_date"]).dt.normalize()
            idx_df = idx_df.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
            idx_close = pd.to_numeric(idx_df["close"], errors="coerce")
        else:
            # Fallback: equal-weighted daily return of pool, then cumulative
            n_stocks = len(aligned)
            daily_rets = None
            for sym in aligned:
                sym_df = aligned[sym].copy()
                sym_close = pd.to_numeric(sym_df["close"], errors="coerce")
                sym_ret = sym_close.pct_change()
                if daily_rets is None:
                    daily_rets = sym_ret
                else:
                    daily_rets = daily_rets + sym_ret
            if daily_rets is not None and n_stocks > 0:
                daily_rets = daily_rets / n_stocks
                idx_close = (1 + daily_rets.fillna(0)).cumprod()
            else:
                idx_close = pd.Series(dtype=float)

        # H-1: Slow MA (compatible with existing MA120 logic)
        slow_ma_period = regime_ma_period
        idx_ma_slow = idx_close.rolling(slow_ma_period, min_periods=slow_ma_period // 2).mean()
        slope_slow = idx_ma_slow.diff(10)
        is_bear_slow = (idx_close < idx_ma_slow) & (slope_slow < 0)

        # H-1: Fast MA for quick bear detection
        if regime_fast_ma_period > 0:
            idx_ma_fast = idx_close.rolling(regime_fast_ma_period, min_periods=regime_fast_ma_period // 2).mean()
            slope_fast = idx_ma_fast.diff(5)
            is_bear_fast = (idx_close < idx_ma_fast * regime_fast_threshold) & (slope_fast < 0)
        else:
            is_bear_fast = pd.Series(False, index=idx_close.index)

        regime_bear = is_bear_slow | is_bear_fast

        # H-2: Drawdown trigger (second defense line)
        if regime_drawdown_trigger > 0:
            rolling_peak = idx_close.rolling(regime_drawdown_lookback, min_periods=20).max()
            drawdown_from_peak = idx_close / rolling_peak - 1.0
            is_bear_drawdown = drawdown_from_peak < -regime_drawdown_trigger
            regime_bear = regime_bear | is_bear_drawdown

        # Align to reference timeline
        if index_ohlcv is not None:
            regime_map = dict(zip(idx_df["trade_date"], regime_bear))
            market_regime_bearish = pd.Series(
                [regime_map.get(d, False) for d in timeline_dates],
                index=ref_df.index,
                dtype=bool,
            )
        else:
            # regime_bear resampled to ref_df timeline
            regime_bear_aligned = regime_bear.reset_index(drop=True)
            if len(regime_bear_aligned) > len(ref_df):
                regime_bear_aligned = regime_bear_aligned.iloc[:len(ref_df)]
            market_regime_bearish = pd.Series(
                regime_bear_aligned.values[:len(ref_df)] if len(regime_bear_aligned) >= len(ref_df)
                else list(regime_bear_aligned) + [False] * (len(ref_df) - len(regime_bear_aligned)),
                index=ref_df.index,
                dtype=bool,
            )

    # ---- backtest state ----
    cash = initial_capital
    positions: dict[str, dict] = {}  # sym -> {entry_price, shares, highest_close, entry_date, profit_lock_active, profit_lock_high}
    equity_curve = pd.Series(np.nan, index=ref_df.index, dtype=float)
    daily_returns = pd.Series(0.0, index=ref_df.index, dtype=float)
    trades: list[dict] = []
    rotation_count = 0
    portfolio_dd_triggered = False
    equity_peak = initial_capital

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

        # Market regime gate: reduce exposure in bear markets
        effective_top_n = top_n
        if market_regime_mode != "off" and market_regime_bearish is not None:
            if i < len(market_regime_bearish) and market_regime_bearish.iloc[i]:
                if market_regime_mode == "exit":
                    effective_top_n = 0
                elif market_regime_mode == "reduce":
                    effective_top_n = min(regime_reduce_top_n, top_n)

        # H-3: Portfolio equity curve drawdown limit (third defense line)
        force_rebalance = False
        if portfolio_dd_limit > 0:
            current_equity = _total_equity(i)
            equity_peak = max(equity_peak, current_equity)
            equity_drawdown = current_equity / equity_peak - 1.0
            if equity_drawdown < -portfolio_dd_limit:
                effective_top_n = 0
                force_rebalance = True
                portfolio_dd_triggered = True

        # Force rebalance if bear regime requires fewer positions than currently held
        force_rebalance = (
            force_rebalance
            or (
                market_regime_mode != "off"
                and market_regime_bearish is not None
                and i < len(market_regime_bearish)
                and market_regime_bearish.iloc[i]
                and len(positions) > effective_top_n
            )
        )

        # Rebalance on schedule (or when a position was exited and we have capacity)
        is_rebalance_day = (i % rebalance_freq == 0)
        capacity_available = len(positions) < effective_top_n

        if is_rebalance_day or force_rebalance or (capacity_available and sells_to_process):
            # Compute ranking scores for all symbols at this date
            scores: list[tuple[str, float]] = []
            for sym in aligned:
                stock_idx = _get_stock_data(sym, i)
                if stock_idx is None:
                    continue
                if sym in positions:
                    continue  # already held
                score = _ranking_score(trends[sym], aligned[sym], stock_idx, ranking_mode)
                if score > 0:  # only consider bullish stocks
                    scores.append((sym, score))

            scores.sort(key=lambda x: x[1], reverse=True)

            # Determine which held stocks to keep
            held_scores: list[tuple[str, float]] = []
            for sym in list(positions.keys()):
                stock_idx = _get_stock_data(sym, i)
                if stock_idx is not None:
                    held_scores.append((sym, _ranking_score(trends[sym], aligned[sym], stock_idx, ranking_mode)))
            held_scores.sort(key=lambda x: x[1], reverse=True)

            # Build target portfolio: top_n by score
            all_ranked = held_scores + scores
            all_ranked.sort(key=lambda x: x[1], reverse=True)
            target = {sym for sym, score in all_ranked[:effective_top_n] if score > 0}

            # Apply sector concentration constraint (max 1 per sector)
            if sector_map:
                target = _apply_sector_constraint(all_ranked, sector_map, effective_top_n)

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

                # Position sizing
                n_positions = len(positions) + 1  # after adding this one
                if position_sizing == "vol_inverse":
                    # Compute weights for all target symbols by inverse volatility
                    target_list = list(target)
                    vols = {}
                    for t_sym in target_list:
                        t_df = aligned[t_sym]
                        t_close = pd.to_numeric(t_df["close"], errors="coerce")
                        t_ret = t_close.pct_change().iloc[max(0, len(t_close) - 20):]
                        vols[t_sym] = float(t_ret.std() * np.sqrt(252)) + 1e-6
                    raw_w = {t_sym: 1.0 / vols[t_sym] for t_sym in target_list}
                    total_w = sum(raw_w.values())
                    weights = {t_sym: min(max(raw_w[t_sym] / total_w, 0.30), 0.70) for t_sym in target_list}
                    # Renormalise after clamping
                    total_w2 = sum(weights.values())
                    weights = {t_sym: w / total_w2 for t_sym, w in weights.items()}
                    sym_weight = weights.get(sym, 1.0 / max(effective_top_n, 1))
                    allocation = cash * sym_weight * 0.95
                else:
                    n_target = max(effective_top_n - len(positions), 1)
                    allocation = (cash / n_target) * 0.95  # leave buffer for costs
                if allocation <= 0:
                    continue

                # I-2: Volatility target position scaling
                if volatility_target_ann > 0 and i >= 20:
                    recent_rets = daily_returns.iloc[max(0, i - 20):i].dropna()
                    if len(recent_rets) >= 10:
                        realized_vol = float(np.std(recent_rets.to_numpy(dtype=np.float64)) * np.sqrt(252))
                        if np.isfinite(realized_vol) and realized_vol > 0:
                            vol_scale = min(1.0, volatility_target_ann / realized_vol)
                            vol_scale = max(vol_scale, volatility_scale_floor)
                            allocation *= vol_scale

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
