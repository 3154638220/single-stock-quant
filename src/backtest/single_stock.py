"""Single-stock DK trend backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.performance_panel import annualized_return_cagr, compute_performance_panel
from src.backtest.risk_metrics import risk_off_multiplier_from_index
from src.backtest.transaction_costs import TransactionCostParams, cost_params_dict_for_logging
from src.indicators import DKTrendParams, compute_dktrend
from src.market.tradability import is_open_limit_down_unsellable, is_open_limit_up_unbuyable, is_row_suspended_like
from src.signals.consensus import compute_consensus_trend
from src.signals.generator import apply_volume_confirmation, compute_signal_quality


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
    buy_hold_annualized_return: float
    excess_annualized_return: float
    information_ratio: float
    beta_to_benchmark: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    stop_loss_exits: int
    trailing_stop_exits: int
    atr_stop_exits: int = 0
    avg_position_fraction: float = 1.0
    cost_model: dict | None = None
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


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


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    atr.name = "atr"
    return atr


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


def _next_sell_index(df: pd.DataFrame, start_idx: int, symbol: str = "") -> int | None:
    for j in range(start_idx, len(df)):
        if not _is_tradable_open(df, j):
            continue
        prev_close = float(df.loc[j - 1, "close"]) if j > 0 else np.nan
        open_px = float(df.loc[j, "open"])
        if symbol and is_open_limit_down_unsellable(open_px, prev_close, symbol):
            continue
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


def _future_exit_index(actions: dict[int, str], start_idx: int) -> int | None:
    exits = [idx for idx, action in actions.items() if idx >= start_idx and action in {"sell", "stop_loss", "trailing_stop", "atr_stop"}]
    return min(exits) if exits else None


def _index_allows_new_position(
    index_ohlcv: pd.DataFrame | None,
    *,
    benchmark_symbol: str,
    asof: pd.Timestamp,
    lookback_days: int,
    drop_threshold: float,
    risk_off_factor: float,
) -> bool:
    if index_ohlcv is None or index_ohlcv.empty:
        return True
    multiplier, _, _ = risk_off_multiplier_from_index(
        index_ohlcv,
        benchmark_symbol=str(benchmark_symbol).zfill(6),
        asof=asof,
        lookback_trading_days=int(lookback_days),
        drop_threshold=float(drop_threshold),
        risk_off_factor=float(risk_off_factor),
    )
    return multiplier > 0.0


def _close_to_returns(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=np.float64, name=name)
    work = df.copy()
    if "trade_date" in work.columns:
        dates = pd.to_datetime(work["trade_date"]).dt.normalize()
    else:
        dates = pd.to_datetime(work.index).normalize()
    close = pd.to_numeric(work["close"], errors="coerce")
    out = pd.Series(close.to_numpy(dtype=np.float64), index=dates, name="close")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.pct_change().fillna(0.0).rename(name)


def _information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: float = 252.0) -> float:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    sd = float(active.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def _beta_to_benchmark(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    strategy = aligned.iloc[:, 0].to_numpy(dtype=np.float64)
    benchmark = aligned.iloc[:, 1].to_numpy(dtype=np.float64)
    var = float(np.var(benchmark, ddof=1))
    if var <= 0 or not np.isfinite(var):
        return float("nan")
    cov = float(np.cov(strategy, benchmark, ddof=1)[0, 1])
    return float(cov / var)


def run_single_stock_backtest(
    symbol: str,
    ohlcv: pd.DataFrame,
    params: DKTrendParams,
    *,
    cost_bps: float = 15.0,
    cost_params: TransactionCostParams | None = None,
    initial_capital: float = 100_000.0,
    stock_name: str = "",
    volume_confirm: bool = False,
    volume_lookback: int = 20,
    volume_ratio_min: float = 1.0,
    consensus_n_agree: int | None = None,
    enable_index_filter: bool = False,
    index_ohlcv: pd.DataFrame | None = None,
    benchmark_symbol: str = "510300",
    extreme_lookback_days: int = 10,
    extreme_drop_threshold: float = 0.05,
    risk_off_factor: float = 0.0,
    stop_loss_pct: float = 0.0,
    trailing_stop_pct: float = 0.0,
    atr_stop_multiplier: float = 0.0,
    atr_stop_period: int = 14,
    risk_per_trade_pct: float = 0.0,
    position_size_cap: float = 1.0,
    stop_reentry_enabled: bool = False,
    stop_reentry_cooldown: int = 3,
    stop_reentry_min_run: int = 2,
    trend_override: pd.DataFrame | None = None,
    min_quality_score: float = 0.0,
    quality_score_mode: str = "hard",
) -> SingleStockBacktestResult:
    """Backtest one stock with T+1 open execution and single-position long/flat state.

    If *trend_override* is provided it must be a DataFrame with at minimum a
    ``dk_signal`` column aligned to the same length as *ohlcv*; it completely
    replaces the on-the-fly DK-trend computation (useful for permutation tests).
    """
    code = str(symbol).strip().zfill(6)
    df = _prepare_ohlcv(ohlcv)
    if trend_override is not None:
        trend = trend_override.reset_index(drop=True)
    elif consensus_n_agree is not None and int(consensus_n_agree) > 1:
        trend = compute_consensus_trend(
            df,
            base_params=params,
            n_agree=int(consensus_n_agree),
            volume_confirm=volume_confirm,
            volume_lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    else:
        trend = compute_dktrend(df, params).reset_index(drop=True)
        trend = apply_volume_confirmation(
            trend,
            enabled=volume_confirm,
            lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    if cost_params is not None:
        buy_frac = cost_params.buy_fraction()
        sell_frac = cost_params.sell_fraction()
        cost_model_dict = cost_params_dict_for_logging(cost_params)
    else:
        c = max(float(cost_bps), 0.0) * 1e-4
        buy_frac = c
        sell_frac = c
        cost_model_dict = {"type": "symmetric", "cost_bps": float(cost_bps)}

    # ── Trade attribution pre-computation ──
    _attr_quality = compute_signal_quality(trend).reset_index(drop=True)
    _attr_atr14 = _compute_atr(df, 14)
    _attr_close = pd.to_numeric(df["close"], errors="coerce")
    _attr_atr_pct = _attr_atr14 / _attr_close
    _attr_vol = pd.to_numeric(df["volume"], errors="coerce")
    _attr_avg_vol = _attr_vol.rolling(window=volume_lookback, min_periods=volume_lookback).mean()
    _attr_vol_ratio = _attr_vol / _attr_avg_vol
    _attr_stock_ret60 = _attr_close.pct_change(60)
    _attr_regime = pd.Series("unknown", index=df.index, dtype=object)
    _attr_rs_60 = pd.Series(np.nan, index=df.index, dtype=np.float64)
    _attr_idx_ret60 = pd.Series(np.nan, index=df.index, dtype=np.float64)
    if index_ohlcv is not None and not index_ohlcv.empty:
        idx_dates = pd.to_datetime(index_ohlcv["trade_date"]).dt.normalize()
        idx_close = pd.to_numeric(index_ohlcv["close"], errors="coerce")
        idx_ret = idx_close.pct_change(60)
        idx_map = dict(zip(idx_dates, idx_ret))
        stock_dates = pd.to_datetime(df["trade_date"]).dt.normalize()
        for _j, _d in enumerate(stock_dates):
            _v = idx_map.get(_d)
            if _v is not None and np.isfinite(_v):
                _attr_idx_ret60.iloc[_j] = _v
        _attr_regime = _attr_idx_ret60.apply(
            lambda x: "bull" if x > 0.10 else ("bear" if x < -0.10 else "ranging") if pd.notna(x) else "unknown"
        )
        _attr_rs_60 = _attr_stock_ret60 - _attr_idx_ret60
    else:
        _attr_rs_60 = _attr_stock_ret60

    actions: dict[int, str] = {}
    entry_meta: dict[int, dict] = {}
    planned_long = False
    pending_buy_idx: int | None = None
    for i, row in trend.iterrows():
        if pending_buy_idx is not None and pending_buy_idx <= i:
            planned_long = True
            pending_buy_idx = None
        sig = str(row.get("dk_signal", ""))
        if sig == "buy" and not planned_long and pending_buy_idx is None and i + 1 < len(df):
            if enable_index_filter and not _index_allows_new_position(
                index_ohlcv,
                benchmark_symbol=benchmark_symbol,
                asof=pd.Timestamp(df.loc[i, "trade_date"]),
                lookback_days=extreme_lookback_days,
                drop_threshold=extreme_drop_threshold,
                risk_off_factor=risk_off_factor,
            ):
                continue
            _q = float(_attr_quality.iloc[i]) if i < len(_attr_quality) else 0.0
            if min_quality_score > 0 and quality_score_mode == "hard" and _q < min_quality_score:
                continue
            j = _next_buy_index(df, code, i + 1)
            if j is None:
                continue
            actions.setdefault(j, "buy")
            pending_buy_idx = j
            entry_meta[j] = {
                "quality": float(_attr_quality.iloc[i]) if i < len(_attr_quality) else 0.0,
                "vol_ratio": float(_attr_vol_ratio.iloc[i]) if i < len(_attr_vol_ratio) and np.isfinite(_attr_vol_ratio.iloc[i]) else float("nan"),
                "atr_pct": float(_attr_atr_pct.iloc[i]) if i < len(_attr_atr_pct) and np.isfinite(_attr_atr_pct.iloc[i]) else float("nan"),
                "regime": str(_attr_regime.iloc[i]) if i < len(_attr_regime) else "unknown",
                "rs_60": float(_attr_rs_60.iloc[i]) if i < len(_attr_rs_60) and np.isfinite(_attr_rs_60.iloc[i]) else float("nan"),
            }
        elif sig == "sell" and planned_long and i + 1 < len(df):
            j = _next_sell_index(df, i + 1, code)
            if j is None:
                continue
            actions.setdefault(j, "sell")
            planned_long = False
        elif sig == "sell" and pending_buy_idx is not None:
            actions.pop(pending_buy_idx, None)
            pending_buy_idx = None

    stop_loss = max(float(stop_loss_pct), 0.0)
    trailing_stop = max(float(trailing_stop_pct), 0.0)
    atr_stop_mult = max(float(atr_stop_multiplier), 0.0)
    atr_period = max(int(atr_stop_period), 1)
    atr_series = _compute_atr(df, atr_period) if atr_stop_mult > 0 else pd.Series(dtype=float)
    risk_per_trade = max(float(risk_per_trade_pct), 0.0)
    pos_cap = max(min(float(position_size_cap), 1.0), 0.0)
    use_risk_sizing = risk_per_trade > 0 and (stop_loss > 0 or atr_stop_mult > 0)
    cash = float(initial_capital)
    total_equity = cash
    shares = 0.0
    in_pos = False
    entry_date = pd.NaT
    entry_price = float("nan")
    entry_cash = float("nan")
    highest_close = float("nan")
    atr_stop_price = float("nan")
    trades: list[dict] = []
    equity = np.zeros(len(df), dtype=np.float64)
    position_fractions: list[float] = []
    reentry_cooldown = max(int(stop_reentry_cooldown), 0)
    reentry_min_run = max(int(stop_reentry_min_run), 1)
    reentry_enabled = bool(stop_reentry_enabled)
    cooldown_remaining = 0

    for i in range(len(df)):
        action = actions.get(i)
        if action == "buy" and not in_pos:
            price = float(df.loc[i, "open"])
            if price > 0 and np.isfinite(price):
                total_equity = cash  # cash is all equity when flat
                if use_risk_sizing:
                    stop_dist_pct = stop_loss
                    if atr_stop_mult > 0 and not atr_series.empty:
                        atr_val = float(atr_series.iloc[i])
                        if np.isfinite(atr_val) and atr_val > 0:
                            atr_stop_price = price - atr_stop_mult * atr_val
                            atr_dist_pct = (price - atr_stop_price) / price
                            if np.isfinite(atr_dist_pct) and atr_dist_pct > 0:
                                stop_dist_pct = max(stop_dist_pct, atr_dist_pct) if stop_loss > 0 else atr_dist_pct
                    if stop_dist_pct > 0:
                        max_risk_amount = total_equity * risk_per_trade
                        position_value = max_risk_amount / stop_dist_pct
                        position_value = min(position_value, total_equity * pos_cap)
                        position_value = position_value * (1.0 - buy_frac)
                    else:
                        position_value = total_equity * (1.0 - buy_frac)
                else:
                    position_value = total_equity * (1.0 - buy_frac)
                shares = position_value / price
                entry_cash = total_equity
                cash = total_equity - position_value
                position_fractions.append(position_value / total_equity if total_equity > 0 else 1.0)
                in_pos = True
                entry_date = pd.Timestamp(df.loc[i, "trade_date"])
                entry_price = price
                highest_close = float(df.loc[i, "close"])
                _meta = entry_meta.get(i, {})
                entry_quality = float(_meta.get("quality", 0.0))
                entry_vol_ratio = float(_meta.get("vol_ratio", float("nan")))
                entry_atr_pct_val = float(_meta.get("atr_pct", float("nan")))
                entry_regime = str(_meta.get("regime", "unknown"))
                entry_rs_60_val = float(_meta.get("rs_60", float("nan")))
                mae = 0.0
                mfe = 0.0
                if atr_stop_mult > 0 and not atr_series.empty:
                    atr_val2 = float(atr_series.iloc[i])
                    if np.isfinite(atr_val2) and atr_val2 > 0:
                        atr_stop_price = price - atr_stop_mult * atr_val2
                    else:
                        atr_stop_price = float("nan")
                elif atr_stop_mult <= 0:
                    atr_stop_price = float("nan")
        elif action in {"sell", "stop_loss", "trailing_stop", "atr_stop"} and in_pos:
            price = float(df.loc[i, "open"])
            if price <= 0 or not np.isfinite(price):
                continue
            cash = shares * price * (1.0 - sell_frac)
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
                    "exit_reason": action if action != "sell" else "signal",
                    "entry_quality_score": entry_quality,
                    "entry_volume_ratio": entry_vol_ratio,
                    "entry_atr_pct": entry_atr_pct_val,
                    "entry_market_regime": entry_regime,
                    "entry_rs_60": entry_rs_60_val,
                    "mae": mae,
                    "mfe": mfe,
                }
            )
            in_pos = False
            highest_close = float("nan")
            if reentry_enabled and action in {"stop_loss", "trailing_stop", "atr_stop"}:
                cooldown_remaining = reentry_cooldown
        equity[i] = shares * float(df.loc[i, "close"]) if in_pos else cash
        if in_pos:
            close_px = float(df.loc[i, "close"])
            if np.isfinite(close_px) and np.isfinite(entry_price) and entry_price > 0:
                pnl = close_px / entry_price - 1.0
                mae = min(mae, pnl)
                mfe = max(mfe, pnl)
            if np.isfinite(close_px):
                highest_close = close_px if not np.isfinite(highest_close) else max(highest_close, close_px)
            forced_reason = ""
            if stop_loss > 0 and np.isfinite(close_px) and np.isfinite(entry_price) and close_px / entry_price - 1.0 < -stop_loss:
                forced_reason = "stop_loss"
            elif atr_stop_mult > 0 and np.isfinite(close_px) and np.isfinite(atr_stop_price) and close_px < atr_stop_price:
                forced_reason = "atr_stop"
            elif trailing_stop > 0 and np.isfinite(close_px) and np.isfinite(highest_close) and close_px / highest_close - 1.0 < -trailing_stop:
                forced_reason = "trailing_stop"
            if forced_reason and i + 1 < len(df):
                j = _next_sell_index(df, i + 1, code)
                existing = _future_exit_index(actions, i + 1)
                if j is not None and (existing is None or j < existing):
                    actions[j] = forced_reason

        if reentry_enabled and not in_pos and cooldown_remaining == 0 and i + 1 < len(df):
            color_now = str(trend.loc[i, "dk_color"]) if i < len(trend) else ""
            run_now = int(trend.loc[i, "dk_run_len"]) if i < len(trend) else 0
            if color_now == "red" and run_now >= reentry_min_run:
                if not _is_tradable_open(df, i + 1):
                    pass
                elif enable_index_filter and not _index_allows_new_position(
                    index_ohlcv,
                    benchmark_symbol=benchmark_symbol,
                    asof=pd.Timestamp(df.loc[i, "trade_date"]),
                    lookback_days=extreme_lookback_days,
                    drop_threshold=extreme_drop_threshold,
                    risk_off_factor=risk_off_factor,
                ):
                    pass
                else:
                    j = _next_buy_index(df, code, i + 1)
                    if j is not None:
                        actions[j] = "buy"
                        planned_long = True

        if cooldown_remaining > 0 and not in_pos:
            cooldown_remaining -= 1

    last_idx = len(df) - 1
    if in_pos:
        price = float(df.loc[last_idx, "close"])
        cash = shares * price * (1.0 - sell_frac)
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
                "entry_quality_score": entry_quality,
                "entry_volume_ratio": entry_vol_ratio,
                "entry_atr_pct": entry_atr_pct_val,
                "entry_market_regime": entry_regime,
                "entry_rs_60": entry_rs_60_val,
                "mae": mae,
                "mfe": mfe,
            }
        )
        in_pos = False
        highest_close = float("nan")
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
    stop_loss_exits = int((trade_log["exit_reason"] == "stop_loss").sum()) if not trade_log.empty else 0
    trailing_stop_exits = int((trade_log["exit_reason"] == "trailing_stop").sum()) if not trade_log.empty else 0
    atr_stop_exits = int((trade_log["exit_reason"] == "atr_stop").sum()) if not trade_log.empty else 0
    avg_position_fraction = float(np.mean(position_fractions)) if position_fractions else 1.0
    buy_hold_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)
    buy_hold_returns = _close_to_returns(df, "buy_hold_ret")
    buy_hold_annualized_return = annualized_return_cagr(buy_hold_returns.to_numpy(dtype=np.float64))
    excess_annualized_return = (
        float(panel.annualized_return - buy_hold_annualized_return)
        if np.isfinite(panel.annualized_return) and np.isfinite(buy_hold_annualized_return)
        else float("nan")
    )
    information_ratio = _information_ratio(daily_returns, buy_hold_returns)
    benchmark_returns = _close_to_returns(index_ohlcv, "benchmark_ret") if index_ohlcv is not None else pd.Series(dtype=np.float64, name="benchmark_ret")
    beta_to_benchmark = _beta_to_benchmark(daily_returns, benchmark_returns)
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
        buy_hold_annualized_return=buy_hold_annualized_return,
        excess_annualized_return=excess_annualized_return,
        information_ratio=information_ratio,
        beta_to_benchmark=beta_to_benchmark,
        sharpe_ratio=panel.sharpe_ratio,
        max_drawdown=panel.max_drawdown,
        calmar_ratio=panel.calmar_ratio,
        stop_loss_exits=stop_loss_exits,
        trailing_stop_exits=trailing_stop_exits,
        atr_stop_exits=atr_stop_exits,
        avg_position_fraction=avg_position_fraction,
        cost_model=cost_model_dict,
        trade_log=trade_log,
        daily_returns=daily_returns,
    )
