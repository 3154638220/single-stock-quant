#!/usr/bin/env python
"""Single-stock parameter grid scan with IS/OOS split.

Usage:
  python scripts/run_single_stock_grid.py \\
    --symbol 300750 --start 2018-01-01 --end 2026-04-30 \\
    --signal-types dktrend macd donchian \\
    --output results/s2a_signal_comparison.csv
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.config import build_bt_kwargs
from src.backtest.single_stock import run_single_stock_backtest
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams, TrendMode
from src.settings import load_config

SIGNAL_GRIDS: dict[str, dict[str, list[Any]]] = {
    "dktrend": {
        "bar_period": [3, 5, 8],
        "state_confirm_days": [1, 2, 3],
    },
    "macd": {
        "macd_fast": [8, 10, 12, 14],
        "macd_signal": [6, 8, 10],
    },
    "donchian": {
        "entry_window": [15, 20, 25, 30],
        "exit_window": [8, 10, 12],
    },
}

EXIT_GRID: dict[str, list[Any]] = {
    "atr_trailing_mult": [2.0, 2.5, 3.0],
    "atr_trailing_min_gain": [0.06, 0.08, 0.10, 0.12],
    "intrapos_dd_limit": [0.04, 0.05, 0.06],
    "stop_loss_pct": [0.07, 0.08, 0.10],
}

# Default base config for params not in grid
BASE_TREND = {
    "mode": "macd_cross",
    "macd_fast": 10, "macd_slow": 26, "macd_signal": 8,
    "min_run_len": 1,
    "bar_period": 5, "state_confirm_days": 2,
    "entry_window": 20, "exit_window": 10,
}

BASE_BT = {
    "cost_bps": 15.0, "initial_capital": 100000,
    "stop_loss_pct": 0.08,
    "atr_trailing_mult": 2.5, "atr_trailing_min_gain": 0.10,
    "intrapos_dd_limit": 0.05,
    "volume_confirm": True, "volume_lookback": 20, "volume_ratio_min": 1.0,
}

IS_END = "2023-12-31"
OOS_START = "2024-01-01"


def _param_combos(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]


def _pct(x): return "nan" if not np.isfinite(x) else f"{x*100:.2f}%"


def run_single_combo(
    symbol, df, trend_overrides, bt_overrides, stock_name="",
    index_ohlcv=None, regime_gate_enabled=False,
) -> dict:
    """Run one parameter combination and return key metrics."""
    trend_cfg = {**BASE_TREND, **trend_overrides}
    bt_cfg = {**BASE_BT, **bt_overrides}

    params = DKTrendParams.from_mapping(trend_cfg)
    res = run_single_stock_backtest(
        symbol, df, params,
        cost_bps=bt_cfg["cost_bps"],
        initial_capital=bt_cfg["initial_capital"],
        stop_loss_pct=bt_cfg["stop_loss_pct"],
        atr_trailing_mult=bt_cfg["atr_trailing_mult"],
        atr_trailing_min_gain=bt_cfg["atr_trailing_min_gain"],
        intrapos_dd_limit=bt_cfg["intrapos_dd_limit"],
        volume_confirm=bt_cfg["volume_confirm"],
        volume_lookback=bt_cfg["volume_lookback"],
        volume_ratio_min=bt_cfg["volume_ratio_min"],
        stock_name=stock_name,
        index_ohlcv=index_ohlcv,
        regime_gate_enabled=regime_gate_enabled,
    )
    return {
        "ann": res.annualized_return,
        "mdd": res.max_drawdown,
        "calmar": res.calmar_ratio,
        "sharpe": res.sharpe_ratio,
        "n_trades": res.n_trades,
        "win_rate": res.win_rate,
        "total_return": res.total_return,
    }


def main():
    parser = argparse.ArgumentParser(description="Single-stock parameter grid scan")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--signal-types", nargs="+", default=["macd"],
                        choices=["dktrend", "macd", "donchian"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--regime-gate", action="store_true", help="Enable regime gate (requires CSI300 data)")
    parser.add_argument("--no-exit-grid", action="store_true", help="Skip Phase 2 exit grid")
    args = parser.parse_args()

    symbol = str(args.symbol).strip().zfill(6)
    cfg = load_config(args.config)
    stock_name = resolve_stock_names([symbol], resolve_stock_name_cache_path(cfg)).get(symbol, symbol)

    with DuckDBManager(config_path=args.config) as db:
        all_df = db.read_daily_frame(symbols=[symbol], start=args.start, end=args.end)
        index_ohlcv = None
        if args.regime_gate:
            index_ohlcv = db.read_daily_frame(symbols=["000300"])
            if index_ohlcv.empty:
                print("WARNING: No CSI300 index data found, regime gate will fail open")
    df = all_df[all_df["symbol"].astype(str) == symbol].copy()
    if df.empty:
        raise SystemExit(f"No data for {symbol}")

    is_df = df[df["trade_date"] <= IS_END].copy()
    oos_df = df[df["trade_date"] >= OOS_START].copy()

    print(f"{stock_name} ({symbol}) — IS: {is_df['trade_date'].iloc[0].date()} ~ {is_df['trade_date'].iloc[-1].date()} ({len(is_df)} rows)")
    print(f"  OOS: {oos_df['trade_date'].iloc[0].date()} ~ {oos_df['trade_date'].iloc[-1].date()} ({len(oos_df)} rows)")

    all_rows = []
    for sig_type in args.signal_types:
        sig_grid = SIGNAL_GRIDS[sig_type]
        sig_combos = _param_combos(sig_grid)

        if sig_type == "dktrend":
            mode = "eastmoney_dkbar"
        elif sig_type == "donchian":
            mode = "donchian_breakout"
        else:
            mode = "macd_cross"

        # Phase 1: Test signal variants with a fixed reasonable exit config
        fixed_exit = {"atr_trailing_mult": 2.5, "atr_trailing_min_gain": 0.10,
                       "intrapos_dd_limit": 0.05, "stop_loss_pct": 0.08}

        print(f"\n--- {sig_type} ({mode}) — {len(sig_combos)} signal combos ---")

        for i, sig_combo in enumerate(sig_combos):
            trend_overrides = dict(sig_combo)
            trend_overrides["mode"] = mode
            if sig_type == "donchian":
                trend_overrides["donchian_entry_window"] = trend_overrides.pop("entry_window")
                trend_overrides["donchian_exit_window"] = trend_overrides.pop("exit_window")

            is_res = run_single_combo(symbol, is_df, trend_overrides, fixed_exit, stock_name,
                                     index_ohlcv=index_ohlcv, regime_gate_enabled=args.regime_gate)
            oos_res = run_single_combo(symbol, oos_df, trend_overrides, fixed_exit, stock_name,
                                      index_ohlcv=index_ohlcv, regime_gate_enabled=args.regime_gate)

            row = {
                "signal_type": sig_type,
                "mode": mode,
                **sig_combo,
                **{f"is_{k}": v for k, v in is_res.items()},
                **{f"oos_{k}": v for k, v in oos_res.items()},
            }
            all_rows.append(row)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(sig_combos)}] {sig_type} signal combos done")

    # Phase 2: On best signal type, test exit grid
    # Find best signal type by median IS Calmar
    results_df = pd.DataFrame(all_rows)
    if not results_df.empty and not args.no_exit_grid:
        sig_ranking = results_df.groupby("signal_type")["is_calmar"].median().sort_values(ascending=False)
        best_sig = sig_ranking.index[0]
        print(f"\nBest signal type by median IS Calmar: {best_sig} ({sig_ranking[best_sig]:.2f})")

        # Run exit grid on best signal
        print(f"\n--- Exit grid on {best_sig} ---")
        exit_combos = _param_combos(EXIT_GRID)
        if best_sig == "dktrend":
            best_mode = "eastmoney_dkbar"
            best_trend = {"mode": best_mode, "bar_period": 5, "state_confirm_days": 2}
        elif best_sig == "donchian":
            best_mode = "donchian_breakout"
            best_trend = {"mode": best_mode, "donchian_entry_window": 20, "donchian_exit_window": 10}
        else:
            best_mode = "macd_cross"
            best_trend = {"mode": best_mode, "macd_fast": 10, "macd_slow": 26, "macd_signal": 8}

        for i, exit_combo in enumerate(exit_combos):
            is_res = run_single_combo(symbol, is_df, best_trend, exit_combo, stock_name,
                                     index_ohlcv=index_ohlcv, regime_gate_enabled=args.regime_gate)
            oos_res = run_single_combo(symbol, oos_df, best_trend, exit_combo, stock_name,
                                      index_ohlcv=index_ohlcv, regime_gate_enabled=args.regime_gate)

            row = {
                "signal_type": f"{best_sig}_exit_grid",
                "mode": best_mode,
                **exit_combo,
                **{f"is_{k}": v for k, v in is_res.items()},
                **{f"oos_{k}": v for k, v in oos_res.items()},
            }
            all_rows.append(row)

        results_df = pd.DataFrame(all_rows)

    # Filter and display IS-qualifying configs (plan S2-A criteria)
    is_mask = (
        (results_df["is_ann"] >= 0.20) &
        (results_df["is_mdd"] <= 0.25) &
        (results_df["is_calmar"] >= 0.8) &
        (results_df["is_n_trades"] >= 15)
    )
    qualifying = results_df[is_mask].copy()
    print(f"\nIS qualifying configs (Ann>=20%, MDD<=20%, Calmar>=1.0, Trades>=15): {len(qualifying)}")

    if not qualifying.empty:
        qualifying = qualifying.sort_values("is_calmar", ascending=False)
        print("\nTop-10 IS configs:")
        cols = ["signal_type", "is_ann", "is_mdd", "is_calmar", "is_sharpe", "is_n_trades",
                "oos_ann", "oos_mdd", "oos_calmar", "oos_sharpe", "oos_n_trades"]
        available = [c for c in cols if c in qualifying.columns]
        print(qualifying[available].head(10).to_string(index=False))

        # OOS validation
        oos_ok = qualifying[qualifying["oos_calmar"] >= 0.8]
        print(f"\nOOS Calmar>=0.8: {len(oos_ok)} configs pass")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"\nFull results written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
