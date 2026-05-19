#!/usr/bin/env python
"""Phase S1: 300750 full-period baseline with per-period (T2/T3/T4/T5) breakdown."""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.config import build_bt_kwargs
from src.backtest.single_stock import run_single_stock_backtest
from src.backtest.performance_panel import compute_performance_panel
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams
from src.settings import load_config

PERIODS = {
    "T2": ("2018-06-11", "2018-12-31"),   # Trade war (300750 listed 2018-06-11)
    "T3": ("2019-01-01", "2020-12-31"),   # Tech bull
    "T4": ("2021-01-01", "2022-12-31"),   # New energy bear
    "T5": ("2023-01-01", "2026-04-30"),   # Recovery oscillation
}


def _pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x * 100:.2f}%"


def _num(x: float, digits: int = 2) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.{digits}f}"


def run_baseline(symbol: str, config_path: str, label: str) -> dict:
    cfg = load_config(config_path)
    bt_cfg = cfg.get("backtest", {}) or {}
    filt_cfg = cfg.get("signal_filter", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}
    stock_name = resolve_stock_names([symbol], resolve_stock_name_cache_path(cfg)).get(symbol, symbol)

    with DuckDBManager(config_path=config_path) as db:
        df = db.read_daily_frame(symbols=[symbol])
    df = df[df["symbol"].astype(str) == symbol].copy()
    if df.empty:
        raise SystemExit("No data for 300750")

    params = DKTrendParams.from_mapping(cfg.get("trend_signal", {}))
    bt_kwargs = build_bt_kwargs(cfg)
    bt_kwargs["stock_name"] = stock_name

    full_res = run_single_stock_backtest(symbol, df, params, **bt_kwargs)

    # Per-period analysis
    periods = {}
    for pname, (start, end) in PERIODS.items():
        pdf = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].copy()
        if len(pdf) < 60:
            periods[pname] = None
            continue
        try:
            pres = run_single_stock_backtest(symbol, pdf, params, **bt_kwargs)
            periods[pname] = pres
        except Exception:
            periods[pname] = None

    return {
        "label": label,
        "full": full_res,
        "periods": periods,
    }


def main():
    symbol = "300750"
    configs = [
        ("configs/single_stock/300750_baseline_a.yaml", "Baseline A (MACD+profit_lock)"),
        ("configs/single_stock/300750_baseline_b.yaml", "Baseline B (DKTrend+ATRtrailing)"),
    ]

    results = []
    for cfg_path, label in configs:
        print(f"\n{'='*64}")
        print(f"Running {label}...")
        r = run_baseline(symbol, cfg_path, label)
        results.append(r)

        res = r["full"]
        print(f"\n{label} — Full Period ({res.period}):")
        print(f"  Ann: {_pct(res.annualized_return)}  Sharpe: {_num(res.sharpe_ratio)}  MDD: {_pct(res.max_drawdown)}  Calmar: {_num(res.calmar_ratio)}")
        print(f"  Trades: {res.n_trades}  WinRate: {_pct(res.win_rate)}  AvgHold: {_num(res.avg_hold_days, 1)}d")
        print(f"  Exit reasons — SL:{res.stop_loss_exits} Trail:{res.trailing_stop_exits} ATRtrail:{res.atr_trailing_exits} PL:{res.profit_lock_exits} DD:{res.intrapos_dd_exits} Time:{res.time_stop_exits} Signal:{res.n_trades - res.stop_loss_exits - res.trailing_stop_exits - res.atr_trailing_exits - res.profit_lock_exits - res.intrapos_dd_exits - res.time_stop_exits - res.market_exit_exits - res.dk_fade_exits}")

        print(f"\n  Period breakdown:")
        print(f"  {'Period':<6} {'Ann':>8} {'MDD':>8} {'Calmar':>8} {'Trades':>7} {'WinRate':>8}")
        print(f"  {'-'*50}")
        for pname in ["T2", "T3", "T4", "T5"]:
            pres = r["periods"].get(pname)
            if pres is None:
                print(f"  {pname:<6} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>7} {'N/A':>8}")
            else:
                print(f"  {pname:<6} {_pct(pres.annualized_return):>8} {_pct(pres.max_drawdown):>8} {_num(pres.calmar_ratio):>8} {pres.n_trades:>7} {_pct(pres.win_rate):>8}")

    # Comparison summary
    print(f"\n{'='*64}")
    print("Comparison Summary:")
    print(f"  {'Metric':<20} {'Baseline A':>15} {'Baseline B':>15}")
    print(f"  {'-'*52}")
    for attr, label in [("annualized_return", "Ann Return"), ("max_drawdown", "Max DD"),
                         ("calmar_ratio", "Calmar"), ("sharpe_ratio", "Sharpe"),
                         ("n_trades", "N Trades"), ("win_rate", "Win Rate")]:
        a_val = getattr(results[0]["full"], attr)
        b_val = getattr(results[1]["full"], attr)
        a_str = _pct(a_val) if "rate" in attr or "return" in attr or "drawdown" in attr else _num(a_val)
        b_str = _pct(b_val) if "rate" in attr or "return" in attr or "drawdown" in attr else _num(b_val)
        print(f"  {label:<20} {a_str:>15} {b_str:>15}")

    # Export trade logs
    out_dir = ROOT / "results" / "s1_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        label_slug = r["label"].split("(")[0].strip().replace(" ", "_").lower()
        r["full"].trade_log.to_csv(out_dir / f"{label_slug}_trades.csv", index=False)
    print(f"\nTrade logs exported to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
