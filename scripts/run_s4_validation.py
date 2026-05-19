#!/usr/bin/env python
"""Phase S4: Final validation — bootstrap, cost sensitivity, per-period for best configs."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.single_stock import run_single_stock_backtest
from src.backtest.performance_panel import compute_performance_panel, annualized_return_cagr
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams
from src.settings import load_config

SYMBOL = "300750"
PERIODS = {
    "T2": ("2018-06-11", "2018-12-31"),
    "T3": ("2019-01-01", "2020-12-31"),
    "T4": ("2021-01-01", "2022-12-31"),
    "T5": ("2023-01-01", "2026-04-30"),
}

CONFIGS = {
    "S2_best": {  # Best OOS Calmar from S2 grid
        "macd_fast": 10, "macd_slow": 26, "macd_signal": 8, "mode": "macd_cross",
        "atr_trailing_mult": 3.0, "atr_trailing_min_gain": 0.06,
        "intrapos_dd_limit": 0.04, "stop_loss_pct": 0.07,
    },
    "WFO_stable": {  # WFO stable params
        "macd_fast": 10, "macd_slow": 26, "macd_signal": 8, "mode": "macd_cross",
        "atr_trailing_mult": 2.5, "atr_trailing_min_gain": 0.08,
        "intrapos_dd_limit": 0.05, "stop_loss_pct": 0.07,
    },
}

BASE_BT = {
    "cost_bps": 15.0, "initial_capital": 100000,
    "volume_confirm": True, "volume_lookback": 20, "volume_ratio_min": 1.0,
}


def _pct(x): return "nan" if not np.isfinite(x) else f"{x*100:.2f}%"
def _num(x, d=2): return "nan" if not np.isfinite(x) else f"{x:.{d}f}"


def bootstrap_metric(daily_returns: np.ndarray, metric_fn, n_boot=1000, seed=42):
    """Bootstrap a metric from daily returns."""
    rng = np.random.default_rng(seed)
    n = len(daily_returns)
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(daily_returns, size=n, replace=True)
        boot_vals[b] = metric_fn(sample)
    return {
        "mean": float(np.mean(boot_vals)),
        "median": float(np.median(boot_vals)),
        "p5": float(np.percentile(boot_vals, 5)),
        "p95": float(np.percentile(boot_vals, 95)),
        "positive_frac": float(np.mean(boot_vals > 0)),
    }


def main():
    cfg = load_config(None)
    stock_name = resolve_stock_names([SYMBOL], resolve_stock_name_cache_path(cfg)).get(SYMBOL, SYMBOL)

    with DuckDBManager() as db:
        df = db.read_daily_frame(symbols=[SYMBOL])
    df = df[df["symbol"].astype(str) == SYMBOL].copy()

    results = {}
    for name, overrides in CONFIGS.items():
        print(f"\n{'='*64}")
        print(f"Config: {name}")
        trend_cfg = {k: overrides[k] for k in ["macd_fast", "macd_slow", "macd_signal", "mode"]}
        trend_cfg["min_run_len"] = 1
        params = DKTrendParams.from_mapping(trend_cfg)

        bt = {**BASE_BT}
        bt.update({k: overrides[k] for k in ["atr_trailing_mult", "atr_trailing_min_gain",
                                               "intrapos_dd_limit", "stop_loss_pct"]})
        bt["stock_name"] = stock_name

        # Full period
        res = run_single_stock_backtest(SYMBOL, df, params, **bt)
        print(f"Full ({res.period}): Ann={_pct(res.annualized_return)} MDD={_pct(res.max_drawdown)} "
              f"Calmar={_num(res.calmar_ratio)} Sharpe={_num(res.sharpe_ratio)} "
              f"Trades={res.n_trades} WinRate={_pct(res.win_rate)}")

        # Per-period
        for pname, (start, end) in PERIODS.items():
            pdf = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].copy()
            if len(pdf) < 60:
                continue
            pres = run_single_stock_backtest(SYMBOL, pdf, params, **bt)
            print(f"  {pname} ({start[:7]}~{end[:7]}): Ann={_pct(pres.annualized_return)} "
                  f"MDD={_pct(pres.max_drawdown)} Calmar={_num(pres.calmar_ratio)} "
                  f"Trades={pres.n_trades}")

        # Bootstrap
        dr = res.daily_returns.dropna().to_numpy(dtype=np.float64)
        if len(dr) > 10:
            def calmar_from_returns(r):
                pnl = np.cumprod(1 + r)
                peak = np.maximum.accumulate(pnl)
                dd = (pnl - peak) / peak
                mdd = float(np.min(dd))
                ann = float(np.mean(r) * 252)
                return ann / abs(mdd) if mdd < 0 else 0.0

            boot_calmar = bootstrap_metric(dr, calmar_from_returns)
            boot_sharpe = bootstrap_metric(dr, lambda r: float(np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r)>0 else 0.0)
            print(f"  Bootstrap Calmar: mean={boot_calmar['mean']:.2f} p5={boot_calmar['p5']:.2f} p95={boot_calmar['p95']:.2f} pos={boot_calmar['positive_frac']:.0%}")
            print(f"  Bootstrap Sharpe: mean={boot_sharpe['mean']:.2f} p5={boot_sharpe['p5']:.2f} p95={boot_sharpe['p95']:.2f} pos={boot_sharpe['positive_frac']:.0%}")

        # Cost sensitivity
        print("  Cost sensitivity:")
        for cost_bps in [10, 15, 25, 50]:
            cres = run_single_stock_backtest(SYMBOL, df, params, **{**bt, "cost_bps": cost_bps})
            print(f"    cost={cost_bps}bps: Ann={_pct(cres.annualized_return)} Calmar={_num(cres.calmar_ratio)} Trades={cres.n_trades}")

        results[name] = res

    # Compare
    print(f"\n{'='*64}")
    print("Final Comparison:")
    print(f"  {'Metric':<20} {'S2_best':>15} {'WFO_stable':>15}")
    for attr, label in [("annualized_return", "Ann Return"), ("max_drawdown", "Max DD"),
                         ("calmar_ratio", "Calmar"), ("sharpe_ratio", "Sharpe"),
                         ("n_trades", "N Trades"), ("win_rate", "Win Rate")]:
        a_val = getattr(results["S2_best"], attr)
        b_val = getattr(results["WFO_stable"], attr)
        print(f"  {label:<20} {_pct(a_val) if 'rate' in attr or 'return' in attr or 'drawdown' in attr else _num(a_val):>15} "
              f"{_pct(b_val) if 'rate' in attr or 'return' in attr or 'drawdown' in attr else _num(b_val):>15}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
