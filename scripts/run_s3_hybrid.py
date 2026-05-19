#!/usr/bin/env python
"""Test hybrid exit strategies: profit_lock + ATR trailing + intrapos_dd."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.single_stock import run_single_stock_backtest
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

HYBRIDS = {
    "baseline_a": {  # Original production
        "stop_loss_pct": 0.08, "profit_lock_trigger": 0.08, "profit_lock_trailing": 0.04,
        "time_stop_days": 30, "time_stop_min_return": 0.03,
        "atr_trailing_mult": 0.0, "atr_trailing_min_gain": 0.0, "intrapos_dd_limit": 0.0,
    },
    "hybrid_1": {  # profit_lock + ATR trailing
        "stop_loss_pct": 0.08, "profit_lock_trigger": 0.08, "profit_lock_trailing": 0.04,
        "time_stop_days": 0, "time_stop_min_return": 0.0,
        "atr_trailing_mult": 3.0, "atr_trailing_min_gain": 0.12,
        "intrapos_dd_limit": 0.05,
    },
    "hybrid_2": {  # profit_lock + tighter ATR trailing
        "stop_loss_pct": 0.08, "profit_lock_trigger": 0.10, "profit_lock_trailing": 0.05,
        "time_stop_days": 0, "time_stop_min_return": 0.0,
        "atr_trailing_mult": 2.5, "atr_trailing_min_gain": 0.10,
        "intrapos_dd_limit": 0.05,
    },
    "hybrid_3": {  # profit_lock + intrapos_dd (no ATR trailing)
        "stop_loss_pct": 0.08, "profit_lock_trigger": 0.10, "profit_lock_trailing": 0.05,
        "time_stop_days": 0, "time_stop_min_return": 0.0,
        "atr_trailing_mult": 0.0, "atr_trailing_min_gain": 0.0,
        "intrapos_dd_limit": 0.05,
    },
    "hybrid_4": {  # Lower trigger, moderate trailing
        "stop_loss_pct": 0.07, "profit_lock_trigger": 0.12, "profit_lock_trailing": 0.06,
        "time_stop_days": 0, "time_stop_min_return": 0.0,
        "atr_trailing_mult": 2.5, "atr_trailing_min_gain": 0.15,
        "intrapos_dd_limit": 0.05,
    },
    "hybrid_5": {  # intraspos_dd primary, profit_lock secondary
        "stop_loss_pct": 0.07, "profit_lock_trigger": 0.15, "profit_lock_trailing": 0.06,
        "time_stop_days": 0, "time_stop_min_return": 0.0,
        "atr_trailing_mult": 0.0, "atr_trailing_min_gain": 0.0,
        "intrapos_dd_limit": 0.05,
    },
}


def _pct(x): return "nan" if not np.isfinite(x) else f"{x*100:.2f}%"
def _num(x, d=2): return "nan" if not np.isfinite(x) else f"{x:.{d}f}"


def main():
    cfg = load_config(None)
    stock_name = resolve_stock_names([SYMBOL], resolve_stock_name_cache_path(cfg)).get(SYMBOL, SYMBOL)

    with DuckDBManager() as db:
        df = db.read_daily_frame(symbols=[SYMBOL])
    df = df[df["symbol"].astype(str) == SYMBOL].copy()

    params = DKTrendParams.from_mapping({
        "mode": "macd_cross", "macd_fast": 10, "macd_slow": 26, "macd_signal": 8,
        "min_run_len": 1,
    })

    base_bt = {
        "cost_bps": 15.0, "initial_capital": 100000,
        "volume_confirm": True, "volume_lookback": 20, "volume_ratio_min": 1.0,
        "stock_name": stock_name,
    }

    best_name = None
    best_calmar = -999

    for name, exits in HYBRIDS.items():
        bt = {**base_bt, **exits}
        res = run_single_stock_backtest(SYMBOL, df, params, **bt)

        # Weighted score: 40% full Calmar + 30% T4 Ann + 30% T3 Calmar
        t4_ann = float("nan")
        t3_calmar = float("nan")
        for pname, (start, end) in PERIODS.items():
            pdf = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].copy()
            if len(pdf) < 60:
                continue
            pres = run_single_stock_backtest(SYMBOL, pdf, params, **bt)
            if pname == "T4":
                t4_ann = pres.annualized_return
            if pname == "T3":
                t3_calmar = pres.calmar_ratio

        score = 0.4 * res.calmar_ratio + 0.3 * min(t4_ann, 0.5) + 0.3 * min(t3_calmar, 3.0)
        if np.isfinite(score) and score > best_calmar:
            best_calmar = score
            best_name = name

        print(f"\n--- {name} (score={score:.3f}) ---")
        print(f"Full: Ann={_pct(res.annualized_return)} MDD={_pct(res.max_drawdown)} "
              f"Calmar={_num(res.calmar_ratio)} Sharpe={_num(res.sharpe_ratio)} "
              f"Trades={res.n_trades} WinRate={_pct(res.win_rate)}")
        print(f"  Exit reasons: SL={res.stop_loss_exits} Trail={res.trailing_stop_exits} "
              f"ATRtrail={res.atr_trailing_exits} PL={res.profit_lock_exits} "
              f"DD={res.intrapos_dd_exits} Time={res.time_stop_exits}")

        for pname in ["T2", "T3", "T4", "T5"]:
            pdf = df[(df["trade_date"] >= PERIODS[pname][0]) & (df["trade_date"] <= PERIODS[pname][1])].copy()
            if len(pdf) < 60:
                continue
            pres = run_single_stock_backtest(SYMBOL, pdf, params, **bt)
            print(f"  {pname}: Ann={_pct(pres.annualized_return)} MDD={_pct(pres.max_drawdown)} "
                  f"Calmar={_num(pres.calmar_ratio)} Trades={pres.n_trades}")

    print(f"\nBest hybrid: {best_name} (score={best_calmar:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
