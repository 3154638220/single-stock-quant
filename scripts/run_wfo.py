#!/usr/bin/env python
"""Run walk-forward optimization for one stock."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.wfo import run_walk_forward_optimization
from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, TrendMode
from src.settings import load_config, project_root


def _params(cfg: dict, mode: str) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    raw["mode"] = mode
    return DKTrendParams.from_mapping(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DK trend walk-forward optimization.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--oos-days", type=int, default=126)
    parser.add_argument("--mode", choices=[m.value for m in TrendMode], default="macd_cross")
    parser.add_argument("--window", choices=["rolling", "expanding"], default="rolling")
    parser.add_argument("--export-results", action="store_true")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bt_cfg = cfg.get("backtest", {}) or {}
    wfo_cfg = cfg.get("wfo", {}) or {}
    symbol = str(args.symbol).strip().zfill(6)
    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        df = db.read_daily_frame(symbols=[symbol], start=args.start, end=args.end)
    if df.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    result = run_walk_forward_optimization(
        symbol,
        df,
        base_params=_params(cfg, args.mode),
        param_grid=wfo_cfg.get("param_grid"),
        train_days=args.train_days,
        oos_days=args.oos_days,
        mode=args.mode,
        window=args.window,
        cost_bps=float(bt_cfg.get("cost_bps", 15.0)),
        initial_capital=float(bt_cfg.get("initial_capital", 100000)),
    )

    agg = result["aggregated"]
    print(f"{symbol} WFO | mode={args.mode} | folds={result['n_folds']}")
    print(
        "OOS combined: "
        f"total={agg.get('total_return_combined', float('nan')):.4f} "
        f"ann={agg.get('annualized_return_combined', float('nan')):.4f} "
        f"sharpe={agg.get('sharpe_ratio_combined', float('nan')):.2f} "
        f"mdd={agg.get('max_drawdown_combined', float('nan')):.4f}"
    )
    if args.export_results:
        out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{symbol}_wfo_{datetime.now().strftime('%Y%m%d')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
