#!/usr/bin/env python
"""Run portfolio backtest on a watchlist cross-section."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.portfolio.backtest import run_portfolio_backtest
from src.backtest.transaction_costs import TransactionCostParams, transaction_cost_params_from_mapping
from src.settings import load_config, project_root


def _read_watchlist(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            symbols.append(s.zfill(6))
    if not symbols:
        raise SystemExit(f"watchlist is empty: {path}")
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio backtest on watchlist cross-section.")
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--n-top", type=int, default=5)
    parser.add_argument("--max-per-stock", type=float, default=0.25)
    parser.add_argument("--export-summary", help="CSV path for portfolio metrics")
    parser.add_argument("--export-weights", help="CSV path for daily weights")
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    risk_cfg = cfg.get("risk", {}) or {}
    symbols = _read_watchlist(Path(args.watchlist).expanduser())

    enable_index_filter = bool(risk_cfg.get("enable_index_filter", False))
    benchmark_symbol = str(risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)
    symbols_to_read = list(symbols)
    if enable_index_filter and benchmark_symbol not in symbols_to_read:
        symbols_to_read.append(benchmark_symbol)

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        data = db.read_daily_frame(symbols=symbols_to_read, start=args.start, end=args.end)
    if data.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    index_df = data[data["symbol"].astype(str) == benchmark_symbol].copy() if enable_index_filter and benchmark_symbol in set(data.get("symbol", [])) else None

    bt_cfg = cfg.get("backtest", {}) or {}
    tc_cfg = bt_cfg.get("transaction_cost", {}) or {}
    cost_params = transaction_cost_params_from_mapping(tc_cfg) if tc_cfg else None

    result = run_portfolio_backtest(
        data,
        index_ohlcv=index_df,
        n_top=args.n_top,
        max_per_stock=args.max_per_stock,
        cost_params=cost_params,
    )

    summary = result.get("summary", {})
    print(f"Portfolio backtest summary:")
    print(f"  年化收益: {summary.get('annualized_return', float('nan')):.4f}")
    print(f"  Sharpe:   {summary.get('sharpe_ratio', float('nan')):.2f}")
    print(f"  Calmar:   {summary.get('calmar_ratio', float('nan')):.2f}")
    print(f"  最大回撤: {summary.get('max_drawdown', float('nan')):.4f}")
    print(f"  调仓日数: {summary.get('n_rebalance_dates', 0)}")
    print(f"  平均持仓: {summary.get('avg_positions', 0):.1f}")

    if args.export_summary:
        out = Path(args.export_summary).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(out, index=False)
        print(f"Summary written to {out}")

    if args.export_weights:
        w = result.get("weights")
        if w is not None:
            out = Path(args.export_weights).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            w.to_csv(out)
            print(f"Weights written to {out}")

    if args.export_html:
        bt = result.get("backtest")
        if bt is not None:
            from src.backtest.report import generate_html_report
            from src.backtest.single_stock import SingleStockBacktestResult
            out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"portfolio_backtest_{datetime.now().strftime('%Y%m%d')}.html"
            dummy_res = SingleStockBacktestResult(
                symbol="portfolio", stock_name="组合",
                daily_returns=bt.daily_returns,
            )
            name_cache_path = Path(args.stock_name_cache).expanduser() if args.stock_name_cache else resolve_stock_name_cache_path(cfg)
            names = resolve_stock_names(symbols, name_cache_path)
            generate_html_report(dummy_res, data[data["symbol"].astype(str) == symbols[0]], output_path=path)
            print(f"HTML report written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
