#!/usr/bin/env python
"""Run a single-stock DK trend backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.single_stock import run_single_stock_backtest
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams, TrendMode
from src.settings import load_config, project_root


def _params(cfg: dict, mode: str | None) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    if mode:
        raw["mode"] = mode
    return DKTrendParams.from_mapping(raw)


def _pct(x: float) -> str:
    return "nan" if x != x else f"{x * 100:.2f}%"


def _print_result(res, mode: str) -> None:
    print(f"{res.stock_name} ({res.symbol}) backtest | {res.period} | mode={mode}")
    print(f"total={_pct(res.total_return)} buy_hold={_pct(res.buy_hold_return)} annual={_pct(res.annualized_return)}")
    print(f"sharpe={res.sharpe_ratio:.2f} max_drawdown={_pct(res.max_drawdown)} calmar={res.calmar_ratio:.2f}")
    print(f"trades={res.n_trades} win_rate={_pct(res.win_rate)} avg_hold_days={res.avg_hold_days:.1f} avg_trade={_pct(res.avg_return_per_trade)}")
    if not res.trade_log.empty:
        print("recent trades:")
        tail = res.trade_log.tail(5)
        for _, row in tail.iterrows():
            print(
                f"  {row['buy_date'].date()} -> {row['sell_date'].date()} "
                f"{row['buy_price']:.2f} -> {row['sell_price']:.2f} return={_pct(row['return'])}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-stock DK trend backtest.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--mode", choices=[m.value for m in TrendMode])
    parser.add_argument("--compare-modes", action="store_true")
    parser.add_argument("--export-trades", action="store_true")
    parser.add_argument("--config", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bt_cfg = cfg.get("backtest", {}) or {}
    symbol = str(args.symbol).strip().zfill(6)
    stock_name = resolve_stock_names([symbol], resolve_stock_name_cache_path(cfg)).get(symbol, symbol)
    with DuckDBManager(config_path=args.config) as db:
        df = db.read_daily_frame(symbols=[symbol], start=args.start, end=args.end)
    if df.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    modes = [m.value for m in TrendMode] if args.compare_modes else [args.mode or str(cfg.get("trend_signal", {}).get("mode", "macd_cross"))]
    results = []
    for mode in modes:
        res = run_single_stock_backtest(
            symbol,
            df,
            _params(cfg, mode),
            cost_bps=float(bt_cfg.get("cost_bps", 15.0)),
            initial_capital=float(bt_cfg.get("initial_capital", 100000)),
            stock_name=stock_name,
        )
        results.append((mode, res))

    if args.compare_modes:
        print("mode        total     buy_hold  sharpe  max_dd   trades win_rate")
        for mode, res in results:
            print(f"{mode:<11} {_pct(res.total_return):>8} {_pct(res.buy_hold_return):>8} {res.sharpe_ratio:>7.2f} {_pct(res.max_drawdown):>8} {res.n_trades:>6} {_pct(res.win_rate):>8}")
    else:
        mode, res = results[0]
        _print_result(res, mode)

    if args.export_trades:
        out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        for mode, res in results:
            path = out_dir / f"{symbol}_{mode}_trades.csv"
            res.trade_log.to_csv(path, index=False)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
