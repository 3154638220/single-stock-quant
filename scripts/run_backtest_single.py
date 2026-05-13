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


def _signed_pct(x: float) -> str:
    return "nan" if x != x else f"{x * 100:+.2f}%"


def _num(x: float, digits: int = 2) -> str:
    return "nan" if x != x else f"{x:.{digits}f}"


def _print_result(res, mode: str) -> None:
    excess = res.total_return - res.buy_hold_return
    print(f"{res.stock_name} ({res.symbol}) 回测报告")
    print(f"回测区间：{res.period} | 指标：{mode}")
    print("-" * 64)
    print(f"总收益率：{_signed_pct(res.total_return)}    买入持有：{_signed_pct(res.buy_hold_return)}    超额：{_signed_pct(excess)}")
    print(f"年化收益：{_signed_pct(res.annualized_return)}    夏普比率：{_num(res.sharpe_ratio)}    最大回撤：{_pct(res.max_drawdown)}")
    print(f"Calmar：{_num(res.calmar_ratio)}")
    print()
    print("交易统计：")
    print(f"  总交易次数：{res.n_trades}    胜率：{_pct(res.win_rate)}    平均持仓天数：{_num(res.avg_hold_days, 1)}")
    print(f"  单笔平均收益：{_signed_pct(res.avg_return_per_trade)}    最大连续盈利：{res.max_consecutive_wins}    最大连续亏损：{res.max_consecutive_losses}")
    if not res.trade_log.empty:
        print()
        print("交易记录（最近 5 笔）：")
        print("  买入日        卖出日        买价      卖价      收益      退出")
        tail = res.trade_log.tail(5)
        for _, row in tail.iterrows():
            print(
                f"  {row['buy_date'].date()}  {row['sell_date'].date()}  "
                f"{row['buy_price']:>8.2f}  {row['sell_price']:>8.2f}  {_signed_pct(row['return']):>8}  {row['exit_reason']}"
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
    parser.add_argument("--duckdb-path", help="Override DuckDB path, e.g. /path/to/market.duckdb")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bt_cfg = cfg.get("backtest", {}) or {}
    symbol = str(args.symbol).strip().zfill(6)
    name_cache_path = Path(args.stock_name_cache).expanduser() if args.stock_name_cache else resolve_stock_name_cache_path(cfg)
    stock_name = resolve_stock_names([symbol], name_cache_path).get(symbol, symbol)
    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
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
        print(f"{stock_name} ({symbol}) 三模式回测对比")
        print(f"回测区间：{results[0][1].period}")
        print("指标          总收益      买入持有    超额       夏普    最大回撤   交易数   胜率")
        for mode, res in results:
            excess = res.total_return - res.buy_hold_return
            print(
                f"{mode:<11} {_signed_pct(res.total_return):>9} {_signed_pct(res.buy_hold_return):>9} "
                f"{_signed_pct(excess):>9} {res.sharpe_ratio:>7.2f} {_pct(res.max_drawdown):>9} "
                f"{res.n_trades:>6} {_pct(res.win_rate):>8}"
            )
    else:
        mode, res = results[0]
        _print_result(res, mode)

    if args.export_trades:
        out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        for mode, res in results:
            path = out_dir / f"{symbol}_{mode}_trades.csv"
            res.trade_log.to_csv(path, index=False)
            print(f"已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
