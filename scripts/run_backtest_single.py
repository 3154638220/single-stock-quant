#!/usr/bin/env python
"""Run a single-stock DK trend backtest."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.config import build_bt_kwargs
from src.backtest.single_stock import run_single_stock_backtest
from src.backtest.transaction_costs import TransactionCostParams, transaction_cost_params_from_mapping
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
    print(
        f"超额年化：{_signed_pct(res.excess_annualized_return)}    "
        f"信息比率：{_num(res.information_ratio)}    Beta(CSI300)：{_num(res.beta_to_benchmark)}"
    )
    print(f"Calmar：{_num(res.calmar_ratio)}")
    print()
    print("交易统计：")
    print(f"  总交易次数：{res.n_trades}    胜率：{_pct(res.win_rate)}    平均持仓天数：{_num(res.avg_hold_days, 1)}")
    print(f"  单笔平均收益：{_signed_pct(res.avg_return_per_trade)}    最大连续盈利：{res.max_consecutive_wins}    最大连续亏损：{res.max_consecutive_losses}")
    print(f"  固定止损：{res.stop_loss_exits}    追踪止损：{res.trailing_stop_exits}    ATR止损：{res.atr_stop_exits}")
    print(f"  盈利保护：{res.profit_lock_exits}    市场退出：{res.market_exit_exits}    时间止损：{res.time_stop_exits}")
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


def _bt_kwargs(cfg: dict, *, index_ohlcv: pd.DataFrame | None = None) -> dict:
    return build_bt_kwargs(cfg, index_ohlcv=index_ohlcv)


def _stop_comparison_rows(symbol: str, df: pd.DataFrame, params: DKTrendParams, base_kwargs: dict, stock_name: str) -> list[dict]:
    scenarios = [
        ("无止损", 0.0, 0.0),
        ("固定5%", 0.05, 0.0),
        ("固定8%", 0.08, 0.0),
        ("追踪10%", 0.0, 0.10),
        ("追踪15%", 0.0, 0.15),
    ]
    rows = []
    for label, stop_loss, trailing_stop in scenarios:
        kwargs = dict(base_kwargs)
        kwargs.update({"stop_loss_pct": stop_loss, "trailing_stop_pct": trailing_stop, "stock_name": stock_name})
        res = run_single_stock_backtest(symbol, df, params, **kwargs)
        rows.append(
            {
                "scenario": label,
                "stop_loss_pct": stop_loss,
                "trailing_stop_pct": trailing_stop,
                "total_return": res.total_return,
                "annualized_return": res.annualized_return,
                "max_drawdown": res.max_drawdown,
                "sharpe_ratio": res.sharpe_ratio,
                "calmar_ratio": res.calmar_ratio,
                "stop_exits": res.stop_loss_exits + res.trailing_stop_exits,
                "n_trades": res.n_trades,
                "win_rate": res.win_rate,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-stock DK trend backtest.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--mode", choices=[m.value for m in TrendMode])
    parser.add_argument("--consensus", action="store_true", help="Use multi-mode consensus instead of one DK mode")
    parser.add_argument("--compare-modes", action="store_true")
    parser.add_argument("--compare-stops", action="store_true")
    parser.add_argument("--compare-costs", action="store_true")
    parser.add_argument("--export-trades", action="store_true")
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--permutation-test", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path, e.g. /path/to/market.duckdb")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbol = str(args.symbol).strip().zfill(6)
    risk_cfg = cfg.get("risk", {}) or {}
    benchmark_symbol = str(risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)
    name_cache_path = Path(args.stock_name_cache).expanduser() if args.stock_name_cache else resolve_stock_name_cache_path(cfg)
    stock_name = resolve_stock_names([symbol], name_cache_path).get(symbol, symbol)
    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        symbols_to_read = [symbol]
        if bool(risk_cfg.get("enable_index_filter", False)) and benchmark_symbol not in symbols_to_read:
            symbols_to_read.append(benchmark_symbol)
        all_df = db.read_daily_frame(symbols=symbols_to_read, start=args.start, end=args.end)
    df = all_df[all_df["symbol"].astype(str) == symbol].copy()
    index_df = all_df[all_df["symbol"].astype(str) == benchmark_symbol].copy() if benchmark_symbol in set(all_df.get("symbol", [])) else None
    if df.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    configured_mode = str(cfg.get("trend_signal", {}).get("mode", "macd_cross"))
    use_consensus = args.consensus or (configured_mode == "consensus" and not args.compare_modes)
    selected_mode = args.mode or ("macd_cross" if use_consensus else configured_mode)
    modes = [m.value for m in TrendMode] if args.compare_modes else [selected_mode]
    base_kwargs = _bt_kwargs(cfg, index_ohlcv=index_df)
    if use_consensus:
        base_kwargs["consensus_n_agree"] = int((cfg.get("trend_signal", {}) or {}).get("consensus_n_agree", 2))
    else:
        base_kwargs["consensus_n_agree"] = None
    if args.compare_stops:
        rows = _stop_comparison_rows(symbol, df, _params(cfg, selected_mode), base_kwargs, stock_name)
        table = pd.DataFrame(rows)
        print(f"{stock_name} ({symbol}) 止损参数对比")
        print("方案        固定止损  追踪止损  总收益    年化收益  最大回撤  Sharpe  Calmar  止损次数")
        for row in rows:
            print(
                f"{row['scenario']:<10} {_pct(row['stop_loss_pct']):>8} {_pct(row['trailing_stop_pct']):>8} "
                f"{_signed_pct(row['total_return']):>8} {_signed_pct(row['annualized_return']):>8} "
                f"{_pct(row['max_drawdown']):>8} {row['sharpe_ratio']:>7.2f} {row['calmar_ratio']:>7.2f} "
                f"{row['stop_exits']:>8}"
            )
        out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{symbol}_stop_compare_{datetime.now().strftime('%Y%m%d')}.csv"
        table.to_csv(path, index=False)
        print(f"已写入 {path}")
        return 0

    results = []
    for mode in modes:
        kwargs = dict(base_kwargs)
        kwargs["stock_name"] = stock_name
        res = run_single_stock_backtest(
            symbol,
            df,
            _params(cfg, mode),
            **kwargs,
        )
        results.append((mode, res))

    if args.compare_modes:
        print(f"{stock_name} ({symbol}) 三模式回测对比")
        print(f"回测区间：{results[0][1].period}")
        print("指标          总收益      买入持有    超额年化   IR      Beta    夏普    最大回撤   交易数   胜率")
        for mode, res in results:
            print(
                f"{mode:<11} {_signed_pct(res.total_return):>9} {_signed_pct(res.buy_hold_return):>9} "
                f"{_signed_pct(res.excess_annualized_return):>9} {res.information_ratio:>7.2f} "
                f"{res.beta_to_benchmark:>7.2f} {res.sharpe_ratio:>7.2f} {_pct(res.max_drawdown):>9} "
                f"{res.n_trades:>6} {_pct(res.win_rate):>8}"
            )
    else:
        mode, res = results[0]
        _print_result(res, mode)

    if args.export_trades or args.export_html:
        out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        mode, res = results[0]
        if args.export_trades:
            path = out_dir / f"{symbol}_{mode}_trades.csv"
            res.trade_log.to_csv(path, index=False)
            print(f"已写入 {path}")
        if args.export_html:
            from src.backtest.report import generate_html_report
            path = out_dir / f"{symbol}_backtest_{datetime.now().strftime('%Y%m%d')}.html"
            generate_html_report(res, df, index_ohlcv=index_df, output_path=path)
            print(f"已写入 {path}")

    if args.permutation_test:
        from src.backtest.permutation_test import run_permutation_test
        mode, res = results[0]
        pt_result = run_permutation_test(
            symbol, df, _params(cfg, mode),
            n_permutations=args.n_permutations, **base_kwargs,
        )
        print(f"\n置换检验 (n={pt_result.n_permutations}):")
        print(f"  观测 Sharpe: {pt_result.observed_sharpe:.3f}")
        print(f"  零分布均值: {pt_result.null_sharpe_mean:.3f}  (95%分位: {pt_result.null_sharpe_p95:.3f})")
        print(f"  p-value: {pt_result.p_value:.4f}  {'✅ 显著' if pt_result.is_significant else '⚠️ 未通过显著性检验，策略可能为噪声拟合'}")

    if args.compare_costs:
        from src.backtest.transaction_costs import TransactionCostParams
        out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        cost_scenarios = [
            ("零成本（上界）", None),
            ("对称15bps", 15.0),
            ("真实A股", TransactionCostParams(commission_buy_bps=2.5, commission_sell_bps=2.5, slippage_bps_per_side=2.0, stamp_duty_sell_bps=5.0)),
            ("高成本（机构）", TransactionCostParams(commission_buy_bps=10, commission_sell_bps=10, slippage_bps_per_side=5.0, stamp_duty_sell_bps=5.0)),
        ]
        rows = []
        mode = results[0][0] if isinstance(results[0], tuple) else selected_mode
        for label, cost_spec in cost_scenarios:
            kwargs = dict(base_kwargs)
            kwargs["stock_name"] = stock_name
            if cost_spec is None:
                kwargs["cost_bps"] = 0.0
                kwargs["cost_params"] = None
                buy_bps = sell_bps = 0.0
            elif isinstance(cost_spec, TransactionCostParams):
                kwargs["cost_params"] = cost_spec
                buy_bps = cost_spec.buy_fraction() * 1e4
                sell_bps = cost_spec.sell_fraction() * 1e4
            else:
                kwargs["cost_bps"] = float(cost_spec)
                kwargs["cost_params"] = None
                buy_bps = sell_bps = float(cost_spec)
            res2 = run_single_stock_backtest(symbol, df, _params(cfg, mode), **kwargs)
            total_cost = buy_bps + sell_bps
            rows.append({
                "model": label, "buy_bps": buy_bps, "sell_bps": sell_bps,
                "ann_return": res2.annualized_return, "sharpe": res2.sharpe_ratio,
                "n_trades": res2.n_trades, "total_cost_bps": total_cost * res2.n_trades,
            })
        print(f"\n{stock_name} ({symbol}) 成本敏感性分析")
        print(f"{'成本模型':<16} {'买方bps':>8} {'卖方bps':>8} {'年化收益':>9} {'Sharpe':>7} {'交易数':>6}")
        for row in rows:
            print(f"{row['model']:<16} {row['buy_bps']:>8.1f} {row['sell_bps']:>8.1f} {_signed_pct(row['ann_return']):>9} {row['sharpe']:>7.2f} {row['n_trades']:>6}")
        table = pd.DataFrame(rows)
        path = out_dir / f"{symbol}_cost_compare_{datetime.now().strftime('%Y%m%d')}.csv"
        table.to_csv(path, index=False)
        print(f"已写入 {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
