#!/usr/bin/env python
"""Run single-stock backtests for a watchlist and export a summary."""

from __future__ import annotations

import argparse
import sys
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
from src.settings import load_config


def _read_watchlist(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            symbols.append(s.zfill(6))
    if not symbols:
        raise SystemExit(f"watchlist is empty: {path}")
    return symbols


def _params(cfg: dict, mode: str) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    raw["mode"] = mode
    return DKTrendParams.from_mapping(raw)


def _pct(x: float) -> str:
    return "nan" if x != x else f"{x * 100:.2f}%"


def _signed_pct(x: float) -> str:
    return "nan" if x != x else f"{x * 100:+.2f}%"


COST_SCENARIOS = [
    ("zero_cost", "零成本（上界）", None, None),
    ("symmetric_15bps", "对称15bps", 15.0, None),
    ("real_ashare", "真实A股", None, TransactionCostParams(
        commission_buy_bps=2.5, commission_sell_bps=2.5,
        slippage_bps_per_side=2.0, stamp_duty_sell_bps=5.0,
    )),
    ("high_slippage", "高滑点（机构）", None, TransactionCostParams(
        commission_buy_bps=10.0, commission_sell_bps=10.0,
        slippage_bps_per_side=5.0, stamp_duty_sell_bps=5.0,
    )),
]


def _cost_sensitivity_rows(
    symbol: str,
    df: pd.DataFrame,
    params: DKTrendParams,
    base_kwargs: dict,
    stock_name: str,
) -> list[dict]:
    rows = []
    for scenario_id, label, cost_bps, cost_params in COST_SCENARIOS:
        kwargs = dict(base_kwargs)
        kwargs["stock_name"] = stock_name
        if cost_bps is not None:
            kwargs["cost_bps"] = float(cost_bps)
            kwargs["cost_params"] = None
        else:
            kwargs["cost_params"] = cost_params
        res = run_single_stock_backtest(symbol, df, params, **kwargs)
        buy_bps = (cost_params.buy_fraction() * 1e4) if cost_params else (cost_bps or 0.0)
        sell_bps = (cost_params.sell_fraction() * 1e4) if cost_params else (cost_bps or 0.0)
        rows.append({
            "symbol": symbol,
            "stock_name": stock_name,
            "scenario": scenario_id,
            "label": label,
            "buy_bps": buy_bps,
            "sell_bps": sell_bps,
            "total_bps_per_roundtrip": buy_bps + sell_bps,
            "annualized_return": res.annualized_return,
            "sharpe_ratio": res.sharpe_ratio,
            "calmar_ratio": res.calmar_ratio,
            "max_drawdown": res.max_drawdown,
            "n_trades": res.n_trades,
            "win_rate": res.win_rate,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backtests for every symbol in a watchlist.")
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--mode", choices=[m.value for m in TrendMode])
    parser.add_argument("--consensus", action="store_true", help="Use multi-mode consensus instead of one DK mode")
    parser.add_argument("--export-summary", required=True)
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--compare-costs", action="store_true", help="Run cost sensitivity analysis across all stocks")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    risk_cfg = cfg.get("risk", {}) or {}
    trend_cfg = cfg.get("trend_signal", {}) or {}
    configured_mode = str(trend_cfg.get("mode", "macd_cross"))
    use_consensus = args.consensus or (configured_mode == "consensus" and args.mode is None)
    selected_mode = args.mode or ("macd_cross" if use_consensus else configured_mode)
    symbols = _read_watchlist(Path(args.watchlist).expanduser())
    name_cache_path = Path(args.stock_name_cache).expanduser() if args.stock_name_cache else resolve_stock_name_cache_path(cfg)
    names = resolve_stock_names(symbols, name_cache_path)

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
    base_kwargs = build_bt_kwargs(cfg, index_ohlcv=index_df)
    if not use_consensus:
        base_kwargs["consensus_n_agree"] = None

    rows = []
    for symbol in symbols:
        df = data[data["symbol"].astype(str) == symbol].copy()
        if df.empty:
            rows.append({"symbol": symbol, "stock_name": names.get(symbol, symbol), "status": "no_data"})
            continue
        kwargs = dict(base_kwargs)
        kwargs["stock_name"] = names.get(symbol, symbol)
        res = run_single_stock_backtest(
            symbol,
            df,
            _params(cfg, selected_mode),
            **kwargs,
        )
        rows.append(
            {
                "symbol": res.symbol,
                "stock_name": res.stock_name,
                "status": "ok",
                "period": res.period,
                "total_return": res.total_return,
                "annualized_return": res.annualized_return,
                "buy_hold_return": res.buy_hold_return,
                "sharpe_ratio": res.sharpe_ratio,
                "max_drawdown": res.max_drawdown,
                "calmar_ratio": res.calmar_ratio,
                "n_trades": res.n_trades,
                "win_rate": res.win_rate,
                "stop_loss_exits": res.stop_loss_exits,
                "trailing_stop_exits": res.trailing_stop_exits,
                "atr_stop_exits": res.atr_stop_exits,
                "profit_lock_exits": res.profit_lock_exits,
                "market_exit_exits": res.market_exit_exits,
                "time_stop_exits": res.time_stop_exits,
            }
        )

    out = Path(args.export_summary).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out, index=False)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"已完成 {ok}/{len(rows)} 个标的，汇总写入 {out}")

    if args.compare_costs:
        cost_rows = []
        selected_params = _params(cfg, selected_mode)
        for symbol in symbols:
            df = data[data["symbol"].astype(str) == symbol].copy()
            if df.empty:
                continue
            stock_name = names.get(symbol, symbol)
            kwargs = dict(base_kwargs)
            kwargs["stock_name"] = stock_name
            cost_rows.extend(
                _cost_sensitivity_rows(symbol, df, selected_params, base_kwargs, stock_name)
            )
        cost_df = pd.DataFrame(cost_rows)
        cost_path = out.parent / f"{out.stem}_cost_sensitivity.csv"
        cost_df.to_csv(cost_path, index=False)
        print(f"成本敏感性分析已写入 {cost_path}")

        # Summarise Sharpe decay from zero-cost to real A-share
        zero = cost_df[cost_df["scenario"] == "zero_cost"]
        real = cost_df[cost_df["scenario"] == "real_ashare"]
        if not zero.empty and not real.empty:
            merged = zero.merge(real, on="symbol", suffixes=("_zero", "_real"))
            merged["sharpe_decay"] = merged["sharpe_ratio_zero"] - merged["sharpe_ratio_real"]
            avg_decay = merged["sharpe_decay"].mean()
            n_high_decay = int((merged["sharpe_decay"] > 0.25).sum())
            print(f"\n成本压力测试摘要：")
            print(f"  平均 Sharpe 衰减（零成本→真实A股）：{avg_decay:.3f}")
            print(f"  Sharpe 衰减 > 0.25 的标的数：{n_high_decay}/{len(merged)}")
            if avg_decay > 0.25:
                print(f"  ⚠️ 平均衰减超过 0.25，建议优先降低换手而非调参追收益")
            else:
                print(f"  ✅ 平均衰减在可接受范围")

    if args.export_html:
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        sharpe_ok = sum(1 for r in ok_rows if r.get("sharpe_ratio", 0) > 0)
        calmar_ok = sum(1 for r in ok_rows if r.get("calmar_ratio", 0) > 0.5)
        median_calmar = float(pd.DataFrame(ok_rows)["calmar_ratio"].median()) if ok_rows else float("nan")

        import numpy as np
        sorted_rows = sorted(ok_rows, key=lambda r: r.get("sharpe_ratio", -99), reverse=True)
        top10 = "".join(
            f"<tr><td>{r['symbol']}</td><td>{r.get('stock_name','')}</td>"
            f"<td>{r.get('annualized_return',0)*100:.1f}%</td><td>{r.get('sharpe_ratio',0):.2f}</td>"
            f"<td>{r.get('calmar_ratio',0):.2f}</td><td>{r.get('n_trades',0)}</td></tr>"
            for r in sorted_rows[:10]
        )
        bottom10 = "".join(
            f"<tr><td>{r['symbol']}</td><td>{r.get('stock_name','')}</td>"
            f"<td>{r.get('annualized_return',0)*100:.1f}%</td><td>{r.get('sharpe_ratio',0):.2f}</td>"
            f"<td>{r.get('calmar_ratio',0):.2f}</td><td>{r.get('n_trades',0)}</td></tr>"
            for r in sorted_rows[-10:]
        )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>批量回测报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 20px; }}
.kpi {{ display: inline-block; background: #f8f9fa; border-radius: 6px; padding: 10px 18px; margin: 6px; text-align: center; }}
.kpi .v {{ font-size: 20px; font-weight: 700; }}
.kpi .l {{ font-size: 11px; color: #888; }}
.pass {{ color: #27ae60; font-weight: 700; }}
.fail {{ color: #c0392b; font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; }}
th, td {{ border: 1px solid #e0e0e0; padding: 5px 10px; text-align: right; }}
th {{ background: #f5f5f5; }}
td:first-child, th:first-child {{ text-align: left; }}
</style></head>
<body>
<h1>批量回测汇总报告</h1>
<div style="margin:16px 0">
<div class="kpi"><div class="v">{ok}</div><div class="l">有效标的</div></div>
<div class="kpi"><div class="v">{sharpe_ok}/{ok}</div><div class="l">Sharpe &gt; 0 {'✅' if sharpe_ok/ok>=0.6 else '⚠️'}</div></div>
<div class="kpi"><div class="v">{calmar_ok}/{ok}</div><div class="l">Calmar &gt; 0.5</div></div>
<div class="kpi"><div class="v">{median_calmar:.2f}</div><div class="l">中位 Calmar {'✅' if median_calmar>=0.5 else '⚠️'}</div></div>
</div>

<h2>Top 10</h2>
<table><tr><th>代码</th><th>名称</th><th>年化</th><th>Sharpe</th><th>Calmar</th><th>交易</th></tr>{top10}</table>

<h2>Bottom 10</h2>
<table><tr><th>代码</th><th>名称</th><th>年化</th><th>Sharpe</th><th>Calmar</th><th>交易</th></tr>{bottom10}</table>

<h2>全部标的</h2>
<table><tr><th>代码</th><th>名称</th><th>年化</th><th>Sharpe</th><th>Calmar</th><th>最大回撤</th><th>胜率</th><th>交易</th></tr>
{"".join(
    f"<tr><td>{r.get('symbol','')}</td><td>{r.get('stock_name','')}</td>"
    f"<td>{r.get('annualized_return',0)*100:.1f}%</td><td>{r.get('sharpe_ratio',0):.2f}</td>"
    f"<td>{r.get('calmar_ratio',0):.2f}</td><td>{r.get('max_drawdown',0)*100:.1f}%</td>"
    f"<td>{r.get('win_rate',0)*100:.1f}%</td><td>{r.get('n_trades',0)}</td></tr>"
    for r in sorted_rows
)}
</table>
<div style="color:#aaa;font-size:11px;text-align:center;margin-top:24px">Generated by single-stock-quant</div>
</body></html>"""
        html_path = out.parent / f"{out.stem}.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"HTML报告已写入 {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
