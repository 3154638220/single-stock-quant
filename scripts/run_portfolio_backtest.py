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

from src.backtest.config import build_bt_kwargs
from src.backtest.transaction_costs import transaction_cost_params_from_mapping
from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams
from src.portfolio.backtest import build_meta_label_score_panel, run_portfolio_backtest
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


def _read_symbol_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        raw = str(value).strip()
        if not raw:
            return []
        path = Path(raw).expanduser()
        if path.is_file():
            raw_items = path.read_text(encoding="utf-8").splitlines()
        else:
            raw_items = raw.split(",")
    symbols = []
    for item in raw_items:
        sym = str(item).split("#", 1)[0].strip()
        if sym:
            symbols.append(sym.zfill(6))
    return list(dict.fromkeys(symbols))


def _load_industry_map(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k).zfill(6): str(v) for k, v in value.items() if str(k).strip()}

    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise SystemExit(f"industry map not found: {path}")
    df = pd.read_csv(path)
    lowered = {c.lower(): c for c in df.columns}
    symbol_col = lowered.get("symbol") or lowered.get("code")
    industry_col = lowered.get("industry") or lowered.get("industry_name")
    if not symbol_col or not industry_col:
        raise SystemExit("--industry-map must contain symbol/code and industry/industry_name columns")
    return {
        str(row[symbol_col]).zfill(6): str(row[industry_col])
        for _, row in df[[symbol_col, industry_col]].dropna().iterrows()
    }


def _params(cfg: dict) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    if raw.get("mode") == "consensus":
        raw["mode"] = "macd_cross"
    return DKTrendParams.from_mapping(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio backtest on watchlist cross-section.")
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--n-top", type=int, default=5)
    parser.add_argument("--max-per-stock", type=float, default=0.25)
    parser.add_argument("--max-per-industry", type=float, help="Maximum portfolio weight per industry")
    parser.add_argument("--index-symbol", default=None, help="Benchmark/index symbol used for RS and regime features")
    parser.add_argument(
        "--ranking-profile",
        choices=("balanced", "meta_priority", "dk_meta", "dk_fresh_meta", "dk_calibrated_meta"),
        help="Cross-sectional ranking profile for portfolio ranking experiments",
    )
    parser.add_argument("--enable-meta-label", action="store_true", help="Build expanding-window p_win scores for ranking")
    parser.add_argument("--min-meta-score", type=float, help="Set rank score to zero when p_win is below this threshold")
    parser.add_argument("--exclude-symbols", help="Comma-separated symbols or a text file to zero out in E20 experiments")
    parser.add_argument("--greylist-symbols", help="Comma-separated symbols or a text file to down-weight in E20 experiments")
    parser.add_argument("--greylist-score-scale", type=float, help="Score multiplier for greylisted symbols")
    parser.add_argument("--industry-map", help="CSV with symbol/code and industry/industry_name columns")
    parser.add_argument("--meta-label-min-train-days", type=int, default=504)
    parser.add_argument("--meta-label-refit-days", type=int, default=63)
    parser.add_argument("--meta-label-min-samples", type=int, default=10)
    parser.add_argument("--require-above-ma120", action="store_true")
    parser.add_argument("--require-positive-rs60", action="store_true")
    parser.add_argument("--export-summary", help="CSV path for portfolio metrics")
    parser.add_argument("--export-weights", help="CSV path for daily weights")
    parser.add_argument("--export-scores", help="CSV path for daily rank scores")
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    risk_cfg = cfg.get("risk", {}) or {}
    portfolio_cfg = cfg.get("portfolio", {}) or {}
    symbols = _read_watchlist(Path(args.watchlist).expanduser())
    trend_params = _params(cfg)
    ranking_profile = str(args.ranking_profile or portfolio_cfg.get("ranking_profile", "balanced"))
    exclude_symbols = _read_symbol_list(args.exclude_symbols or portfolio_cfg.get("exclude_symbols"))
    greylist_symbols = _read_symbol_list(args.greylist_symbols or portfolio_cfg.get("greylist_symbols"))
    greylist_score_scale = float(
        args.greylist_score_scale
        if args.greylist_score_scale is not None
        else portfolio_cfg.get("greylist_score_scale", 0.50)
    )
    industry_map = _load_industry_map(args.industry_map or portfolio_cfg.get("industry_map"))
    max_per_industry = float(
        args.max_per_industry
        if args.max_per_industry is not None
        else portfolio_cfg.get("max_per_industry", 0.40)
    )

    benchmark_symbol = str(args.index_symbol or risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)
    symbols_to_read = list(symbols)
    if benchmark_symbol not in symbols_to_read:
        symbols_to_read.append(benchmark_symbol)

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        data = db.read_daily_frame(symbols=symbols_to_read, start=args.start, end=args.end)
    if data.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    portfolio_df = data[data["symbol"].isin(symbols)].copy()
    index_df = data[data["symbol"] == benchmark_symbol].copy()
    if index_df.empty:
        index_df = None
    if portfolio_df.empty:
        raise SystemExit("no watchlist daily data found; run scripts/fetch_stock.py first")

    bt_cfg = cfg.get("backtest", {}) or {}
    tc_cfg = bt_cfg.get("transaction_cost", {}) or {}
    cost_params = transaction_cost_params_from_mapping(tc_cfg) if tc_cfg else None
    bt_kwargs = build_bt_kwargs(cfg, index_ohlcv=index_df)
    bt_kwargs["consensus_n_agree"] = None

    meta_scores = None
    if args.enable_meta_label:
        meta_scores = build_meta_label_score_panel(
            portfolio_df,
            trend_params,
            index_ohlcv=index_df,
            bt_kwargs=bt_kwargs,
            min_train_days=args.meta_label_min_train_days,
            refit_every=args.meta_label_refit_days,
            min_samples=args.meta_label_min_samples,
        )

    result = run_portfolio_backtest(
        portfolio_df,
        index_ohlcv=index_df,
        n_top=args.n_top,
        max_per_stock=args.max_per_stock,
        max_per_industry=max_per_industry,
        industry_map=industry_map,
        cost_params=cost_params,
        meta_label_scores=meta_scores,
        min_meta_score=args.min_meta_score,
        ranking_profile=ranking_profile,
        dk_params=trend_params,
        require_above_ma120=args.require_above_ma120,
        require_positive_rs60=args.require_positive_rs60,
        exclude_symbols=exclude_symbols,
        greylist_symbols=greylist_symbols,
        greylist_score_scale=greylist_score_scale,
    )

    summary = result.get("summary", {})
    print("Portfolio backtest summary:")
    print(f"  年化收益: {summary.get('annualized_return', float('nan')):.4f}")
    print(f"  Sharpe:   {summary.get('sharpe_ratio', float('nan')):.2f}")
    print(f"  Calmar:   {summary.get('calmar_ratio', float('nan')):.2f}")
    print(f"  最大回撤: {summary.get('max_drawdown', float('nan')):.4f}")
    print(f"  调仓日数: {summary.get('n_rebalance_dates', 0)}")
    print(f"  平均持仓: {summary.get('avg_positions', 0):.1f}")
    print(f"  排名画像: {ranking_profile}")
    if exclude_symbols:
        print(f"  排除标的: {','.join(exclude_symbols)}")
    if greylist_symbols:
        print(f"  降权标的: {','.join(greylist_symbols)} x {greylist_score_scale:.2f}")

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

    if args.export_scores:
        s = result.get("scores")
        if s is not None:
            out = Path(args.export_scores).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            s.to_csv(out)
            print(f"Scores written to {out}")

    if args.export_html:
        bt = result.get("backtest")
        if bt is not None:
            from src.backtest.report import generate_html_report
            from src.backtest.single_stock import SingleStockBacktestResult
            out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"portfolio_backtest_{datetime.now().strftime('%Y%m%d')}.html"
            ret_index = pd.to_datetime(bt.daily_returns.index)
            period = f"{ret_index.min().date()} ~ {ret_index.max().date()}" if len(ret_index) else ""
            dummy_res = SingleStockBacktestResult(
                symbol="portfolio",
                stock_name="组合",
                period=period,
                n_trades=0,
                win_rate=float("nan"),
                avg_hold_days=float("nan"),
                avg_return_per_trade=float("nan"),
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                total_return=float(summary.get("total_return", float("nan"))),
                annualized_return=float(summary.get("annualized_return", float("nan"))),
                buy_hold_return=float("nan"),
                buy_hold_annualized_return=float("nan"),
                excess_annualized_return=float("nan"),
                information_ratio=float("nan"),
                beta_to_benchmark=float("nan"),
                sharpe_ratio=float(summary.get("sharpe_ratio", float("nan"))),
                max_drawdown=float(summary.get("max_drawdown", float("nan"))),
                calmar_ratio=float(summary.get("calmar_ratio", float("nan"))),
                stop_loss_exits=0,
                trailing_stop_exits=0,
                daily_returns=bt.daily_returns,
            )
            generate_html_report(dummy_res, portfolio_df[portfolio_df["symbol"] == symbols[0]], output_path=path)
            print(f"HTML report written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
