#!/usr/bin/env python
"""Quality-score bucket analysis across the watchlist.

Runs the same strategy with min_quality_score at 0, 20, 40, 60 and compares
key metrics per bucket to validate whether higher-quality signals produce
better risk-adjusted returns.
"""

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
from src.indicators import DKTrendParams, TrendMode
from src.settings import load_config


def _read_watchlist(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            symbols.append(s.zfill(6))
    return symbols


def main() -> int:
    cfg = load_config()
    trend_cfg = cfg.get("trend_signal", {}) or {}
    mode = str(trend_cfg.get("mode", "macd_cross"))
    symbols = _read_watchlist(Path("configs/watchlist_25.txt"))
    name_cache_path = resolve_stock_name_cache_path(cfg)
    names = resolve_stock_names(symbols, name_cache_path)

    quality_thresholds = [0, 20, 40, 60]
    start = "2020-01-01"
    end = "2026-05-08"

    with DuckDBManager() as db:
        data = db.read_daily_frame(symbols=symbols, start=start, end=end)

    if data.empty:
        raise SystemExit("no daily data found")

    all_rows = []
    for threshold in quality_thresholds:
        params = DKTrendParams.from_mapping({"mode": mode})
        for symbol in symbols:
            df = data[data["symbol"].astype(str) == symbol].copy()
            if df.empty:
                continue
            res = run_single_stock_backtest(
                symbol,
                df,
                params,
                cost_bps=15,
                initial_capital=100_000,
                stock_name=names.get(symbol, symbol),
                min_quality_score=float(threshold),
                quality_score_mode="hard",
            )
            all_rows.append(
                {
                    "quality_threshold": threshold,
                    "symbol": res.symbol,
                    "stock_name": res.stock_name,
                    "n_trades": res.n_trades,
                    "total_return": res.total_return,
                    "annualized_return": res.annualized_return,
                    "sharpe_ratio": res.sharpe_ratio,
                    "max_drawdown": res.max_drawdown,
                    "calmar_ratio": res.calmar_ratio,
                    "win_rate": res.win_rate,
                }
            )

    df_all = pd.DataFrame(all_rows)
    out_path = Path("data/output/quality_bucket_analysis.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)

    # Print summary per bucket
    print("质量评分分桶分析")
    print("=" * 80)
    for threshold in quality_thresholds:
        subset = df_all[df_all["quality_threshold"] == threshold]
        if subset.empty:
            continue
        n = len(subset)
        print(f"\nmin_quality_score = {threshold} ({n} stocks)")
        print(
            f"  年化中位数: {subset['annualized_return'].median():.2%}  "
            f"Sharpe中位数: {subset['sharpe_ratio'].median():.2f}  "
            f"Calmar中位数: {subset['calmar_ratio'].median():.2f}  "
            f"MDD中位数: {subset['max_drawdown'].median():.2%}  "
            f"交易数中位数: {subset['n_trades'].median():.0f}  "
            f"胜率中位数: {subset['win_rate'].median():.1%}"
        )
        print(f"  Sharpe>0: {(subset['sharpe_ratio'] > 0).sum()}/{n}  "
              f"Calmar>0.5: {(subset['calmar_ratio'] > 0.5).sum()}/{n}  "
              f"年均交易>=3: {(subset['n_trades'] >= 3*6.3).sum()}/{n}")

    print(f"\n完整结果写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
