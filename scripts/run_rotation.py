#!/usr/bin/env python
"""Run multi-stock rotation backtest (Section 7, X1)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.backtest.rotation import run_rotation_backtest, RotationResult
from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, TrendMode
from src.settings import load_config, project_root


def _read_watchlist(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"watchlist not found: {path}")
    symbols = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            symbols.append(s.zfill(6))
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-stock rotation backtest.")
    parser.add_argument("--watchlist", required=True, help="Path to watchlist file")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--mode", choices=[m.value for m in TrendMode], default="donchian_breakout")
    parser.add_argument("--top-n", type=int, default=2, help="Number of stocks to hold simultaneously")
    parser.add_argument("--rebalance-freq", type=int, default=5, help="Rebalance every N trading days")
    parser.add_argument("--ranking-mode", choices=["trend_strength", "rs_momentum", "multi_factor"], default="trend_strength")
    parser.add_argument("--donchian-entry", type=int, default=20)
    parser.add_argument("--donchian-exit", type=int, default=10)
    parser.add_argument("--min-run-len", type=int, default=1)
    parser.add_argument("--atr-trailing-mult", type=float, default=2.0)
    parser.add_argument("--atr-trailing-min-gain", type=float, default=0.05)
    parser.add_argument("--intrapos-dd-limit", type=float, default=0.15)
    parser.add_argument("--profit-lock-trigger", type=float, default=0.12)
    parser.add_argument("--profit-lock-trailing", type=float, default=0.05)
    parser.add_argument("--stop-loss-pct", type=float, default=0.08)
    parser.add_argument("--time-stop-days", type=int, default=30)
    parser.add_argument("--time-stop-min-return", type=float, default=0.03)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--export-results", action="store_true")
    parser.add_argument("--bootstrap", action="store_true", help="Run block bootstrap CI (P3-B)")
    parser.add_argument("--experiment-id", help="Experiment ID for output filename")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    args = parser.parse_args()

    symbols = _read_watchlist(args.watchlist)
    if len(symbols) < args.top_n:
        raise SystemExit(f"Need at least {args.top_n} symbols, found {len(symbols)}")

    print(f"Rotation backtest: {len(symbols)} symbols, top {args.top_n}, "
          f"rebalance every {args.rebalance_freq} days, mode={args.mode}")

    # Load data for all symbols
    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        all_df = db.read_daily_frame(symbols=symbols, start=args.start, end=args.end)

    ohlcv_map: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = all_df[all_df["symbol"].astype(str).str.zfill(6) == sym].copy()
        if len(df) >= 100:
            ohlcv_map[sym] = df
        else:
            print(f"  Skipping {sym}: insufficient data ({len(df)} rows)")

    if len(ohlcv_map) < args.top_n:
        raise SystemExit(f"Not enough symbols with sufficient data: {len(ohlcv_map)}")

    print(f"  Loaded {len(ohlcv_map)} symbols with sufficient data")

    # Build trend params
    mode = TrendMode(str(args.mode))
    trend_params = DKTrendParams(
        mode=mode,
        donchian_entry_window=args.donchian_entry,
        donchian_exit_window=args.donchian_exit,
        min_run_len=args.min_run_len,
    )

    result = run_rotation_backtest(
        ohlcv_map,
        trend_params=trend_params,
        top_n=args.top_n,
        rebalance_freq=args.rebalance_freq,
        ranking_mode=args.ranking_mode,
        volume_confirm=True,
        stop_loss_pct=args.stop_loss_pct,
        atr_trailing_mult=args.atr_trailing_mult,
        atr_trailing_min_gain=args.atr_trailing_min_gain,
        intrapos_dd_limit=args.intrapos_dd_limit,
        profit_lock_trigger=args.profit_lock_trigger,
        profit_lock_trailing=args.profit_lock_trailing,
        time_stop_days=args.time_stop_days,
        time_stop_min_return=args.time_stop_min_return,
        cost_bps=args.cost_bps,
        initial_capital=args.initial_capital,
    )

    print(f"\n{'='*60}")
    print(f"Rotation Backtest Results")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(result.symbols)}")
    print(f"Total Return:     {result.total_return:.4f} ({result.total_return*100:.2f}%)")
    print(f"Annualized Return: {result.annualized_return:.4f} ({result.annualized_return*100:.2f}%)")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:      {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%)")
    print(f"Calmar Ratio:      {result.calmar_ratio:.2f}")
    print(f"Total Trades:      {result.n_trades}")
    print(f"Rotations:         {result.n_rotations}")

    if len(result.trade_log) > 0:
        tl = result.trade_log
        win_rate = (tl["return"] > 0).mean()
        avg_ret = tl["return"].mean()
        print(f"Win Rate:          {win_rate:.2%}")
        print(f"Avg Trade Return:  {avg_ret:.4f} ({avg_ret*100:.2f}%)")
        print(f"\nTrade breakdown by symbol:")
        for sym in sorted(tl["symbol"].unique()):
            sym_trades = tl[tl["symbol"] == sym]
            print(f"  {sym}: {len(sym_trades)} trades, "
                  f"win_rate={(sym_trades['return'] > 0).mean():.2%}, "
                  f"avg_ret={sym_trades['return'].mean():.4f}")
        print(f"\nExit reason distribution:")
        for reason in tl["exit_reason"].value_counts().index:
            print(f"  {reason}: {tl['exit_reason'].value_counts()[reason]}")

    # ---- Bootstrap confidence intervals (P3-B) ----
    if args.bootstrap:
        print(f"\n{'='*60}")
        print(f"Block Bootstrap (1000 iterations, block=21 days)")
        print(f"{'='*60}")
        daily = result.daily_returns.dropna()
        if len(daily) < 252:
            print("  Insufficient daily returns for bootstrap")
        else:
            rng = np.random.default_rng(42)
            n = len(daily)
            block_len = 21
            n_blocks = int(np.ceil(n / block_len))
            n_boot = 1000

            boot_ann = np.empty(n_boot, dtype=np.float64)
            boot_sharpe = np.empty(n_boot, dtype=np.float64)
            boot_mdd = np.empty(n_boot, dtype=np.float64)

            vals = daily.to_numpy(dtype=np.float64)
            for b in range(n_boot):
                # Block bootstrap: sample blocks with replacement
                block_starts = rng.integers(0, max(1, n - block_len), size=n_blocks)
                sample = []
                for start in block_starts:
                    sample.append(vals[start:start + block_len])
                boot_rets = np.concatenate(sample)[:n]

                ann = float(np.mean(boot_rets) * 252)
                std = float(np.std(boot_rets))
                sharpe = ann / (std * np.sqrt(252)) if std > 0 else 0.0

                equity = (1 + boot_rets).cumprod()
                running_max = np.maximum.accumulate(equity)
                dd = (equity - running_max) / running_max
                mdd = float(np.min(dd))

                boot_ann[b] = ann
                boot_sharpe[b] = sharpe
                boot_mdd[b] = mdd

            def _pctiles(arr, name):
                p5 = np.percentile(arr, 5)
                p25 = np.percentile(arr, 25)
                p50 = np.percentile(arr, 50)
                p75 = np.percentile(arr, 75)
                p95 = np.percentile(arr, 95)
                print(f"  {name}: median={p50:.4f} [P5: {p5:.4f}, P25: {p25:.4f}, P75: {p75:.4f}, P95: {p95:.4f}]")
                return {"p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95}

            ann_pct = _pctiles(boot_ann, "Annualized Return")
            sharpe_pct = _pctiles(boot_sharpe, "Sharpe Ratio")
            mdd_pct = _pctiles(boot_mdd, "Max Drawdown")

            p_ann_14 = float(np.mean(boot_ann >= 0.14))
            p5_sharpe_pos = float(np.percentile(boot_sharpe, 5) > 0)
            p95_mdd_ok = float(np.percentile(boot_mdd, 95) < 0.40)
            print(f"\n  P(ann >= 14%): {p_ann_14:.1%}  (target >= 70%)")
            print(f"  P5 Sharpe > 0: {p5_sharpe_pos:.1%}  (target = 100%)")
            print(f"  P95 MDD < 40%: {p95_mdd_ok:.1%}  (target = 100%)")

            bootstrap_result = {
                "n_iterations": n_boot,
                "block_length": block_len,
                "annualized_return": ann_pct,
                "sharpe": sharpe_pct,
                "max_drawdown": mdd_pct,
                "p_ann_ge_14pct": p_ann_14,
                "p5_sharpe_gt_0": p5_sharpe_pos,
                "p95_mdd_lt_40pct": p95_mdd_ok,
            }

    if args.export_results:
        out_dir = project_root() / "data/output/experiments/plan_05_19"
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_id = args.experiment_id or f"rotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output = {
            "experiment_id": exp_id,
            "plan_version": "05-19",
            "section": "X1",
            "symbols": result.symbols,
            "top_n": args.top_n,
            "rebalance_freq": args.rebalance_freq,
            "ranking_mode": args.ranking_mode,
            "trend_mode": str(mode.value),
            "metrics": {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "calmar_ratio": result.calmar_ratio,
                "n_trades": result.n_trades,
                "n_rotations": result.n_rotations,
            },
        }
        if args.bootstrap:
            output["bootstrap"] = bootstrap_result
        path = out_dir / f"{exp_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"\nResults written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
