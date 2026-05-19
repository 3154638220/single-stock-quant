#!/usr/bin/env python
"""P0-B: Rotation meta-parameter Walk-Forward validation.

Tests 24 parameter combinations across 4 independent sub-periods to verify
that top_n=2, rebalance_freq=10 is not an artifact of 2024-2026 overfitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.backtest.rotation import run_rotation_backtest
from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, TrendMode
from src.settings import project_root


SUB_PERIODS = [
    ("2018-2020", "2018-01-01", "2020-12-31", "mixed_bull_bear"),
    ("2020-2022", "2020-01-01", "2022-12-31", "covid_to_bear"),
    ("2022-2024H1", "2022-01-01", "2024-06-30", "bear_to_bull"),
    ("2024H2-2026", "2024-07-01", "2026-04-30", "recent_bull"),
]

ROTATION_GRID = {
    "top_n": [1, 2, 3],
    "rebalance_freq": [5, 10, 15],
    "ranking_mode": ["trend_strength", "rs_momentum", "multi_factor"],
}


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


def _load_ohlcv_map(db: DuckDBManager, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    all_df = db.read_daily_frame(symbols=symbols, start=start, end=end)
    ohlcv_map: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = all_df[all_df["symbol"].astype(str).str.zfill(6) == sym].copy()
        if len(df) >= 100:
            ohlcv_map[sym] = df
    return ohlcv_map


def _run_one(
    ohlcv_map: dict[str, pd.DataFrame],
    top_n: int,
    rebalance_freq: int,
    ranking_mode: str,
    trend_params: DKTrendParams,
) -> dict:
    try:
        result = run_rotation_backtest(
            ohlcv_map,
            trend_params=trend_params,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
            ranking_mode=ranking_mode,
            volume_confirm=True,
            stop_loss_pct=0.08,
            atr_trailing_mult=2.0,
            atr_trailing_min_gain=0.05,
            intrapos_dd_limit=0.15,
            profit_lock_trigger=0.12,
            profit_lock_trailing=0.05,
            time_stop_days=30,
            time_stop_min_return=0.03,
            cost_bps=15.0,
            initial_capital=100_000.0,
        )
        return {
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar_ratio": result.calmar_ratio,
            "n_trades": result.n_trades,
            "n_rotations": result.n_rotations,
            "error": None,
        }
    except Exception as e:
        return {
            "annualized_return": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "calmar_ratio": float("nan"),
            "n_trades": 0,
            "n_rotations": 0,
            "error": str(e),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rotation parameter grid across sub-periods.")
    parser.add_argument("--watchlist", required=True, help="Path to watchlist file")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    parser.add_argument("--mode", choices=[m.value for m in TrendMode], default="donchian_breakout")
    parser.add_argument("--donchian-entry", type=int, default=20)
    parser.add_argument("--donchian-exit", type=int, default=10)
    parser.add_argument("--min-run-len", type=int, default=1)
    parser.add_argument("--export-results", action="store_true")
    args = parser.parse_args()

    symbols = _read_watchlist(args.watchlist)
    mode = TrendMode(str(args.mode))
    trend_params = DKTrendParams(
        mode=mode,
        donchian_entry_window=args.donchian_entry,
        donchian_exit_window=args.donchian_exit,
        min_run_len=args.min_run_len,
    )

    combos = [
        dict(zip(ROTATION_GRID.keys(), vals))
        for vals in product(*ROTATION_GRID.values())
    ]
    print(f"Testing {len(combos)} parameter combinations across {len(SUB_PERIODS)} sub-periods")
    print(f"Symbols: {symbols}")

    all_results: list[dict] = []

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        for period_name, start, end, regime in SUB_PERIODS:
            print(f"\n{'='*60}")
            print(f"Period: {period_name} ({start} ~ {end}) [{regime}]")
            print(f"{'='*60}")

            ohlcv_map = _load_ohlcv_map(db, symbols, start, end)
            valid_symbols = list(ohlcv_map.keys())
            print(f"  Loaded {len(valid_symbols)} symbols: {valid_symbols}")

            period_results = []
            for combo in combos:
                top_n = combo["top_n"]
                if len(valid_symbols) < top_n:
                    period_results.append({**combo, "error": f"need {top_n} symbols, have {len(valid_symbols)}"})
                    continue

                metrics = _run_one(
                    ohlcv_map,
                    top_n=top_n,
                    rebalance_freq=combo["rebalance_freq"],
                    ranking_mode=combo["ranking_mode"],
                    trend_params=trend_params,
                )
                entry = {**combo, "period": period_name, "regime": regime, **metrics}
                period_results.append(entry)

                ann = metrics["annualized_return"]
                calmar = metrics["calmar_ratio"]
                mdd = metrics["max_drawdown"]
                err = metrics["error"]
                if err:
                    print(f"  top_n={combo['top_n']} freq={combo['rebalance_freq']} "
                          f"rank={combo['ranking_mode']} -> ERROR: {err}")
                else:
                    print(f"  top_n={combo['top_n']} freq={combo['rebalance_freq']} "
                          f"rank={combo['ranking_mode']} -> "
                          f"ann={ann:.4f} calmar={calmar:.2f} mdd={mdd:.4f} trades={metrics['n_trades']}")

            all_results.extend(period_results)

    # ---- Cross-period summary ----
    print(f"\n{'='*80}")
    print("Cross-Period Consistency Analysis")
    print(f"{'='*80}")

    df = pd.DataFrame(all_results)
    df_valid = df[df["error"].isna() & df["annualized_return"].notna()].copy()

    if len(df_valid) == 0:
        print("No valid results to analyze.")
        return 1

    # Rank combos within each period by annualized return
    df_valid["rank_in_period"] = df_valid.groupby("period")["annualized_return"].rank(ascending=False)
    df_valid["rank_calmar"] = df_valid.groupby("period")["calmar_ratio"].rank(ascending=False)

    # For each combo, compute average rank across periods
    combo_key = df_valid.groupby(["top_n", "rebalance_freq", "ranking_mode"])
    avg_ranks = combo_key.agg(
        avg_ann_rank=("rank_in_period", "mean"),
        avg_calmar_rank=("rank_calmar", "mean"),
        avg_ann=("annualized_return", "mean"),
        avg_calmar=("calmar_ratio", "mean"),
        avg_mdd=("max_drawdown", "mean"),
        n_periods=("period", "nunique"),
    ).reset_index()

    avg_ranks = avg_ranks.sort_values("avg_ann_rank")

    print(f"\n{'top_n':<8} {'freq':<8} {'rank_mode':<18} {'avg_ann_rank':<14} {'avg_calmar_rank':<16} {'avg_ann':<10} {'avg_calmar':<10} {'avg_mdd':<10} {'n_periods':<10}")
    print("-" * 110)
    for _, row in avg_ranks.iterrows():
        print(f"{int(row['top_n']):<8} {int(row['rebalance_freq']):<8} {row['ranking_mode']:<18} "
              f"{row['avg_ann_rank']:<14.2f} {row['avg_calmar_rank']:<16.2f} "
              f"{row['avg_ann']:<10.4f} {row['avg_calmar']:<10.2f} {row['avg_mdd']:<10.4f} {int(row['n_periods']):<10}")

    # Check current params (top_n=2, freq=10)
    current = avg_ranks[(avg_ranks["top_n"] == 2) & (avg_ranks["rebalance_freq"] == 10)]
    print(f"\n--- Current params (top_n=2, rebalance_freq=10) ---")
    for _, row in current.iterrows():
        total_combos = len(avg_ranks)
        top_third = total_combos / 3
        in_top_third = row["avg_ann_rank"] <= top_third
        print(f"  ranking_mode={row['ranking_mode']}: avg_ann_rank={row['avg_ann_rank']:.1f}/{total_combos} "
              f"(in top 1/3: {'YES' if in_top_third else 'NO'})")

    # Check top_n consistency
    print(f"\n--- Parameter consistency across periods ---")
    for param in ["top_n", "rebalance_freq"]:
        best_per_period = df_valid.loc[df_valid.groupby("period")["annualized_return"].idxmax()]
        mode_val = best_per_period[param].mode()
        consistency = len(mode_val) > 0
        if consistency:
            print(f"  {param}: best value = {mode_val.iloc[0]} in "
                  f"{(best_per_period[param] == mode_val.iloc[0]).sum()}/{len(SUB_PERIODS)} periods")

    if args.export_results:
        out_dir = project_root() / "data/output/experiments/plan_05_19"
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_id = f"rotation_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output = {
            "experiment_id": exp_id,
            "plan_version": "05-19",
            "section": "P0-B",
            "n_combos": len(combos),
            "n_periods": len(SUB_PERIODS),
            "symbols": symbols,
            "trend_mode": str(mode.value),
            "grid_results": all_results,
            "cross_period_summary": avg_ranks.to_dict(orient="records"),
        }
        path = out_dir / f"{exp_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"\nResults written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
