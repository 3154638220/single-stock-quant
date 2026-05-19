#!/usr/bin/env python
"""P3-A: Full-history stress test across 5 distinct market regimes.

Tests rotation strategy robustness in: crash, trade-war bear, COVID crash,
new-energy bear (critical: 300750 -30%), and recent bull.
"""

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
import yaml

from src.backtest.rotation import run_rotation_backtest
from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, TrendMode
from src.settings import project_root


STRESS_PERIODS = [
    ("T1_crash", "2015-06-01", "2015-12-31", "2015 crash (CSI300 -40%)", 0.20, 0.30),
    ("T2_tradewar", "2018-01-01", "2019-01-31", "Trade-war bear", 0.0, -0.05),
    ("T3_covid", "2020-01-01", "2020-04-30", "COVID crash + V-recovery", 0.20, 0.30),
    ("T4_newenergy_bear", "2021-12-01", "2022-12-31", "New-energy bear (300750 -30%)", 0.05, 0.0),
    ("T5_recent_bull", "2024-07-01", "2026-04-30", "Recent bull market", 0.20, 0.15),
]


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


def _load_sector_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        data = yaml.safe_load(f)
    sector_map: dict[str, str] = {}
    for sector, symbols in data.get("sectors", {}).items():
        for sym in symbols:
            sector_map[str(sym).zfill(6)] = sector
    return sector_map


def _format_pct(v: float) -> str:
    if np.isnan(v):
        return "N/A"
    return f"{v*100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-history stress test for rotation strategy.")
    parser.add_argument("--watchlist", required=True, help="Path to watchlist file")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    parser.add_argument("--sector-map", default="configs/sector_map.yaml", help="Sector map YAML")
    parser.add_argument("--ranking-mode", choices=["trend_strength", "rs_momentum", "multi_factor"], default="rs_momentum")
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--rebalance-freq", type=int, default=10)
    parser.add_argument("--position-sizing", choices=["equal", "vol_inverse"], default="equal")
    parser.add_argument("--export-results", action="store_true")
    parser.add_argument("--market-regime-mode", choices=["off", "reduce", "exit"], default="off",
                        help="Market regime gating: off/reduce/exit (default: off)")
    parser.add_argument("--regime-ma-period", type=int, default=120,
                        help="MA period for regime detection (default: 120)")
    parser.add_argument("--regime-reduce-top-n", type=int, default=1,
                        help="Max positions in bear regime reduce mode (default: 1)")
    parser.add_argument("--index-symbol", default="000300",
                        help="Index symbol for market regime (default: 000300)")
    parser.add_argument("--regime-fast-ma-period", type=int, default=0,
                        help="Fast MA period for quick bear detection, 0=disable (default: 0)")
    parser.add_argument("--regime-fast-threshold", type=float, default=0.97,
                        help="Price threshold below fast MA to trigger bear (default: 0.97)")
    parser.add_argument("--regime-drawdown-trigger", type=float, default=0.0,
                        help="Index drawdown trigger threshold, 0=disable (default: 0.0)")
    parser.add_argument("--regime-drawdown-lookback", type=int, default=60,
                        help="Lookback days for drawdown rolling peak (default: 60)")
    parser.add_argument("--portfolio-dd-limit", type=float, default=0.0,
                        help="Portfolio equity DD limit, 0=disable (default: 0.0)")
    parser.add_argument("--stop-loss-pct", type=float, default=0.08,
                        help="Fixed stop loss pct (default: 0.08)")
    parser.add_argument("--atr-trailing-mult", type=float, default=2.0,
                        help="ATR trailing stop multiplier (default: 2.0)")
    parser.add_argument("--atr-trailing-min-gain", type=float, default=0.05,
                        help="Min gain to activate ATR trailing stop (default: 0.05)")
    parser.add_argument("--volatility-target-ann", type=float, default=0.0,
                        help="Target annualized volatility, 0=disable (default: 0.0)")
    parser.add_argument("--volatility-scale-floor", type=float, default=0.30,
                        help="Min position scale under vol targeting (default: 0.30)")
    parser.add_argument("--symbol-params", help="YAML file with per-symbol DKTrendParams")
    args = parser.parse_args()

    symbols = _read_watchlist(args.watchlist)
    sector_map = _load_sector_map(args.sector_map)

    trend_params = DKTrendParams(
        mode=TrendMode.DONCHIAN_BREAKOUT,
        donchian_entry_window=20,
        donchian_exit_window=10,
        min_run_len=1,
    )

    # Load per-symbol DK params if provided (J-1)
    symbol_params: dict[str, DKTrendParams] | None = None
    if args.symbol_params:
        sp_path = Path(args.symbol_params)
        if sp_path.exists():
            with open(sp_path) as f:
                sp_data = yaml.safe_load(f)
            symbol_params = {}
            for sym, cfg in sp_data.get("symbol_params", {}).items():
                sym_mode = TrendMode(str(cfg.get("mode", "donchian_breakout")))
                symbol_params[str(sym).zfill(6)] = DKTrendParams(
                    mode=sym_mode,
                    donchian_entry_window=cfg.get("donchian_entry_window", 20),
                    donchian_exit_window=cfg.get("donchian_exit_window", 10),
                    min_run_len=cfg.get("min_run_len", 1),
                    macd_fast=cfg.get("macd_fast", 12),
                    macd_slow=cfg.get("macd_slow", 26),
                    macd_signal=cfg.get("macd_signal", 9),
                )
            print(f"Loaded per-symbol params for {len(symbol_params)} symbols")

    print(f"Full-History Stress Test: {len(STRESS_PERIODS)} periods")
    print(f"Symbols: {symbols}")
    print(f"Params: top_n={args.top_n}, freq={args.rebalance_freq}, "
          f"ranking={args.ranking_mode}, sizing={args.position_sizing}")
    print()

    results = []
    all_passed = True

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        for period_id, start, end, desc, ann_target, ann_min in STRESS_PERIODS:
            print(f"{'='*70}")
            print(f"  {period_id}: {desc} ({start} ~ {end})")
            print(f"  Target: ann >= {ann_target:.0%}, Min: ann >= {ann_min:.0%}")
            print(f"{'='*70}")

            # Load data for this period
            all_df = db.read_daily_frame(symbols=symbols, start=start, end=end)
            ohlcv_map: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                df = all_df[all_df["symbol"].astype(str).str.zfill(6) == sym].copy()
                if len(df) >= 60:
                    ohlcv_map[sym] = df
                else:
                    print(f"  {sym}: only {len(df)} rows, skipping")

            # Load index data for regime gate (None = use pool composite fallback)
            index_df = None
            if args.market_regime_mode != "off":
                try:
                    idx_all = db.read_daily_frame(symbols=[args.index_symbol], start=start, end=end)
                    idx_sub = idx_all[idx_all["symbol"].astype(str).str.zfill(6) == args.index_symbol].copy()
                    if len(idx_sub) >= 30:
                        index_df = idx_sub
                except Exception:
                    pass  # fallback to pool composite in rotation.py

            valid_syms = list(ohlcv_map.keys())
            if len(valid_syms) < args.top_n:
                print(f"  SKIP: need {args.top_n} symbols, have {len(valid_syms)}")
                results.append({
                    "period": period_id, "description": desc,
                    "status": "skipped", "reason": f"insufficient symbols ({len(valid_syms)})",
                })
                continue

            try:
                result = run_rotation_backtest(
                    ohlcv_map,
                    trend_params=trend_params,
                    top_n=args.top_n,
                    rebalance_freq=args.rebalance_freq,
                    ranking_mode=args.ranking_mode,
                    position_sizing=args.position_sizing,
                    volume_confirm=True,
                    stop_loss_pct=args.stop_loss_pct,
                    atr_trailing_mult=args.atr_trailing_mult,
                    atr_trailing_min_gain=args.atr_trailing_min_gain,
                    intrapos_dd_limit=0.15,
                    profit_lock_trigger=0.12,
                    profit_lock_trailing=0.05,
                    time_stop_days=30,
                    time_stop_min_return=0.03,
                    cost_bps=15.0,
                    initial_capital=100_000.0,
                    sector_map=sector_map,
                    index_ohlcv=index_df,
                    market_regime_mode=args.market_regime_mode,
                    regime_ma_period=args.regime_ma_period,
                    regime_reduce_top_n=args.regime_reduce_top_n,
                    regime_fast_ma_period=args.regime_fast_ma_period,
                    regime_fast_threshold=args.regime_fast_threshold,
                    regime_drawdown_trigger=args.regime_drawdown_trigger,
                    regime_drawdown_lookback=args.regime_drawdown_lookback,
                    portfolio_dd_limit=args.portfolio_dd_limit,
                    volatility_target_ann=args.volatility_target_ann,
                    volatility_scale_floor=args.volatility_scale_floor,
                    symbol_params=symbol_params,
                )

                ann = result.annualized_return
                mdd = result.max_drawdown
                calmar = result.calmar_ratio
                n_trades = result.n_trades

                target_met = ann >= ann_target
                min_met = ann >= ann_min
                passed = min_met
                if not passed:
                    all_passed = False

                status = "PASS" if target_met else ("MIN" if min_met else "FAIL")
                print(f"  ann={_format_pct(ann)} mdd={_format_pct(mdd)} "
                      f"calmar={calmar:.2f} trades={n_trades} -> {status}")

                if period_id == "T4_newenergy_bear":
                    tl = result.trade_log
                    if len(tl) > 0:
                        has_300750 = (tl["symbol"] == "300750").any()
                        print(f"  [T4 critical] 300750 traded: {has_300750}")
                        if not has_300750:
                            print(f"  ==> Rotation successfully avoided 300750!")

                results.append({
                    "period": period_id,
                    "description": desc,
                    "status": status,
                    "annualized_return": ann,
                    "max_drawdown": mdd,
                    "calmar_ratio": calmar,
                    "n_trades": n_trades,
                    "target_met": target_met,
                    "min_met": min_met,
                    "symbols_available": valid_syms,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "period": period_id, "description": desc,
                    "status": "error", "error": str(e),
                })
                all_passed = False

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("Stress Test Summary")
    print(f"{'='*70}")
    print(f"{'Period':<25} {'Status':<8} {'Ann.Ret':>10} {'MaxDD':>10} {'Calmar':>8} {'Trades':>8}")
    print("-" * 75)
    for r in results:
        if r["status"] == "skipped":
            print(f"{r['period']:<25} {'SKIP':<8} {'-':>10} {'-':>10} {'-':>8} {'-':>8}")
        elif r["status"] == "error":
            print(f"{r['period']:<25} {'ERROR':<8} {'-':>10} {'-':>10} {'-':>8} {'-':>8}")
        else:
            print(f"{r['period']:<25} {r['status']:<8} {r['annualized_return']:>10.4f} "
                  f"{r['max_drawdown']:>10.4f} {r['calmar_ratio']:>8.2f} {r['n_trades']:>8}")

    passed_count = sum(1 for r in results if r.get("min_met", False))
    print(f"\nOverall: {passed_count}/{len(results)} periods passed minimum thresholds")
    print(f"ALL PASS: {'YES' if all_passed else 'NO — see T4 for rotation protection failure'}")

    if args.export_results:
        out_dir = project_root() / "data/output/experiments/plan_05_19"
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_id = f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output = {
            "experiment_id": exp_id,
            "plan_version": "05-19",
            "section": "P3-A",
            "params": {
                "top_n": args.top_n,
                "rebalance_freq": args.rebalance_freq,
                "ranking_mode": args.ranking_mode,
                "position_sizing": args.position_sizing,
            },
            "all_passed": all_passed,
            "results": results,
        }
        path = out_dir / f"{exp_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=True)
        print(f"\nResults written to {path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
