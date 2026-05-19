#!/usr/bin/env python
"""P1-B: Measure IC for each ranking factor across rotation pool stocks.

Computes IC (Spearman correlation) between each candidate ranking factor
and forward N-day returns. Outputs IC mean, ICIR, and optimal weights.
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
from scipy.stats import spearmanr

from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.settings import project_root


FORWARD_DAYS = [5, 10, 20, 30]


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


def _compute_factors(df: pd.DataFrame, params: DKTrendParams) -> pd.DataFrame:
    """Compute all candidate ranking factors for each day."""
    trend = compute_dktrend(df, params).reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce").reset_index(drop=True)
    high = pd.to_numeric(df["high"], errors="coerce").reset_index(drop=True)
    low = pd.to_numeric(df["low"], errors="coerce").reset_index(drop=True)

    factors = pd.DataFrame(index=df.index)

    # dk_value from trend
    factors["dk_value"] = pd.to_numeric(trend["dk_value"], errors="coerce")

    # run_len normalized
    run_len = pd.to_numeric(trend["dk_run_len"], errors="coerce")
    factors["run_len_norm"] = run_len.clip(upper=20) / 20.0

    # rs_20: 20-day return
    factors["rs_20"] = close.pct_change(20)

    # rs_60: 60-day return
    factors["rs_60"] = close.pct_change(60)

    # vol_adj: inverse of 20-day annualized volatility
    ret_1d = close.pct_change()
    vol_20 = ret_1d.rolling(20, min_periods=10).std() * np.sqrt(252)
    factors["vol_adj"] = 1.0 / (vol_20 + 0.01)

    # above_ma120: binary
    ma120 = close.rolling(120, min_periods=60).mean()
    factors["above_ma120"] = (close > ma120).astype(float)

    # dk_combined: current heuristic dk_value * (1 + run_len/10)
    factors["dk_combined"] = factors["dk_value"] * (1.0 + run_len / 10.0)

    # Forward returns
    for fwd in FORWARD_DAYS:
        factors[f"fwd_ret_{fwd}d"] = close.shift(-fwd) / close - 1.0

    return factors, trend


def _compute_ic(factors: pd.DataFrame, factor_name: str, fwd: int, signal_only: bool = False, trend: pd.DataFrame | None = None) -> dict:
    """Compute Spearman IC for one factor and forward period."""
    fwd_col = f"fwd_ret_{fwd}d"
    if fwd_col not in factors.columns:
        return {"ic": float("nan"), "icir": float("nan"), "n": 0, "p_value": float("nan")}

    fvals = factors[factor_name]
    fwd_vals = factors[fwd_col]

    if signal_only and trend is not None:
        mask = trend["dk_signal"].astype(str).eq("buy")
        fvals = fvals[mask]
        fwd_vals = fwd_vals[mask]

    mask = fvals.notna() & fwd_vals.notna() & (np.abs(fwd_vals) < 0.5)
    if mask.sum() < 20:
        return {"ic": float("nan"), "icir": float("nan"), "n": int(mask.sum()), "p_value": float("nan")}

    ic, pv = spearmanr(fvals[mask], fwd_vals[mask])
    return {"ic": float(ic), "p_value": float(pv), "n": int(mask.sum()), "icir": float("nan")}


def _rolling_ic(factors: pd.DataFrame, factor_name: str, fwd: int, window: int = 252) -> dict:
    """Compute rolling IC (252-day windows) to get ICIR."""
    fwd_col = f"fwd_ret_{fwd}d"
    fvals = factors[factor_name].values
    fwd_vals = factors[fwd_col].values
    mask = np.isfinite(fvals) & np.isfinite(fwd_vals) & (np.abs(fwd_vals) < 0.5)

    rolling_ics = []
    for i in range(window, len(fvals)):
        end = i
        start = i - window
        w_mask = mask[start:end]
        if w_mask.sum() < 30:
            continue
        w_f = fvals[start:end][w_mask]
        w_r = fwd_vals[start:end][w_mask]
        if np.std(w_f) == 0 or np.std(w_r) == 0:
            continue
        ic, _ = spearmanr(w_f, w_r)
        rolling_ics.append(ic)

    if len(rolling_ics) < 10:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "icir": float("nan"), "n_windows": len(rolling_ics)}

    arr = np.array(rolling_ics)
    ic_mean = float(np.mean(arr))
    ic_std = float(np.std(arr))
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": ic_mean / ic_std if ic_std > 0 else float("nan"),
        "n_windows": len(rolling_ics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure IC for ranking factors.")
    parser.add_argument("--watchlist", required=True, help="Path to watchlist file")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    parser.add_argument("--export-results", action="store_true")
    args = parser.parse_args()

    symbols = _read_watchlist(args.watchlist)
    params = DKTrendParams(
        mode=TrendMode.MACD_CROSS,
        macd_fast=10, macd_slow=26, macd_signal=9,
        min_run_len=2,
    )

    factor_names = ["dk_value", "run_len_norm", "rs_20", "rs_60", "vol_adj", "above_ma120", "dk_combined"]

    print(f"Ranking Factor IC Analysis: {len(symbols)} symbols")
    print(f"Factors: {factor_names}")
    print(f"Forward windows: {FORWARD_DAYS}\n")

    all_ics: dict[str, list[float]] = {f"{fn}_{fwd}d": [] for fn in factor_names for fwd in FORWARD_DAYS}
    all_icirs: dict[str, list[float]] = {f"{fn}_{fwd}d": [] for fn in factor_names for fwd in FORWARD_DAYS}

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        for sym in symbols:
            all_df = db.read_daily_frame(symbols=[sym], start=args.start, end=args.end)
            df = all_df[all_df["symbol"].astype(str).str.zfill(6) == sym].copy()
            if len(df) < 200:
                print(f"  {sym}: insufficient data")
                continue
            df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)

            factors, trend = _compute_factors(df, params)

            for fn in factor_names:
                for fwd in FORWARD_DAYS:
                    key = f"{fn}_{fwd}d"
                    ric = _rolling_ic(factors, fn, fwd)
                    if np.isfinite(ric["icir"]):
                        all_icirs[key].append(ric["icir"])
                    ic_info = _compute_ic(factors, fn, fwd)
                    if np.isfinite(ic_info["ic"]):
                        all_ics[key].append(ic_info["ic"])

    # ---- Pooled summary across all stocks ----
    print(f"\n{'='*90}")
    print("Pooled Factor IC Analysis (mean ICIR across stocks, 20d forward)")
    print(f"{'='*90}")
    print(f"{'Factor':<18} {'IC20_mean':>10} {'IC20_std':>10} {'ICIR20_mean':>12} {'ICIR20_std':>12} {'N_stocks':>10}")
    print("-" * 75)

    summary = []
    for fn in factor_names:
        key = f"{fn}_20d"
        ics = [v for v in all_ics[key] if np.isfinite(v)]
        icirs = [v for v in all_icirs[key] if np.isfinite(v)]
        if ics:
            summary.append({
                "factor": fn,
                "ic_mean": float(np.mean(ics)),
                "ic_std": float(np.std(ics)),
                "icir_mean": float(np.mean(icirs)) if icirs else float("nan"),
                "icir_std": float(np.std(icirs)) if icirs else float("nan"),
                "n_stocks": len(ics),
            })

    summary.sort(key=lambda x: abs(x["icir_mean"]) if np.isfinite(x["icir_mean"]) else 0, reverse=True)
    for s in summary:
        print(f"{s['factor']:<18} {s['ic_mean']:>10.4f} {s['ic_std']:>10.4f} "
              f"{s['icir_mean']:>12.4f} {s['icir_std']:>12.4f} {s['n_stocks']:>10}")

    # ---- Determine optimal weights from ICIR ----
    positive_icirs = [s for s in summary if np.isfinite(s["icir_mean"]) and s["icir_mean"] > 0]
    if positive_icirs:
        total_icir = sum(s["icir_mean"] for s in positive_icirs)
        print(f"\n--- Suggested factor weights (ICIR-proportional, only ICIR>0 factors) ---")
        for s in positive_icirs:
            w = s["icir_mean"] / total_icir
            print(f"  {s['factor']}: weight={w:.3f}")

    if args.export_results:
        out_dir = project_root() / "data/output/experiments/plan_05_19"
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_id = f"ranking_ic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output = {
            "experiment_id": exp_id,
            "plan_version": "05-19",
            "section": "P1-B",
            "factor_summary": summary,
            "suggested_weights": {s["factor"]: s["icir_mean"] / total_icir for s in positive_icirs} if positive_icirs else {},
        }
        path = out_dir / f"{exp_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=True)
        print(f"\nResults written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
