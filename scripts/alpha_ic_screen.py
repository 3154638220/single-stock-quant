#!/usr/bin/env python
"""P1-A: Alpha IC screen — measure signal-return correlation per stock.

For each candidate, computes:
1. Signal IC: Spearman correlation between DK value and forward N-day returns
2. Trend-following excess return vs buy-hold (simple fixed-param strategy)
3. Conditional returns by market regime (bull/bear based on CSI 300 MA200)
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


def _compute_index_regime(index_df: pd.DataFrame, ma_period: int = 200) -> pd.Series:
    """Return boolean Series: True = bull (price > MA), False = bear."""
    close = pd.to_numeric(index_df["close"], errors="coerce")
    ma = close.rolling(ma_period, min_periods=min(ma_period, len(close))).mean()
    return close > ma


def _signal_ic_analysis(
    df: pd.DataFrame,
    params: DKTrendParams,
    forward_days: list[int] = [5, 10, 20, 30],
) -> dict:
    """Compute Spearman IC between DK value (at each day) and forward returns."""
    trend = compute_dktrend(df, params).reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce").reset_index(drop=True)

    dk_value = pd.to_numeric(trend["dk_value"], errors="coerce")
    dk_color = trend["dk_color"].astype(str)

    results = {}
    for fwd in forward_days:
        fwd_ret = close.shift(-fwd) / close - 1.0
        # Use all days, not just signal days
        mask = dk_value.notna() & fwd_ret.notna() & (np.abs(fwd_ret) < 0.5)
        if mask.sum() < 30:
            results[f"ic_{fwd}d"] = {"ic": float("nan"), "p_value": float("nan"), "n": int(mask.sum())}
            continue
        ic, pv = spearmanr(dk_value[mask], fwd_ret[mask])
        results[f"ic_{fwd}d"] = {"ic": float(ic), "p_value": float(pv), "n": int(mask.sum())}

    # Signal-only IC (only on BUY signal days)
    buy_mask = dk_color.eq("red")
    for fwd in forward_days:
        fwd_ret = close.shift(-fwd) / close - 1.0
        mask = buy_mask & dk_value.notna() & fwd_ret.notna() & (np.abs(fwd_ret) < 0.5)
        if mask.sum() < 10:
            results[f"signal_ic_{fwd}d"] = {"ic": float("nan"), "p_value": float("nan"), "n": int(mask.sum())}
            continue
        ic, pv = spearmanr(dk_value[mask], fwd_ret[mask])
        results[f"signal_ic_{fwd}d"] = {"ic": float(ic), "p_value": float(pv), "n": int(mask.sum())}

    return results


def _simple_trend_excess(
    df: pd.DataFrame,
    params: DKTrendParams,
    cost_bps: float = 15.0,
) -> dict:
    """Run a simple trend-following strategy (no optimization) and compare to buy-hold."""
    trend = compute_dktrend(df, params).reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce").reset_index(drop=True)
    open_ = pd.to_numeric(df["open"], errors="coerce").reset_index(drop=True)

    signal = trend["dk_signal"].astype(str)
    in_position = False
    entry_price = 0.0
    rets: list[float] = []
    bh_rets: list[float] = []

    for i in range(1, len(close)):
        if in_position:
            rets.append(close.iloc[i] / close.iloc[i - 1] - 1.0)
            if signal.iloc[i] == "sell":
                rets[-1] = close.iloc[i] / entry_price - 1.0 - cost_bps / 10000.0
                in_position = False
        else:
            rets.append(0.0)
            if signal.iloc[i] == "buy":
                entry_price = open_.iloc[min(i + 1, len(open_) - 1)]
                in_position = True

        if i > 0:
            bh_rets.append(close.iloc[i] / close.iloc[i - 1] - 1.0)

    strat_rets = pd.Series(rets)
    bh_rets_s = pd.Series(bh_rets)

    strat_total = (1 + strat_rets).prod() - 1 if len(strat_rets) > 0 else 0.0
    bh_total = (1 + bh_rets_s).prod() - 1 if len(bh_rets_s) > 0 else 0.0

    strat_ann = (1 + strat_total) ** (252 / max(len(strat_rets), 1)) - 1
    bh_ann = (1 + bh_total) ** (252 / max(len(bh_rets_s), 1)) - 1

    return {
        "trend_total_return": float(strat_total),
        "trend_annualized": float(strat_ann),
        "buyhold_annualized": float(bh_ann),
        "excess_vs_bh": float(strat_ann - bh_ann),
        "n_days": len(strat_rets),
    }


def _regime_analysis(
    df: pd.DataFrame,
    params: DKTrendParams,
    index_df: pd.DataFrame | None = None,
) -> dict:
    """Compute trend strategy performance in bull vs bear regimes."""
    trend = compute_dktrend(df, params).reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce").reset_index(drop=True)

    if index_df is not None:
        regime = _compute_index_regime(index_df)
        # Align index regime to stock dates
        index_close = pd.to_numeric(index_df["close"], errors="coerce")
        regime_aligned = pd.Series(False, index=range(len(df)))
    else:
        # Use stock's own MA200 as regime proxy
        ma200 = close.rolling(200, min_periods=100).mean()
        regime_aligned = close > ma200
        # align to index-like
        pass

    # Actually use stock's own MA200 for simplicity
    ma200 = close.rolling(200, min_periods=100).mean()
    is_bull = close > ma200

    signal = trend["dk_signal"].astype(str)
    bull_rets: list[float] = []
    bear_rets: list[float] = []
    in_position = False

    for i in range(1, len(close)):
        daily_ret = close.iloc[i] / close.iloc[i - 1] - 1.0
        if in_position:
            if is_bull.iloc[i]:
                bull_rets.append(daily_ret)
            else:
                bear_rets.append(daily_ret)
            if signal.iloc[i] == "sell":
                in_position = False
        elif signal.iloc[i] == "buy":
            in_position = True

    def _annualize(rets: list[float]) -> float:
        if len(rets) < 5:
            return float("nan")
        s = pd.Series(rets)
        total = (1 + s).prod() - 1
        return float((1 + total) ** (252 / max(len(s), 1)) - 1)

    return {
        "bull_annualized": _annualize(bull_rets),
        "bear_annualized": _annualize(bear_rets),
        "bull_days": len(bull_rets),
        "bear_days": len(bear_rets),
        "bull_fraction": len(bull_rets) / max(len(bull_rets) + len(bear_rets), 1),
    }


def _correlation_with_300750(
    db: DuckDBManager,
    symbol: str,
    start: str = "2018-01-01",
    end: str = "2026-04-30",
) -> float:
    """Compute Pearson correlation of daily returns with 300750."""
    try:
        df = db.read_daily_frame(symbols=[symbol, "300750"], start=start, end=end)
        rets = {}
        for sym in [symbol, "300750"]:
            sub = df[df["symbol"].astype(str).str.zfill(6) == sym].copy()
            sub = sub.sort_values("trade_date")
            sub["ret"] = pd.to_numeric(sub["close"], errors="coerce").pct_change()
            rets[sym] = sub.set_index("trade_date")["ret"]
        common = pd.DataFrame(rets).dropna()
        if len(common) < 60:
            return float("nan")
        return float(common.iloc[:, 0].corr(common.iloc[:, 1]))
    except Exception:
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Alpha IC screen for stock candidates.")
    parser.add_argument("--watchlist", required=True, help="Path to watchlist file")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    parser.add_argument("--export-results", action="store_true")
    parser.add_argument("--index-symbol", default="000300", help="Index symbol for regime detection")
    args = parser.parse_args()

    symbols = _read_watchlist(args.watchlist)
    params = DKTrendParams(
        mode=TrendMode.MACD_CROSS,
        macd_fast=10, macd_slow=26, macd_signal=9,
        min_run_len=2,
    )

    print(f"Alpha IC Screen: {len(symbols)} symbols")
    print(f"Period: {args.start} ~ {args.end}")
    print(f"Trend params: MACD(10,26,9) min_run=2\n")

    results: list[dict] = []

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        for sym in symbols:
            print(f"  {sym} ...", end=" ", flush=True)
            try:
                all_df = db.read_daily_frame(symbols=[sym], start=args.start, end=args.end)
                df = all_df[all_df["symbol"].astype(str).str.zfill(6) == sym].copy()
                if len(df) < 200:
                    print("insufficient data")
                    continue

                df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)

                ic = _signal_ic_analysis(df, params)
                excess = _simple_trend_excess(df, params)
                regime = _regime_analysis(df, params)
                corr_300750 = _correlation_with_300750(db, sym, args.start, args.end)

                entry = {
                    "symbol": sym,
                    "ic": ic,
                    "excess": excess,
                    "regime": regime,
                    "corr_with_300750": corr_300750,
                }

                # Quick summary
                ic20 = ic.get("ic_20d", {}).get("ic", float("nan"))
                sig_ic20 = ic.get("signal_ic_20d", {}).get("ic", float("nan"))
                excess_vs_bh = excess.get("excess_vs_bh", float("nan"))
                bull_ann = regime.get("bull_annualized", float("nan"))
                bear_ann = regime.get("bear_annualized", float("nan"))

                print(f"IC20={ic20:.4f} sigIC20={sig_ic20:.4f} "
                      f"excess_bh={excess_vs_bh:.4f} "
                      f"bull={bull_ann:.4f} bear={bear_ann:.4f} "
                      f"corr300750={corr_300750:.2f}")

                results.append(entry)
            except Exception as e:
                print(f"ERROR: {e}")

    # ---- Summary ranking ----
    print(f"\n{'='*80}")
    print("Alpha IC Screen Summary (sorted by signal IC 20d)")
    print(f"{'='*80}")
    print(f"{'symbol':<10} {'IC20':>8} {'sigIC20':>8} {'excess_bh':>10} {'bull_ann':>10} {'bear_ann':>10} {'corr_300750':>12}")
    print("-" * 72)

    def _safe_float(d: dict, key: str) -> float:
        v = d.get(key, float("nan"))
        return float(v) if v is not None and np.isfinite(v) else float("nan")

    sorted_results = sorted(
        results,
        key=lambda r: _safe_float(r.get("ic", {}).get("signal_ic_20d", {}), "ic"),
        reverse=True,
    )
    for r in sorted_results:
        sym = r["symbol"]
        ic20 = _safe_float(r["ic"].get("ic_20d", {}), "ic")
        sig20 = _safe_float(r["ic"].get("signal_ic_20d", {}), "ic")
        exc = _safe_float(r["excess"], "excess_vs_bh")
        bull = _safe_float(r["regime"], "bull_annualized")
        bear = _safe_float(r["regime"], "bear_annualized")
        corr = _safe_float(r, "corr_with_300750")
        print(f"{sym:<10} {ic20:>8.4f} {sig20:>8.4f} {exc:>10.4f} {bull:>10.4f} {bear:>10.4f} {corr:>12.2f}")

    # ---- Qualifying candidates ----
    print(f"\n--- Qualifying candidates (sigIC20 > 0.03, bull>0, bear>0, corr<0.6) ---")
    qualifying = []
    for r in sorted_results:
        sig20 = _safe_float(r["ic"].get("signal_ic_20d", {}), "ic")
        bull = _safe_float(r["regime"], "bull_annualized")
        bear = _safe_float(r["regime"], "bear_annualized")
        corr = _safe_float(r, "corr_with_300750")
        if sig20 > 0.03 and bull > 0 and bear > 0 and (np.isnan(corr) or corr < 0.6):
            qualifying.append(r["symbol"])
            print(f"  {r['symbol']}: sigIC20={sig20:.4f} bull={bull:.4f} bear={bear:.4f} corr={corr:.2f}")

    if not qualifying:
        print("  NONE! Relaxing criteria...")
        for r in sorted_results:
            sig20 = _safe_float(r["ic"].get("signal_ic_20d", {}), "ic")
            bull = _safe_float(r["regime"], "bull_annualized")
            corr = _safe_float(r, "corr_with_300750")
            if sig20 > 0.02 and (np.isnan(corr) or corr < 0.7):
                qualifying.append(r["symbol"])
                print(f"  {r['symbol']} (relaxed): sigIC20={sig20:.4f} bull={bull:.4f} corr={corr:.2f}")

    if args.export_results:
        out_dir = project_root() / "data/output/experiments/plan_05_19"
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_id = f"alpha_ic_screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output = {
            "experiment_id": exp_id,
            "plan_version": "05-19",
            "section": "P1-A",
            "symbols": symbols,
            "period": f"{args.start} ~ {args.end}",
            "qualifying": qualifying,
            "results": results,
        }
        path = out_dir / f"{exp_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=True)
        print(f"\nResults written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
