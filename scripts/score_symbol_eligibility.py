#!/usr/bin/env python
"""S5: Score symbol eligibility for DK trend strategy.

Computes a composite score (0-100) for each symbol based on:
  1. Trend quality (30%): consistency of price direction during DK red periods
  2. Rolling IS Sharpe (40%): 2-year rolling window backtest Sharpe
  3. Signal frequency (15%): annualised BUY signals (target 4-12)
  4. Liquidity (15%): average daily turnover in yuan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.config import build_bt_kwargs
from src.backtest.single_stock import run_single_stock_backtest
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams, compute_dktrend
from src.settings import load_config


def _read_watchlist(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            symbols.append(s.zfill(6))
    return symbols


def _trend_quality_score(df: pd.DataFrame, params: DKTrendParams) -> float:
    """Fraction of DK red days where close > previous close (price rising)."""
    trend = compute_dktrend(df, params)
    red_mask = trend["dk_color"] == "red"
    if red_mask.sum() < 10:
        return 0.0
    # Align: trend has trade_date index, df has integer index
    close = pd.to_numeric(df["close"], errors="coerce")
    rising = (close.diff() > 0).values
    red_indices = trend.index[red_mask]
    df_dates = pd.to_datetime(df["trade_date"]).dt.normalize()
    # Map trend dates back to df positions
    aligned = []
    for d in red_indices:
        pos = df_dates[df_dates == d].index
        if len(pos) > 0:
            i = pos[0]
            if i > 0 and i < len(rising):
                aligned.append(rising[i])
    if not aligned:
        return 0.0
    return float(np.mean(aligned))


def _signal_frequency_score(df: pd.DataFrame, params: DKTrendParams) -> tuple[float, float]:
    """Annual BUY signals. Returns (score, annual_signals)."""
    trend = compute_dktrend(df, params)
    n_buy = int((trend["dk_signal"] == "buy").sum())
    n_days = len(df)
    n_years = max(n_days / 252.0, 0.5)
    annual = n_buy / n_years
    # Score: 4-12 per year is ideal, penalties outside
    if 4 <= annual <= 12:
        return 1.0, annual
    elif annual < 1:
        return 0.0, annual
    elif annual < 4:
        return annual / 4.0, annual
    elif annual <= 20:
        return 1.0 - (annual - 12) / 8.0, annual
    else:
        return 0.0, annual


def _liquidity_score(df: pd.DataFrame) -> tuple[float, float]:
    """Score based on average daily turnover. Returns (score, avg_daily_amount)."""
    if "volume" not in df.columns or "close" not in df.columns:
        return 0.0, 0.0
    amount = pd.to_numeric(df["volume"], errors="coerce") * pd.to_numeric(df["close"], errors="coerce")
    avg_daily = float(amount.mean())
    # 200M (2e8) qualified, below 50M is untradable
    if avg_daily >= 5e8:
        return 1.0, avg_daily
    elif avg_daily >= 2e8:
        return 0.8, avg_daily
    elif avg_daily >= 1e8:
        return 0.5, avg_daily
    elif avg_daily >= 5e7:
        return 0.3, avg_daily
    else:
        return 0.1, avg_daily


def _rolling_sharpe_score(
    symbol: str,
    df: pd.DataFrame,
    params: DKTrendParams,
    base_kwargs: dict,
    *,
    window_years: int = 2,
) -> tuple[float, list[float]]:
    """Rolling 2-year backtest Sharpe scores. Returns (median_score, sharpes)."""
    window_days = window_years * 252
    step_days = 126  # half-year steps
    sharpes = []
    start = 0
    while start + window_days <= len(df):
        window_df = df.iloc[start:start + window_days].copy()
        try:
            res = run_single_stock_backtest(symbol, window_df, params, **base_kwargs)
            if np.isfinite(res.sharpe_ratio):
                sharpes.append(res.sharpe_ratio)
        except Exception:
            pass
        start += step_days

    if not sharpes:
        return 0.0, []

    median_sh = float(np.median(sharpes))
    # Map median Sharpe to 0-1 score
    if median_sh >= 0.5:
        score = 1.0
    elif median_sh >= 0.25:
        score = 0.5 + (median_sh - 0.25) / 0.25 * 0.5
    elif median_sh > 0:
        score = 0.25 + median_sh / 0.25 * 0.25
    elif median_sh > -0.25:
        score = 0.25 + median_sh / 0.25 * 0.25
    else:
        score = 0.0
    return max(0.0, min(1.0, score)), sharpes


def main() -> int:
    parser = argparse.ArgumentParser(description="S5: Score symbol eligibility for DK strategy")
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--duckdb-path")
    parser.add_argument("--stock-name-cache")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = _read_watchlist(Path(args.watchlist).expanduser())
    name_cache_path = Path(args.stock_name_cache).expanduser() if args.stock_name_cache else resolve_stock_name_cache_path(cfg)
    names = resolve_stock_names(symbols, name_cache_path)

    risk_cfg = cfg.get("risk", {}) or {}
    benchmark_symbol = str(risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)
    symbols_to_read = list(symbols)
    if benchmark_symbol not in symbols_to_read:
        symbols_to_read.append(benchmark_symbol)

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        data = db.read_daily_frame(symbols=symbols_to_read, start=args.start, end=args.end)

    index_df = data[data["symbol"].astype(str) == benchmark_symbol].copy() if benchmark_symbol in set(data.get("symbol", [])) else None
    base_kwargs = build_bt_kwargs(cfg, index_ohlcv=index_df)
    base_kwargs["consensus_n_agree"] = None
    params = DKTrendParams.from_mapping(dict(cfg.get("trend_signal", {}) or {}))

    rows = []
    for symbol in symbols:
        df = data[data["symbol"].astype(str) == symbol].copy()
        if df.empty:
            rows.append({"symbol": symbol, "stock_name": names.get(symbol, symbol),
                         "total_score": 0, "grade": "no_data"})
            continue

        kwargs = dict(base_kwargs)
        kwargs["stock_name"] = names.get(symbol, symbol)

        tq = _trend_quality_score(df, params)
        sf, annual_sig = _signal_frequency_score(df, params)
        liq, avg_amt = _liquidity_score(df)
        rs, sharpes = _rolling_sharpe_score(symbol, df, params, kwargs)

        total = 0.30 * tq + 0.40 * rs + 0.15 * sf + 0.15 * liq
        if total >= 0.60:
            grade = "green"
        elif total >= 0.40:
            grade = "watch"
        else:
            grade = "exclude"

        rows.append({
            "symbol": symbol,
            "stock_name": names.get(symbol, symbol),
            "trend_quality": round(tq, 3),
            "rolling_sharpe_score": round(rs, 3),
            "signal_freq_score": round(sf, 3),
            "annual_signals": round(annual_sig, 1),
            "liquidity_score": round(liq, 3),
            "avg_daily_amount": round(avg_amt, 0),
            "rolling_sharpes": ",".join(f"{s:.2f}" for s in sharpes),
            "total_score": round(total * 100, 1),
            "grade": grade,
        })

        print(f"  {symbol} ({names.get(symbol, symbol)}): score={total*100:.0f}, grade={grade}, "
              f"tq={tq:.2f}, rs={rs:.2f}, sf={sf:.2f}, liq={liq:.2f}")

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)

    # Summary
    greens = sum(1 for r in rows if r["grade"] == "green")
    watches = sum(1 for r in rows if r["grade"] == "watch")
    excludes = sum(1 for r in rows if r["grade"] == "exclude")
    print(f"\nSymbol eligibility: {greens} green, {watches} watch, {excludes} exclude")
    print(f"Written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
