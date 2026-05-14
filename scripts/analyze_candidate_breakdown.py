#!/usr/bin/env python
"""Break down candidate forward returns by symbol, regime, and industry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.db_manager import DuckDBManager
from src.portfolio.attribution import compute_candidate_forward_return_breakdown
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


def _parse_horizons(raw: str) -> tuple[int, ...]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            vals.append(max(int(part), 1))
    if not vals:
        raise SystemExit("--horizons must contain at least one positive integer")
    return tuple(vals)


def _parse_group_by(raw: str) -> tuple[str, ...]:
    vals = tuple(part.strip() for part in raw.split(",") if part.strip())
    return vals or ("symbol",)


def _load_industry_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    df = pd.read_csv(Path(path).expanduser())
    lowered = {str(c).lower(): c for c in df.columns}
    symbol_col = lowered.get("symbol") or lowered.get("code")
    industry_col = lowered.get("industry") or lowered.get("industry_name")
    if not symbol_col or not industry_col:
        raise SystemExit("--industry-map must contain symbol/code and industry/industry_name columns")
    return {
        str(row[symbol_col]).zfill(6): str(row[industry_col])
        for _, row in df[[symbol_col, industry_col]].dropna().iterrows()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze which candidate groups contribute positive or negative forward returns."
    )
    parser.add_argument("--scores", required=True, help="Wide CSV of daily candidate/rank scores")
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--horizons", default="1,5,20")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--group-by", default="symbol,market_regime")
    parser.add_argument("--index-symbol", default=None, help="Benchmark/index symbol used for market regime")
    parser.add_argument("--industry-map", help="CSV with symbol/code and industry/industry_name columns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    risk_cfg = cfg.get("risk", {}) or {}
    benchmark_symbol = str(args.index_symbol or risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)
    symbols = _read_watchlist(Path(args.watchlist).expanduser())
    symbols_to_read = list(symbols)
    if benchmark_symbol not in symbols_to_read:
        symbols_to_read.append(benchmark_symbol)

    scores = pd.read_csv(Path(args.scores).expanduser(), index_col=0)
    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        data = db.read_daily_frame(symbols=symbols_to_read, start=args.start, end=args.end)
    if data.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    daily = data[data["symbol"].isin(symbols)].copy()
    index_df = data[data["symbol"] == benchmark_symbol].copy()
    if index_df.empty:
        index_df = None

    breakdown = compute_candidate_forward_return_breakdown(
        daily,
        scores,
        horizons=_parse_horizons(args.horizons),
        min_score=args.min_score,
        index_ohlcv=index_df,
        industry_map=_load_industry_map(args.industry_map),
        group_by=_parse_group_by(args.group_by),
    )

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(out, index=False)

    print(f"Candidate breakdown written to {out}")
    if breakdown.empty:
        print("No valid candidate/return pairs.")
        return 0

    max_horizon = int(breakdown["horizon"].max())
    worst = breakdown[breakdown["horizon"] == max_horizon].nsmallest(8, "mean_forward_return")
    print(f"Worst groups at horizon={max_horizon}:")
    for row in worst.to_dict("records"):
        groups = ", ".join(
            f"{k}={v}" for k, v in row.items()
            if k not in {
                "horizon",
                "n",
                "share_of_candidates",
                "score_mean",
                "mean_forward_return",
                "median_forward_return",
                "win_rate",
                "p10_forward_return",
                "p90_forward_return",
                "return_contribution",
            }
        )
        print(
            f"  {groups}: n={int(row['n'])} "
            f"mean={row['mean_forward_return']:.4f} "
            f"win={row['win_rate']:.2%} "
            f"share={row['share_of_candidates']:.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
