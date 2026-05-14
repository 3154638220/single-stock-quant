#!/usr/bin/env python
"""Analyze whether portfolio rank scores predict future returns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.db_manager import DuckDBManager
from src.portfolio.attribution import (
    compute_score_forward_return_attribution,
    summarize_score_monotonicity,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Bucket portfolio rank scores by future open-to-open returns.")
    parser.add_argument("--scores", required=True, help="Wide CSV of daily rank scores")
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--horizons", default="1,5,20")
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--output", required=True, help="Bucket attribution CSV")
    parser.add_argument("--summary-output", help="Monotonicity summary CSV")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    args = parser.parse_args()

    symbols = _read_watchlist(Path(args.watchlist).expanduser())
    scores = pd.read_csv(Path(args.scores).expanduser(), index_col=0)

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        data = db.read_daily_frame(symbols=symbols, start=args.start, end=args.end)
    if data.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    attribution = compute_score_forward_return_attribution(
        data,
        scores,
        horizons=_parse_horizons(args.horizons),
        n_quantiles=args.n_quantiles,
        min_score=args.min_score,
    )
    summary = summarize_score_monotonicity(attribution)

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    attribution.to_csv(out, index=False)

    summary_out = Path(args.summary_output).expanduser() if args.summary_output else out.with_name(f"{out.stem}_summary.csv")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)

    print("Ranking attribution summary:")
    if summary.empty:
        print("  no valid score/return pairs")
    else:
        for row in summary.to_dict("records"):
            print(
                f"  horizon={int(row['horizon'])}: "
                f"top-bottom={row['top_minus_bottom']:.4f} "
                f"corr={row['bucket_return_corr']:.3f} "
                f"monotonic={bool(row['is_monotonic'])}"
            )
    print(f"Bucket attribution written to {out}")
    print(f"Summary written to {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
