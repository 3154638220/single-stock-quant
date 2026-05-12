#!/usr/bin/env python
"""Fetch one or more A-share daily series into DuckDB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.db_manager import DuckDBManager


def _symbols(args: argparse.Namespace) -> list[str]:
    items: list[str] = []
    if args.symbol:
        items.append(args.symbol)
    if args.symbols:
        items.extend(args.symbols)
    out = []
    for item in items:
        code = str(item).strip().zfill(6)
        if len(code) != 6 or not code.isdigit():
            raise SystemExit(f"invalid A-share symbol: {item}")
        out.append(code)
    if not out:
        raise SystemExit("provide --symbol or --symbols")
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch daily OHLCV for selected A-share symbols.")
    parser.add_argument("--symbol", help="Single 6-digit symbol, e.g. 600930")
    parser.add_argument("--symbols", nargs="+", help="Multiple 6-digit symbols")
    parser.add_argument("--start", default="20150101", help="Default start date for new symbols, YYYYMMDD")
    parser.add_argument("--end", help="End date, YYYYMMDD or YYYY-MM-DD. Default: today")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--check-quality", action="store_true", help="Print table quality summary after fetch")
    args = parser.parse_args()

    symbols = _symbols(args)
    with DuckDBManager(config_path=args.config) as db:
        counts = db.incremental_update_many(symbols, default_start=args.start, end_date=args.end)
        for sym in symbols:
            result = counts.get(sym)
            rows = result.rows_written if result else 0
            failed = result.fetch_failed if result else False
            status = "FAILED" if failed else "OK"
            print(f"{sym}: {status}, rows_written={rows}")
        if args.check_quality:
            print(db.quality_report().summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
