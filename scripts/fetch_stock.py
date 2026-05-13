#!/usr/bin/env python
"""Fetch one or more A-share daily series into DuckDB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

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


def _print_recent_quality(db: DuckDBManager, symbols: list[str], *, window: int = 30) -> None:
    print(f"recent quality summary (last {window} rows per symbol):")
    for sym in symbols:
        df = db.read_daily_frame(symbols=[sym])
        if df.empty:
            print(f"{sym}: no rows")
            continue
        recent = df.sort_values("trade_date").tail(window).copy()
        recent["trade_date"] = pd.to_datetime(recent["trade_date"])
        ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in recent.columns]
        nulls = int(recent[ohlcv_cols].isna().sum().sum()) if ohlcv_cols else 0
        invalid_ohlc = 0
        if {"open", "high", "low", "close"} <= set(recent.columns):
            invalid = (
                (recent["high"] < recent[["open", "close", "low"]].max(axis=1))
                | (recent["low"] > recent[["open", "close", "high"]].min(axis=1))
            )
            invalid_ohlc = int(invalid.sum())
        gaps = recent["trade_date"].diff().dt.days.dropna()
        max_gap = int(gaps.max()) if not gaps.empty else 0
        first = recent["trade_date"].iloc[0].date()
        last = recent["trade_date"].iloc[-1].date()
        close = float(recent["close"].iloc[-1]) if "close" in recent.columns else float("nan")
        print(
            f"{sym}: rows={len(recent)}, range={first}~{last}, "
            f"latest_close={close:.2f}, max_calendar_gap={max_gap}d, "
            f"null_ohlcv={nulls}, invalid_ohlc={invalid_ohlc}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch daily OHLCV for selected A-share symbols.")
    parser.add_argument("--symbol", help="Single 6-digit symbol, e.g. 600930")
    parser.add_argument("--symbols", nargs="+", help="Multiple 6-digit symbols")
    parser.add_argument("--start", default="20150101", help="Default start date for new symbols, YYYYMMDD")
    parser.add_argument("--end", help="End date, YYYYMMDD or YYYY-MM-DD. Default: today")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--check-quality", action="store_true", help="Print recent quality summary after fetch")
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
            _print_recent_quality(db, symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
