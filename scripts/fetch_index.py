#!/usr/bin/env python
"""Fetch CSI300 (000300) index daily data into DuckDB for regime gate.

Usage:
    python scripts/fetch_index.py --symbol 000300
    python scripts/fetch_index.py --symbol 000300 --check-quality
    python scripts/fetch_index.py --all  # fetch all default indexes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_fetcher.akshare_client import fetch_index_daily, fill_derived_daily_fields
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.index_benchmarks import DEFAULT_INDEX_SPECS, IndexFetchSpec, parse_index_specs
from src.settings import load_config


def _write_frame(db: DuckDBManager, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    payload = fill_derived_daily_fields(
        df.sort_values(["symbol", "trade_date"])
        .drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        .reset_index(drop=True)
    )
    # Align to table schema: ensure required columns, drop extras
    for col in ("amount", "turnover"):
        if col not in payload.columns:
            payload[col] = None
    for col in ("name", "source_symbol"):
        if col in payload.columns:
            payload = payload.drop(columns=[col])
    payload["trade_date"] = pd.to_datetime(payload["trade_date"]).dt.date
    schema_cols = ["symbol", "trade_date", "open", "close", "high", "low", "volume",
                   "amount", "amplitude_pct", "pct_chg", "change", "turnover"]
    payload = payload[schema_cols]
    db.connection.register("df_fetch_index_manual", payload)
    try:
        db.connection.execute(
            f"""
            INSERT OR REPLACE INTO {table}
            SELECT * FROM df_fetch_index_manual
            """
        )
    finally:
        db.connection.unregister("df_fetch_index_manual")
    return int(len(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch index daily data into DuckDB.")
    parser.add_argument("--symbol", help="Output symbol (6-digit), e.g. 000300")
    parser.add_argument("--all", action="store_true", help="Fetch all default indexes")
    parser.add_argument("--index", nargs="+", help="Index specs: name:output_symbol:akshare_symbol")
    parser.add_argument("--start", default="20150101", help="Start date, YYYYMMDD")
    parser.add_argument("--end", help="End date, YYYYMMDD. Default: today")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--db-table", help="DuckDB table; defaults to configured daily table")
    parser.add_argument("--check-quality", action="store_true", help="Print recent quality summary after fetch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    table = args.db_table or cfg.get("database", {}).get("table_daily", "a_share_daily")
    end_s = args.end or pd.Timestamp.today().strftime("%Y%m%d")

    if args.index:
        specs = list(parse_index_specs(args.index))
    elif args.all:
        specs = list(DEFAULT_INDEX_SPECS)
    elif args.symbol:
        code = str(args.symbol).strip().zfill(6)
        specs = [IndexFetchSpec("custom", code, f"sh{code}")]
    else:
        parser.print_help()
        return 1

    with DuckDBManager(config_path=args.config, table_daily=args.db_table) as db:
        for spec in specs:
            try:
                existing_last = db.last_trade_date(spec.output_symbol)
                start_s = (
                    (pd.Timestamp(existing_last) + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    if existing_last
                    else args.start
                )
                if pd.Timestamp(start_s) > pd.Timestamp(end_s):
                    print(f"{spec.name} ({spec.output_symbol}): up to date")
                    continue
                df = fetch_index_daily(spec, start_s, end_s, config=cfg)
                rows = _write_frame(db, table, df)
                print(f"{spec.name} ({spec.output_symbol}): OK, rows_written={rows}")
            except Exception as exc:
                print(f"{spec.name} ({spec.output_symbol}): FAILED, {type(exc).__name__}: {exc}")
                return 1

        if args.check_quality:
            for spec in specs:
                df = db.read_daily_frame(symbols=[spec.output_symbol])
                if df.empty:
                    print(f"{spec.name} ({spec.output_symbol}): no data")
                    continue
                recent = df.sort_values("trade_date").tail(30)
                first = recent["trade_date"].iloc[0]
                last = recent["trade_date"].iloc[-1]
                close = float(recent["close"].iloc[-1]) if "close" in recent.columns else float("nan")
                print(
                    f"{spec.name} ({spec.output_symbol}): rows={len(recent)}, "
                    f"range={pd.Timestamp(first).date()}~{pd.Timestamp(last).date()}, "
                    f"latest_close={close:.2f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
