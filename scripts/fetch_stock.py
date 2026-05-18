#!/usr/bin/env python
"""Fetch one or more A-share daily series into DuckDB."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.akshare_client import fetch_etf_daily, fill_derived_daily_fields
from src.data_fetcher.db_manager import DuckDBManager
from src.settings import load_config


@dataclass(frozen=True)
class RecentQualitySummary:
    symbol: str
    rows: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    latest_close: float
    max_calendar_gap_days: int
    null_ohlcv: int
    invalid_ohlc: int

    def format_line(self) -> str:
        if self.rows == 0 or self.first_date is None or self.last_date is None:
            return f"{self.symbol}: no rows"
        return (
            f"{self.symbol}: rows={self.rows}, "
            f"range={self.first_date.date()}~{self.last_date.date()}, "
            f"latest_close={self.latest_close:.2f}, max_calendar_gap={self.max_calendar_gap_days}d, "
            f"null_ohlcv={self.null_ohlcv}, invalid_ohlc={self.invalid_ohlc}"
        )

    def violations(
        self,
        *,
        min_rows: int,
        max_gap_days: int,
        fail_on_nulls: bool,
        fail_on_invalid_ohlc: bool,
    ) -> list[str]:
        out: list[str] = []
        if self.rows < int(min_rows):
            out.append(f"rows {self.rows} < {int(min_rows)}")
        if self.max_calendar_gap_days > int(max_gap_days):
            out.append(f"max_calendar_gap {self.max_calendar_gap_days}d > {int(max_gap_days)}d")
        if fail_on_nulls and self.null_ohlcv > 0:
            out.append(f"null_ohlcv {self.null_ohlcv} > 0")
        if fail_on_invalid_ohlc and self.invalid_ohlc > 0:
            out.append(f"invalid_ohlc {self.invalid_ohlc} > 0")
        return out


def _symbols(args: argparse.Namespace) -> list[str]:
    items: list[str] = []
    if args.symbol:
        items.append(args.symbol)
    if args.symbols:
        items.extend(args.symbols)
    if args.watchlist:
        path = Path(args.watchlist).expanduser()
        for line in path.read_text(encoding="utf-8").splitlines():
            sym = line.split("#", 1)[0].strip()
            if sym:
                items.append(sym)
    out = []
    for item in items:
        code = str(item).strip().zfill(6)
        if len(code) != 6 or not code.isdigit():
            raise SystemExit(f"invalid A-share symbol: {item}")
        out.append(code)
    if not out:
        raise SystemExit("provide --symbol or --symbols")
    return sorted(set(out))


def _write_frame(db: DuckDBManager, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    payload = fill_derived_daily_fields(
        df.sort_values(["symbol", "trade_date"])
        .drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        .reset_index(drop=True)
    )
    db.connection.register("df_fetch_stock_manual", payload)
    try:
        db.connection.execute(
            f"""
            INSERT OR REPLACE INTO {table}
            SELECT * FROM df_fetch_stock_manual
            """
        )
    finally:
        db.connection.unregister("df_fetch_stock_manual")
    return int(len(payload))


def _fetch_etfs_to_table(
    db: DuckDBManager,
    symbols: list[str],
    *,
    table: str,
    start: str,
    end: str | None,
    config: dict,
) -> dict[str, tuple[int, bool]]:
    counts: dict[str, tuple[int, bool]] = {}
    end_s = end or pd.Timestamp.today().strftime("%Y%m%d")
    for sym in symbols:
        try:
            existing_last = db.last_trade_date(sym)
            start_s = (pd.Timestamp(existing_last) + pd.Timedelta(days=1)).strftime("%Y%m%d") if existing_last else start
            if pd.Timestamp(start_s) > pd.Timestamp(end_s):
                counts[sym] = (0, False)
                continue
            df = fetch_etf_daily(sym, start_s, end_s, config=config)
            rows = _write_frame(db, table, df)
            counts[sym] = (rows, False)
        except Exception as exc:
            print(f"{sym}: FAILED, {type(exc).__name__}: {exc}")
            counts[sym] = (0, True)
    return counts


def _recent_quality_summary(symbol: str, df: pd.DataFrame, *, window: int = 30) -> RecentQualitySummary:
    if df.empty:
        return RecentQualitySummary(
            symbol=symbol,
            rows=0,
            first_date=None,
            last_date=None,
            latest_close=float("nan"),
            max_calendar_gap_days=0,
            null_ohlcv=0,
            invalid_ohlc=0,
        )

    recent = df.sort_values("trade_date").tail(window).copy()
    recent["trade_date"] = pd.to_datetime(recent["trade_date"])
    ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in recent.columns]
    nulls = int(recent[ohlcv_cols].isna().sum().sum()) if ohlcv_cols else 0
    invalid_ohlc = 0
    if {"open", "high", "low", "close"} <= set(recent.columns):
        invalid = (
            (recent["high"] < recent[["open", "close", "low"]].max(axis=1))
            | (recent["low"] > recent[["open", "close", "high"]].min(axis=1))
            | (recent[["open", "high", "low", "close"]] <= 0).any(axis=1)
        )
        invalid_ohlc = int(invalid.sum())
    gaps = recent["trade_date"].diff().dt.days.dropna()
    max_gap = int(gaps.max()) if not gaps.empty else 0
    close = float(recent["close"].iloc[-1]) if "close" in recent.columns else float("nan")
    return RecentQualitySummary(
        symbol=symbol,
        rows=int(len(recent)),
        first_date=pd.Timestamp(recent["trade_date"].iloc[0]),
        last_date=pd.Timestamp(recent["trade_date"].iloc[-1]),
        latest_close=close,
        max_calendar_gap_days=max_gap,
        null_ohlcv=nulls,
        invalid_ohlc=invalid_ohlc,
    )


def _print_recent_quality(
    db: DuckDBManager,
    symbols: list[str],
    *,
    window: int = 30,
    min_rows: int = 1,
    max_gap_days: int = 20,
    fail_on_quality: bool = False,
    fail_on_nulls: bool = True,
    fail_on_invalid_ohlc: bool = True,
) -> bool:
    print(f"recent quality summary (last {window} rows per symbol):")
    ok = True
    for sym in symbols:
        df = db.read_daily_frame(symbols=[sym])
        summary = _recent_quality_summary(sym, df, window=window)
        violations = summary.violations(
            min_rows=min_rows,
            max_gap_days=max_gap_days,
            fail_on_nulls=fail_on_nulls,
            fail_on_invalid_ohlc=fail_on_invalid_ohlc,
        )
        print(summary.format_line())
        if violations:
            ok = False
            print("  quality violations: " + "; ".join(violations))
    if fail_on_quality and not ok:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch daily OHLCV for selected A-share symbols.")
    parser.add_argument("--symbol", help="Single 6-digit symbol, e.g. 600930")
    parser.add_argument("--symbols", nargs="+", help="Multiple 6-digit symbols")
    parser.add_argument("--watchlist", help="File containing one symbol per line")
    parser.add_argument("--start", default="20150101", help="Default start date for new symbols, YYYYMMDD")
    parser.add_argument("--end", help="End date, YYYYMMDD or YYYY-MM-DD. Default: today")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--db-table", help="DuckDB table to write/read; defaults to configured daily table")
    parser.add_argument("--check-quality", action="store_true", help="Print recent quality summary after fetch")
    parser.add_argument("--quality-window", type=int, default=30, help="Rows per symbol used by --check-quality")
    parser.add_argument("--quality-min-rows", type=int, default=1, help="Minimum recent rows required with --fail-on-quality")
    parser.add_argument("--quality-max-gap-days", type=int, help="Maximum allowed recent calendar gap")
    parser.add_argument("--fail-on-quality", action="store_true", help="Exit non-zero if recent quality thresholds fail")
    parser.add_argument("--quality-allow-nulls", action="store_true", help="Do not fail on recent OHLCV nulls")
    parser.add_argument("--quality-allow-invalid-ohlc", action="store_true", help="Do not fail on invalid recent OHLC rows")
    args = parser.parse_args()

    symbols = _symbols(args)
    cfg = load_config(args.config)
    quality_cfg = cfg.get("quality", {}) or {}
    max_gap_days = (
        int(args.quality_max_gap_days)
        if args.quality_max_gap_days is not None
        else int(quality_cfg.get("max_calendar_gap_days", 20))
    )
    with DuckDBManager(config_path=args.config, table_daily=args.db_table) as db:
        table = args.db_table or cfg.get("database", {}).get("table_daily", "a_share_daily")
        if args.db_table and args.db_table != cfg.get("database", {}).get("table_daily", "a_share_daily"):
            manual_counts = _fetch_etfs_to_table(
                db,
                symbols,
                table=table,
                start=args.start,
                end=args.end,
                config=cfg,
            )
            counts = {
                sym: type("ManualResult", (), {"rows_written": rows, "fetch_failed": failed})()
                for sym, (rows, failed) in manual_counts.items()
            }
        else:
            counts = db.incremental_update_many(symbols, default_start=args.start, end_date=args.end)
        for sym in symbols:
            result = counts.get(sym)
            rows = result.rows_written if result else 0
            failed = result.fetch_failed if result else False
            status = "FAILED" if failed else "OK"
            print(f"{sym}: {status}, rows_written={rows}")
        if args.check_quality:
            ok = _print_recent_quality(
                db,
                symbols,
                window=max(1, int(args.quality_window)),
                min_rows=max(0, int(args.quality_min_rows)),
                max_gap_days=max_gap_days,
                fail_on_quality=bool(args.fail_on_quality),
                fail_on_nulls=not bool(args.quality_allow_nulls),
                fail_on_invalid_ohlc=not bool(args.quality_allow_invalid_ohlc),
            )
            if args.fail_on_quality and not ok:
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
