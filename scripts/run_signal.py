#!/usr/bin/env python
"""Print latest DK trend signal from local DuckDB daily data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.notify import WecomWebhookHandler, send_trend_signal
from src.settings import load_config
from src.signals import Signal


def _params(cfg: dict, mode: str | None) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    if mode:
        raw["mode"] = mode
    return DKTrendParams.from_mapping(raw)


def _symbols(args: argparse.Namespace) -> list[str]:
    values = []
    if args.symbol:
        values.append(args.symbol)
    if args.symbols:
        values.extend(args.symbols)
    if args.watchlist:
        values.extend(args.watchlist)
    if not values:
        raise SystemExit("provide --symbol, --symbols or --watchlist")
    return [str(v).strip().zfill(6) for v in values]


def _row_signal(raw: str) -> Signal:
    if raw == "buy":
        return Signal.BUY
    if raw == "sell":
        return Signal.SELL
    return Signal.HOLD


def _print_history(trend: pd.DataFrame, history: int) -> None:
    recent = trend.tail(history)
    print("date        signal close      trend run")
    for _, row in recent.iterrows():
        sig = str(row.get("dk_signal") or "-").upper()
        color = "LONG" if row["dk_color"] == "red" else "SHORT"
        print(
            f"{pd.Timestamp(row['trade_date']).date()} {sig:<6} "
            f"{float(row['close']):>8.2f} {color:<5} {int(row['dk_run_len']):>3}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Show DK trend signal.")
    parser.add_argument("--symbol")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--watchlist", nargs="+")
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--mode", choices=[m.value for m in TrendMode])
    parser.add_argument("--filter", choices=["buy", "sell", "hold"])
    parser.add_argument("--config", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    params = _params(cfg, args.mode)
    notify_cfg = cfg.get("notify", {}) or {}
    webhook_url = str(notify_cfg.get("wecom_webhook_url", "")).strip()
    handler = WecomWebhookHandler(webhook_url, mention_all=bool(notify_cfg.get("mention_all", False))) if webhook_url else None

    with DuckDBManager(config_path=args.config) as db:
        data = db.read_daily_frame(symbols=_symbols(args))

    if data.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    for symbol, ohlcv in data.groupby("symbol", sort=True):
        trend = compute_dktrend(ohlcv, params)
        trend = trend[trend["dk_color"].isin(["red", "green"])]
        if trend.empty:
            print(f"{symbol}: not enough data")
            continue
        latest = trend.iloc[-1]
        sig = _row_signal(str(latest["dk_signal"]))
        if args.filter and sig.value != args.filter:
            continue
        color = "LONG" if latest["dk_color"] == "red" else "SHORT"
        print(f"{symbol} | latest={pd.Timestamp(latest['trade_date']).date()} | {color} {int(latest['dk_run_len'])}d | close={float(latest['close']):.2f} | signal={sig.value}")
        if args.history:
            _print_history(trend, args.history)
        if handler and sig in (Signal.BUY, Signal.SELL):
            send_trend_signal(
                handler,
                str(symbol),
                str(symbol),
                sig,
                float(latest["close"]),
                int(latest["dk_run_len"]),
                str(pd.Timestamp(latest["trade_date"]).date()),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
