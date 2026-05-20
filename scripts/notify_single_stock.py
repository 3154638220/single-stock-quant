#!/usr/bin/env python
"""Send single-stock trend signal notification via WeCom webhook.

Reads the latest signal for a stock and pushes a formatted message.
Called by daily_single_stock.sh step 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.risk_metrics import risk_off_multiplier_from_index
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams, compute_dktrend
from src.notify import WecomWebhookHandler, send_trend_signal
from src.settings import load_config
import pandas as pd

from src.signals import Signal
from src.signals.generator import apply_volume_confirmation


def main() -> int:
    parser = argparse.ArgumentParser(description="Send single-stock signal notification via WeCom.")
    parser.add_argument("--symbol", required=True, help="Stock symbol, e.g. 300750")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Send test message instead of real signal")
    args = parser.parse_args()

    cfg = load_config(args.config)
    notify_cfg = cfg.get("notify", {}) or {}
    webhook_url = str(notify_cfg.get("wecom_webhook_url", "")).strip()
    if not webhook_url:
        print("No WeCom webhook URL configured, skipping notification.")
        return 0

    handler = WecomWebhookHandler(webhook_url, mention_all=bool(notify_cfg.get("mention_all", False)))
    symbol = str(args.symbol).strip().zfill(6)

    if args.dry_run:
        msg = f"[测试消息] single-stock-quant 通知通道正常，标的：{symbol}"
        ok = handler.send_markdown(msg)
        print(f"Dry-run notification {'sent' if ok else 'failed'} for {symbol}")
        return 0 if ok else 1

    trend_cfg = cfg.get("trend_signal", {}) or {}
    filt_cfg = cfg.get("signal_filter", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}
    benchmark_symbol = str(risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)

    params = DKTrendParams.from_mapping(trend_cfg)
    read_symbols = [symbol]
    if bool(risk_cfg.get("enable_index_filter", False)) and benchmark_symbol not in read_symbols:
        read_symbols.append(benchmark_symbol)

    name_cache_path = resolve_stock_name_cache_path(cfg)
    names = resolve_stock_names([symbol], name_cache_path)
    display_name = names.get(symbol, symbol)

    with DuckDBManager(config_path=args.config) as db:
        data = db.read_daily_frame(symbols=read_symbols)

    if data.empty:
        print(f"No daily data found for {symbol}")
        return 1

    stock_data = data[data["symbol"].astype(str).str.zfill(6) == symbol].copy()
    if stock_data.empty:
        print(f"No data for {symbol}")
        return 1

    trend = compute_dktrend(stock_data, params)
    trend = apply_volume_confirmation(
        trend,
        enabled=bool(filt_cfg.get("volume_confirm", False)),
        lookback=int(filt_cfg.get("volume_lookback", 20)),
        volume_ratio_min=float(filt_cfg.get("volume_ratio_min", 1.0)),
    )
    trend = trend[trend["dk_color"].isin(["red", "green"])]
    if trend.empty:
        print(f"Not enough trend data for {symbol}")
        return 1

    latest = trend.iloc[-1]
    raw_sig = str(latest["dk_signal"])
    sig = Signal.BUY if raw_sig == "buy" else (Signal.SELL if raw_sig == "sell" else Signal.HOLD)

    # Apply index filter if configured
    if sig == Signal.BUY and bool(risk_cfg.get("enable_index_filter", False)):
        index_data = data[data["symbol"].astype(str).str.zfill(6) == benchmark_symbol]
        if not index_data.empty:
            multiplier, _, _ = risk_off_multiplier_from_index(
                index_data,
                benchmark_symbol=benchmark_symbol,
                asof=pd.Timestamp(latest["trade_date"]),
                lookback_trading_days=int(risk_cfg.get("extreme_lookback_days", 10)),
                drop_threshold=float(risk_cfg.get("extreme_drop_threshold", 0.05)),
                risk_off_factor=float(risk_cfg.get("risk_off_factor", 0.0)),
            )
            if multiplier <= 0.0:
                print(f"Index filter blocked BUY signal for {symbol}")
                return 0

    if sig in (Signal.BUY, Signal.SELL):
        ok = send_trend_signal(
            handler,
            symbol,
            display_name,
            sig,
            float(latest["close"]),
            int(latest["dk_run_len"]),
            str(pd.Timestamp(latest["trade_date"]).date()),
        )
        print(f"Notification {'sent' if ok else 'failed'} for {symbol}: {sig.value}")
        return 0 if ok else 1
    else:
        print(f"No actionable signal for {symbol} (current: HOLD), skipping notification.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
