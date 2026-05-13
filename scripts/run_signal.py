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

from src.backtest.risk_metrics import risk_off_multiplier_from_index
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.notify import WecomWebhookHandler, send_trend_signal
from src.settings import load_config
from src.signals import Signal
from src.signals.consensus import compute_consensus_trend
from src.signals.generator import apply_volume_confirmation


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


def _trend_label(color: str) -> str:
    return "多头" if color == "red" else "空头"


def _signal_label(signal: Signal) -> str:
    if signal == Signal.BUY:
        return "BUY"
    if signal == Signal.SELL:
        return "SELL"
    return "HOLD"


def _advice(signal: Signal, color: str, *, risk_blocked: bool = False) -> str:
    if risk_blocked:
        return "大盘风控触发：原 BUY 已屏蔽，空仓观望。"
    if signal == Signal.BUY:
        return "买入信号：下个交易日开盘关注执行；若一字涨停则按回测规则顺延。"
    if signal == Signal.SELL:
        return "卖出信号：下个交易日开盘关注离场。"
    if color == "red":
        return "持仓观望（多头趋势持续中）。"
    return "空仓观望（空头趋势持续中）。"


def _print_history(trend: pd.DataFrame, history: int) -> None:
    recent = trend.tail(history)
    changes = recent[recent["dk_signal"].isin(["buy", "sell"])]
    print(f"\n近期信号记录（最近 {history} 条日线内）:")
    if changes.empty:
        print("  无变色信号")
        return
    print("  日期          信号    收盘      趋势   连续天")
    for _, row in changes.iterrows():
        sig = _signal_label(_row_signal(str(row.get("dk_signal") or "")))
        color = _trend_label(str(row["dk_color"]))
        print(
            f"  {pd.Timestamp(row['trade_date']).date()}  {sig:<5} "
            f"{float(row['close']):>8.2f}  {color:<4} {int(row['dk_run_len']):>4}"
        )


def _print_latest(
    *,
    symbol: str,
    display_name: str,
    mode: str,
    latest: pd.Series,
    signal: Signal,
    risk_blocked: bool = False,
) -> None:
    trade_date = pd.Timestamp(latest["trade_date"]).date()
    color = str(latest["dk_color"])
    trend = _trend_label(color)
    run_len = int(latest["dk_run_len"])
    close = float(latest["close"])
    print(f"{display_name} ({symbol}) | 最新交易日：{trade_date} | 指标：{mode}")
    print("-" * 64)
    print(f"当前多空趋势：{trend} | 已连续 {run_len} 天 | 最新收盘：{close:.2f} | 最新信号：{_signal_label(signal)}")
    print(f"操作建议：{_advice(signal, color, risk_blocked=risk_blocked)}")


def _risk_blocks_buy(index_df: pd.DataFrame | None, risk_cfg: dict, asof: pd.Timestamp) -> bool:
    if index_df is None or index_df.empty or not bool(risk_cfg.get("enable_index_filter", False)):
        return False
    benchmark_symbol = str(risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)
    multiplier, _, _ = risk_off_multiplier_from_index(
        index_df,
        benchmark_symbol=benchmark_symbol,
        asof=asof,
        lookback_trading_days=int(risk_cfg.get("extreme_lookback_days", 10)),
        drop_threshold=float(risk_cfg.get("extreme_drop_threshold", 0.05)),
        risk_off_factor=float(risk_cfg.get("risk_off_factor", 0.0)),
    )
    return multiplier <= 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Show DK trend signal.")
    parser.add_argument("--symbol")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--watchlist", nargs="+")
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--mode", choices=[m.value for m in TrendMode])
    parser.add_argument("--consensus", action="store_true", help="Use multi-mode consensus instead of one DK mode")
    parser.add_argument("--filter", choices=["buy", "sell", "hold"])
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path, e.g. /home/x12dpg/hjx/lh/data/market.duckdb")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configured_mode = str((cfg.get("trend_signal", {}) or {}).get("mode", "macd_cross"))
    params = _params(cfg, args.mode or ("macd_cross" if configured_mode == "consensus" else None))
    filt_cfg = cfg.get("signal_filter", {}) or {}
    trend_cfg = cfg.get("trend_signal", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}
    notify_cfg = cfg.get("notify", {}) or {}
    webhook_url = str(notify_cfg.get("wecom_webhook_url", "")).strip()
    handler = WecomWebhookHandler(webhook_url, mention_all=bool(notify_cfg.get("mention_all", False))) if webhook_url else None
    symbols = _symbols(args)
    benchmark_symbol = str(risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)
    read_symbols = list(symbols)
    if bool(risk_cfg.get("enable_index_filter", False)) and benchmark_symbol not in read_symbols:
        read_symbols.append(benchmark_symbol)
    name_cache_path = Path(args.stock_name_cache).expanduser() if args.stock_name_cache else resolve_stock_name_cache_path(cfg)
    names = resolve_stock_names(symbols, name_cache_path)

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        data = db.read_daily_frame(symbols=read_symbols)

    if data.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    index_df = (
        data[data["symbol"].astype(str).str.zfill(6) == benchmark_symbol].copy()
        if bool(risk_cfg.get("enable_index_filter", False))
        else None
    )
    for symbol, ohlcv in data.groupby("symbol", sort=True):
        symbol = str(symbol).zfill(6)
        if symbol not in set(symbols):
            continue
        display_name = names.get(str(symbol), str(symbol))
        use_consensus = args.consensus or str(trend_cfg.get("mode", "macd_cross")) == "consensus"
        if use_consensus:
            trend = compute_consensus_trend(
                ohlcv,
                base_params=params,
                n_agree=int(trend_cfg.get("consensus_n_agree", 2)),
                volume_confirm=bool(filt_cfg.get("volume_confirm", False)),
                volume_lookback=int(filt_cfg.get("volume_lookback", 20)),
                volume_ratio_min=float(filt_cfg.get("volume_ratio_min", 1.0)),
            )
            mode_label = f"consensus_{int(trend_cfg.get('consensus_n_agree', 2))}"
        else:
            trend = compute_dktrend(ohlcv, params)
            trend = apply_volume_confirmation(
                trend,
                enabled=bool(filt_cfg.get("volume_confirm", False)),
                lookback=int(filt_cfg.get("volume_lookback", 20)),
                volume_ratio_min=float(filt_cfg.get("volume_ratio_min", 1.0)),
            )
            mode_label = str(params.mode.value)
        trend = trend[trend["dk_color"].isin(["red", "green"])]
        if trend.empty:
            print(f"{display_name} ({symbol}): not enough data")
            continue
        latest = trend.iloc[-1]
        sig = _row_signal(str(latest["dk_signal"]))
        risk_blocked = sig == Signal.BUY and _risk_blocks_buy(index_df, risk_cfg, pd.Timestamp(latest["trade_date"]))
        if risk_blocked:
            sig = Signal.HOLD
        if args.filter and sig.value != args.filter:
            continue
        _print_latest(
            symbol=str(symbol),
            display_name=display_name,
            mode=mode_label,
            latest=latest,
            signal=sig,
            risk_blocked=risk_blocked,
        )
        if args.history:
            _print_history(trend, args.history)
        if handler and sig in (Signal.BUY, Signal.SELL):
            send_trend_signal(
                handler,
                str(symbol),
                display_name,
                sig,
                float(latest["close"]),
                int(latest["dk_run_len"]),
                str(pd.Timestamp(latest["trade_date"]).date()),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
