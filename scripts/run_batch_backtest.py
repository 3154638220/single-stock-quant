#!/usr/bin/env python
"""Run single-stock backtests for a watchlist and export a summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.single_stock import run_single_stock_backtest
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams, TrendMode
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


def _params(cfg: dict, mode: str) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    raw["mode"] = mode
    return DKTrendParams.from_mapping(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backtests for every symbol in a watchlist.")
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--mode", choices=[m.value for m in TrendMode])
    parser.add_argument("--consensus", action="store_true", help="Use multi-mode consensus instead of one DK mode")
    parser.add_argument("--export-summary", required=True)
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bt_cfg = cfg.get("backtest", {}) or {}
    filt_cfg = cfg.get("signal_filter", {}) or {}
    trend_cfg = cfg.get("trend_signal", {}) or {}
    configured_mode = str(trend_cfg.get("mode", "macd_cross"))
    use_consensus = args.consensus or (configured_mode == "consensus" and args.mode is None)
    selected_mode = args.mode or ("macd_cross" if use_consensus else configured_mode)
    symbols = _read_watchlist(Path(args.watchlist).expanduser())
    name_cache_path = Path(args.stock_name_cache).expanduser() if args.stock_name_cache else resolve_stock_name_cache_path(cfg)
    names = resolve_stock_names(symbols, name_cache_path)

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        data = db.read_daily_frame(symbols=symbols, start=args.start, end=args.end)
    if data.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    rows = []
    for symbol in symbols:
        df = data[data["symbol"].astype(str) == symbol].copy()
        if df.empty:
            rows.append({"symbol": symbol, "stock_name": names.get(symbol, symbol), "status": "no_data"})
            continue
        res = run_single_stock_backtest(
            symbol,
            df,
            _params(cfg, selected_mode),
            cost_bps=float(bt_cfg.get("cost_bps", 15.0)),
            initial_capital=float(bt_cfg.get("initial_capital", 100000)),
            stock_name=names.get(symbol, symbol),
            volume_confirm=bool(filt_cfg.get("volume_confirm", False)),
            volume_lookback=int(filt_cfg.get("volume_lookback", 20)),
            volume_ratio_min=float(filt_cfg.get("volume_ratio_min", 1.0)),
            consensus_n_agree=int(trend_cfg.get("consensus_n_agree", 2)) if use_consensus else None,
            stop_loss_pct=float(bt_cfg.get("stop_loss_pct", 0.0)),
            trailing_stop_pct=float(bt_cfg.get("trailing_stop_pct", 0.0)),
        )
        rows.append(
            {
                "symbol": res.symbol,
                "stock_name": res.stock_name,
                "status": "ok",
                "period": res.period,
                "total_return": res.total_return,
                "annualized_return": res.annualized_return,
                "buy_hold_return": res.buy_hold_return,
                "sharpe_ratio": res.sharpe_ratio,
                "max_drawdown": res.max_drawdown,
                "calmar_ratio": res.calmar_ratio,
                "n_trades": res.n_trades,
                "win_rate": res.win_rate,
                "stop_loss_exits": res.stop_loss_exits,
                "trailing_stop_exits": res.trailing_stop_exits,
            }
        )

    out = Path(args.export_summary).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"已完成 {ok}/{len(rows)} 个标的，汇总写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
