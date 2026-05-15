#!/usr/bin/env python
"""S1: Trade-level forensics — MAE/MFE, Exit Efficiency, DK dynamics per trade.

Computes for every trade across the watchlist:
  - MFE (Maximum Favourable Excursion): peak return during holding period
  - MAE (Maximum Adverse Excursion): worst return during holding period
  - Exit Efficiency: actual exit return / MFE
  - Time to MFE: days from entry to peak close
  - MFE/MAE Ratio
  - DK value peak-to-exit lag: days from dk_value peak to actual exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.config import build_bt_kwargs
from src.backtest.single_stock import run_single_stock_backtest
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams, compute_dktrend
from src.settings import load_config


def _read_watchlist(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            symbols.append(s.zfill(6))
    return symbols


def _trade_mfe_mae(trade: pd.Series, ohlcv: pd.DataFrame) -> dict:
    """Compute MFE/MAE/Exit Efficiency from OHLCV for one trade."""
    buy_date = pd.Timestamp(trade["buy_date"])
    sell_date = pd.Timestamp(trade["sell_date"])
    entry_price = float(trade["buy_price"])
    exit_return = float(trade["return"])

    mask = (pd.to_datetime(ohlcv["trade_date"]).dt.normalize() >= buy_date) & \
           (pd.to_datetime(ohlcv["trade_date"]).dt.normalize() <= sell_date)
    window = ohlcv[mask].copy()
    if window.empty or entry_price <= 0:
        return {
            "mfe": float("nan"), "mae": float("nan"),
            "exit_efficiency": float("nan"), "time_to_mfe": float("nan"),
            "mfe_mae_ratio": float("nan"),
        }

    closes = pd.to_numeric(window["close"], errors="coerce").values
    pnl = closes / entry_price - 1.0
    mfe = float(np.max(pnl)) if len(pnl) > 0 else float("nan")
    mae = float(np.min(pnl)) if len(pnl) > 0 else float("nan")
    exit_eff = exit_return / mfe if (mfe and mfe > 0 and np.isfinite(mfe)) else float("nan")
    time_to_mfe = int(np.argmax(pnl)) + 1 if len(pnl) > 0 else float("nan")
    mfe_mae = abs(mfe / mae) if (mae and mae < 0 and np.isfinite(mae)) else float("nan")

    return {
        "mfe": mfe, "mae": mae,
        "exit_efficiency": exit_eff,
        "time_to_mfe": time_to_mfe,
        "mfe_mae_ratio": mfe_mae,
    }


def _dk_dynamics(trade: pd.Series, ohlcv: pd.DataFrame, params: DKTrendParams) -> dict:
    """Compute DK value dynamics during the holding period."""
    buy_date = pd.Timestamp(trade["buy_date"])
    sell_date = pd.Timestamp(trade["sell_date"])

    mask = (pd.to_datetime(ohlcv["trade_date"]).dt.normalize() >= buy_date) & \
           (pd.to_datetime(ohlcv["trade_date"]).dt.normalize() <= sell_date)
    window = ohlcv[mask].copy()
    if window.empty:
        return {"dk_peak_to_exit_days": float("nan"), "dk_value_at_exit": float("nan")}

    trend = compute_dktrend(window, params)
    dk_vals = pd.to_numeric(trend["dk_value"], errors="coerce").values
    if len(dk_vals) == 0:
        return {"dk_peak_to_exit_days": float("nan"), "dk_value_at_exit": float("nan")}

    dk_peak_idx = int(np.argmax(dk_vals))
    dk_peak_to_exit = len(dk_vals) - 1 - dk_peak_idx
    dk_at_exit = float(dk_vals[-1])

    return {
        "dk_peak_to_exit_days": dk_peak_to_exit,
        "dk_value_at_exit": dk_at_exit,
        "dk_peak_value": float(dk_vals[dk_peak_idx]),
        "dk_value_at_entry": float(dk_vals[0]) if len(dk_vals) > 0 else float("nan"),
    }


def _format_pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x*100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="S1: Trade-level forensics analysis")
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-08")
    parser.add_argument("--export", required=True, help="Output directory for CSV files")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = _read_watchlist(Path(args.watchlist).expanduser())
    name_cache_path = Path(args.stock_name_cache).expanduser() if args.stock_name_cache else resolve_stock_name_cache_path(cfg)
    names = resolve_stock_names(symbols, name_cache_path)

    risk_cfg = cfg.get("risk", {}) or {}
    benchmark_symbol = str(risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)
    symbols_to_read = list(symbols)
    if benchmark_symbol not in symbols_to_read:
        symbols_to_read.append(benchmark_symbol)

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        data = db.read_daily_frame(symbols=symbols_to_read, start=args.start, end=args.end)

    index_df = data[data["symbol"].astype(str) == benchmark_symbol].copy() if benchmark_symbol in set(data.get("symbol", [])) else None
    base_kwargs = build_bt_kwargs(cfg, index_ohlcv=index_df)
    base_kwargs["consensus_n_agree"] = None

    trend_cfg = cfg.get("trend_signal", {}) or {}
    params = DKTrendParams.from_mapping(dict(trend_cfg))

    all_trades = []
    symbol_stats = []

    for symbol in symbols:
        df = data[data["symbol"].astype(str) == symbol].copy()
        if df.empty:
            print(f"  {symbol}: no data, skipping")
            continue

        kwargs = dict(base_kwargs)
        kwargs["stock_name"] = names.get(symbol, symbol)
        res = run_single_stock_backtest(symbol, df, params, **kwargs)

        if res.trade_log.empty:
            symbol_stats.append({
                "symbol": symbol,
                "stock_name": names.get(symbol, symbol),
                "n_trades": 0,
                "sharpe": float("nan"),
                "mdd": float("nan"),
                "median_exit_efficiency": float("nan"),
                "median_mfe_mae": float("nan"),
            })
            print(f"  {symbol} ({names.get(symbol, symbol)}): 0 trades")
            continue

        # Enrich each trade with MFE/MAE and DK dynamics
        for _, trade in res.trade_log.iterrows():
            mfe_mae = _trade_mfe_mae(trade, df)
            dk_dyn = _dk_dynamics(trade, df, params)
            all_trades.append({
                "symbol": symbol,
                "stock_name": names.get(symbol, symbol),
                "buy_date": trade["buy_date"],
                "sell_date": trade["sell_date"],
                "buy_price": trade["buy_price"],
                "sell_price": trade["sell_price"],
                "hold_days": trade["hold_days"],
                "return": trade["return"],
                "exit_reason": trade.get("exit_reason", "signal"),
                "entry_regime": trade.get("entry_market_regime", "unknown"),
                "mfe": mfe_mae["mfe"],
                "mae": mfe_mae["mae"],
                "exit_efficiency": mfe_mae["exit_efficiency"],
                "time_to_mfe": mfe_mae["time_to_mfe"],
                "mfe_mae_ratio": mfe_mae["mfe_mae_ratio"],
                "dk_peak_to_exit_days": dk_dyn["dk_peak_to_exit_days"],
                "dk_value_at_entry": dk_dyn["dk_value_at_entry"],
                "dk_peak_value": dk_dyn["dk_peak_value"],
                "dk_value_at_exit": dk_dyn["dk_value_at_exit"],
            })

        med_eff = float(np.nanmedian([t["exit_efficiency"] for t in all_trades[-len(res.trade_log):] if np.isfinite(t["exit_efficiency"])]))
        med_mfe_mae = float(np.nanmedian([t["mfe_mae_ratio"] for t in all_trades[-len(res.trade_log):] if np.isfinite(t["mfe_mae_ratio"])]))
        symbol_stats.append({
            "symbol": symbol,
            "stock_name": names.get(symbol, symbol),
            "n_trades": res.n_trades,
            "sharpe": res.sharpe_ratio,
            "mdd": res.max_drawdown,
            "median_exit_efficiency": med_eff,
            "median_mfe_mae": med_mfe_mae,
        })
        print(f"  {symbol} ({names.get(symbol, symbol)}): {res.n_trades} trades, "
              f"median exit_eff={_format_pct(med_eff)}, median mfe/mae={med_mfe_mae:.1f}")

    # Export
    out_dir = Path(args.export).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame(all_trades)
    trades_path = out_dir / "trade_mfe_mae.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"\nTrade-level forensics written to {trades_path} ({len(trades_df)} trades)")

    stats_df = pd.DataFrame(symbol_stats)
    stats_path = out_dir / "symbol_forensics_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Symbol summary written to {stats_path}")

    # Print aggregate statistics
    print("\n" + "=" * 64)
    print("AGGREGATE FORENSICS SUMMARY")
    print("=" * 64)
    valid = trades_df[trades_df["mfe"].notna() & (trades_df["mfe"] > 0)]
    if not valid.empty:
        print(f"Total trades:            {len(trades_df)}")
        print(f"Median MFE:              {_format_pct(valid['mfe'].median())}")
        print(f"Median MAE:              {_format_pct(valid['mae'].median())}")
        print(f"Median Exit Efficiency:  {valid['exit_efficiency'].median():.2f}")
        print(f"Mean Exit Efficiency:    {valid['exit_efficiency'].mean():.2f}")
        print(f"Median MFE/MAE Ratio:    {valid['mfe_mae_ratio'].median():.1f}")
        print(f"Median Time to MFE:      {valid['time_to_mfe'].median():.0f} days")
        print(f"Median DK peak→exit lag: {valid['dk_peak_to_exit_days'].median():.0f} days")

        low_eff = (valid["exit_efficiency"] < 0.5).sum()
        print(f"\nTrades with Exit Eff < 0.5: {low_eff}/{len(valid)} ({low_eff/len(valid)*100:.0f}%)")

        # DK lag breakdown
        dk_lag = valid["dk_peak_to_exit_days"].dropna()
        if not dk_lag.empty:
            print(f"\nDK peak→exit lag distribution:")
            for bin_label, (lo, hi) in [(" 0-5d", (0, 5)), (" 6-10d", (6, 10)),
                                          ("11-15d", (11, 15)), ("16-20d", (16, 20)),
                                          ("21-30d", (21, 30)), ("31d+", (31, 999))]:
                count = ((dk_lag >= lo) & (dk_lag <= hi)).sum()
                print(f"  {bin_label}: {count:>4} ({count/len(dk_lag)*100:.0f}%)")

        # Exit reason breakdown
        print(f"\nExit Efficiency by exit reason:")
        for reason in valid["exit_reason"].unique():
            sub = valid[valid["exit_reason"] == reason]
            print(f"  {reason:<16}: n={len(sub):>4}, median eff={sub['exit_efficiency'].median():.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
