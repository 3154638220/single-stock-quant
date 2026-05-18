#!/usr/bin/env python
"""Run the 2026-05-18 return-improvement experiment plan."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.config import build_bt_kwargs
from src.backtest.performance_panel import compute_performance_panel
from src.backtest.single_stock import run_single_stock_backtest
from src.backtest.wfo import (
    _VALID_BT_KWARGS,
    _oos_trend_with_warmup,
    bootstrap_sharpe_ci,
    json_safe,
    trade_contribution_metrics,
)
from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams
from src.settings import load_config, project_root


def _read_symbol_frame(symbol: str, start: str | None, end: str | None, config: str | None, duckdb_path: str | None) -> pd.DataFrame:
    with DuckDBManager(config_path=config, duckdb_path=duckdb_path) as db:
        df = db.read_daily_frame(symbols=[symbol], start=start, end=end)
    if df.empty:
        raise SystemExit(f"{symbol}: no daily data found; run scripts/fetch_stock.py first")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    return df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def _selected_symbol_config(symbol: str) -> dict[str, Any]:
    payload = yaml.safe_load((project_root() / "configs/research/selected_single_stock_best.yaml").read_text(encoding="utf-8"))
    item = payload["symbols"][symbol]
    cfg = {
        "paths": {"duckdb_path": "data/market.duckdb", "output_dir": "data/output"},
        "trend_signal": dict(item["trend_signal"]),
        "backtest": dict(payload["shared_backtest"]),
        "signal_filter": dict(payload["shared_signal_filter"]),
    }
    cfg["trend_signal"]["mode"] = item["trend_signal"].get("mode", "macd_cross")
    return cfg


def _base_config_for_symbol(symbol: str, explicit_config: str | None) -> dict[str, Any]:
    if explicit_config:
        return load_config(explicit_config)
    if symbol == "000783":
        return load_config("configs/research/000783_best_return.yaml")
    return _selected_symbol_config(symbol)


def _mfe_utilization_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty or "mfe" not in trades or "return" not in trades:
        return {"mfe_utilization_p25": float("nan"), "mfe_utilization_p50": float("nan"), "mfe_utilization_p75": float("nan")}
    mfe = pd.to_numeric(trades["mfe"], errors="coerce")
    ret = pd.to_numeric(trades["return"], errors="coerce")
    util = (ret / mfe).where(mfe > 0).replace([np.inf, -np.inf], np.nan).dropna()
    if util.empty:
        return {"mfe_utilization_p25": float("nan"), "mfe_utilization_p50": float("nan"), "mfe_utilization_p75": float("nan")}
    return {
        "mfe_utilization_p25": float(util.quantile(0.25)),
        "mfe_utilization_p50": float(util.quantile(0.50)),
        "mfe_utilization_p75": float(util.quantile(0.75)),
    }


def _exit_reason_share(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty or "exit_reason" not in trades:
        return {}
    counts = trades["exit_reason"].astype(str).value_counts(normalize=True)
    return {f"exit_share_{k}": float(v) for k, v in counts.items()}


def _fixed_param_wfo(
    *,
    symbol: str,
    ohlcv: pd.DataFrame,
    params: DKTrendParams,
    bt_kwargs: dict[str, Any],
    experiment_id: str,
    train_days: int,
    oos_days: int,
    step_days: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cost_bps = float(bt_kwargs.get("cost_bps", 15.0))
    initial_capital = float(bt_kwargs.get("initial_capital", 100000.0))
    run_kwargs = {k: v for k, v in bt_kwargs.items() if k in _VALID_BT_KWARGS}

    returns: list[pd.Series] = []
    trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    fold = 0
    start = 0
    while start + train_days + oos_days <= len(ohlcv):
        train_df = ohlcv.iloc[start : start + train_days].copy()
        oos_df = ohlcv.iloc[start + train_days : start + train_days + oos_days].copy()
        trend = _oos_trend_with_warmup(train_df, oos_df, params, run_kwargs)
        res = run_single_stock_backtest(
            symbol,
            oos_df,
            params,
            cost_bps=cost_bps,
            initial_capital=initial_capital,
            trend_override=trend,
            **run_kwargs,
        )
        returns.append(res.daily_returns)
        if not res.trade_log.empty:
            t = res.trade_log.copy()
            t.insert(0, "fold", fold)
            t.insert(0, "experiment_id", experiment_id)
            trades.append(t)
        fold_rows.append(
            {
                "fold": fold,
                "oos_start": pd.Timestamp(oos_df["trade_date"].iloc[0]).date().isoformat(),
                "oos_end": pd.Timestamp(oos_df["trade_date"].iloc[-1]).date().isoformat(),
                "total_return": float(res.total_return),
                "annualized_return": float(res.annualized_return),
                "max_drawdown": float(res.max_drawdown),
                "calmar_ratio": float(res.calmar_ratio),
                "sharpe_ratio": float(res.sharpe_ratio),
                "n_trades": int(res.n_trades),
            }
        )
        fold += 1
        start += step_days

    stitched = pd.concat(returns).sort_index() if returns else pd.Series(dtype=float)
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    panel = compute_performance_panel(stitched.to_numpy(dtype=np.float64)) if not stitched.empty else None
    total_return = float(panel.total_return) if panel is not None else float("nan")
    bootstrap = bootstrap_sharpe_ci(stitched.to_numpy(dtype=np.float64)) if not stitched.empty else {}
    signal_params = asdict(params)
    signal_params["mode"] = params.mode.value if hasattr(params.mode, "value") else str(params.mode)
    summary = {
        "experiment_id": experiment_id,
        "symbol": symbol,
        "run_date": date.today().isoformat(),
        "signal_params": signal_params,
        "exit_params": {
            k: bt_kwargs.get(k)
            for k in ["profit_lock_trigger", "profit_lock_trailing", "intrapos_dd_limit", "atr_trailing_mult", "dk_fade_exit_n", "time_stop_days"]
        },
        "position_params": {
            k: bt_kwargs.get(k)
            for k in ["volatility_target_ann", "volatility_lookback", "volatility_high_vol_multiple", "volatility_high_vol_scale", "drawdown_throttle_enabled"]
        },
        "entry_params": {
            k: bt_kwargs.get(k)
            for k in ["require_price_breakout", "breakout_lookback", "volume_ratio_min"]
        },
        "wfo_config": {"train_days": train_days, "oos_days": oos_days, "step_days": step_days, "n_folds": len(fold_rows)},
        "oos_metrics": {
            "total_return": total_return,
            "annualized_return": float(panel.annualized_return) if panel is not None else float("nan"),
            "max_drawdown": float(panel.max_drawdown) if panel is not None else float("nan"),
            "calmar_ratio": float(panel.calmar_ratio) if panel is not None else float("nan"),
            "sharpe_ratio": float(panel.sharpe_ratio) if panel is not None else float("nan"),
            "n_trades": int(len(all_trades)),
            "win_rate": float((pd.to_numeric(all_trades.get("return", pd.Series(dtype=float)), errors="coerce") > 0).mean())
            if not all_trades.empty
            else float("nan"),
        },
        "trade_quality": {
            **trade_contribution_metrics(all_trades, total_return),
            **_mfe_utilization_stats(all_trades),
            **_exit_reason_share(all_trades),
            "avg_hold_days": float(pd.to_numeric(all_trades.get("hold_days", pd.Series(dtype=float)), errors="coerce").mean())
            if not all_trades.empty
            else float("nan"),
        },
        "bootstrap": {
            "sharpe_ci_lower": bootstrap.get("ci_lower", float("nan")),
            "sharpe_ci_upper": bootstrap.get("ci_upper", float("nan")),
            "positive_fraction": bootstrap.get("positive_fraction", float("nan")),
        },
        "folds": fold_rows,
    }
    m = summary["oos_metrics"]
    q = summary["trade_quality"]
    b = summary["bootstrap"]
    summary["decision"] = (
        "promote"
        if m["annualized_return"] >= 0.14
        and m["calmar_ratio"] >= 1.30
        and m["max_drawdown"] <= 0.15
        and m["n_trades"] >= 10
        and (not np.isfinite(q["largest_trade_contribution"]) or q["largest_trade_contribution"] <= 0.50)
        and b["positive_fraction"] >= 0.55
        else "retry"
        if m["total_return"] > 0 and m["max_drawdown"] <= 0.20
        else "reject"
    )
    return summary, all_trades


def _exit_variants() -> list[dict[str, Any]]:
    base = [
        ("E01", 0.08, 0.04, 0.05, 0.0),
        ("E02", 0.12, 0.05, 0.05, 0.0),
        ("E03", 0.15, 0.07, 0.05, 0.0),
        ("E04", 0.08, 0.04, 0.04, 0.0),
        ("E05", 0.08, 0.04, 0.06, 0.0),
        ("E06", 0.10, 0.05, 0.0, 2.5),
        ("E07", 0.10, 0.05, 0.0, 3.0),
        ("E08", 0.12, 0.06, 0.0, 2.0),
    ]
    rows = []
    for suffix, trigger, trailing, dd_limit, atr_mult in base:
        rows.append(
            {
                "id": suffix,
                "bt": {
                    "profit_lock_trigger": trigger,
                    "profit_lock_trailing": trailing,
                    "intrapos_dd_limit": dd_limit,
                    "atr_trailing_mult": atr_mult,
                },
            }
        )
        rows.append(
            {
                "id": suffix.replace("E", "E") + "_no_time_stop",
                "bt": {
                    "profit_lock_trigger": trigger,
                    "profit_lock_trailing": trailing,
                    "intrapos_dd_limit": dd_limit,
                    "atr_trailing_mult": atr_mult,
                    "time_stop_days": 0,
                },
            }
        )
    return rows


def _run_group(
    *,
    symbols: list[str],
    group: str,
    variants: list[dict[str, Any]],
    out_dir: Path,
    start: str | None,
    end: str | None,
    config: str | None,
    duckdb_path: str | None,
    train_days: int,
    oos_days: int,
    step_days: int,
) -> list[dict[str, Any]]:
    group_dir = out_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    for symbol in symbols:
        cfg = _base_config_for_symbol(symbol, config)
        params = DKTrendParams.from_mapping(dict(cfg.get("trend_signal", {}) or {}))
        bt_base = build_bt_kwargs(cfg)
        bt_base["stock_name"] = symbol
        ohlcv = _read_symbol_frame(symbol, start, end, config, duckdb_path)
        for variant in variants:
            bt = {**bt_base, **variant.get("bt", {})}
            params_data = asdict(params)
            params_data.update(variant.get("trend", {}))
            exp_id = f"{group}_{symbol}_{variant['id']}"
            summary, trades = _fixed_param_wfo(
                symbol=symbol,
                ohlcv=ohlcv,
                params=DKTrendParams.from_mapping(params_data),
                bt_kwargs=bt,
                experiment_id=exp_id,
                train_days=train_days,
                oos_days=oos_days,
                step_days=step_days,
            )
            summary["notes"] = variant.get("notes", "")
            (group_dir / f"{exp_id}.json").write_text(
                json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            flat = {
                "experiment_id": exp_id,
                "symbol": symbol,
                "decision": summary["decision"],
                **summary["oos_metrics"],
                **summary["trade_quality"],
                **summary["bootstrap"],
                **variant.get("bt", {}),
                **variant.get("trend", {}),
            }
            rows.append(flat)
            if not trades.empty:
                trade_frames.append(trades)
    pd.DataFrame(rows).sort_values(
        ["decision", "calmar_ratio", "annualized_return"],
        ascending=[True, False, False],
        na_position="last",
    ).to_csv(group_dir / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(group_dir / "trades.csv", index=False)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute docs/plan-05-18.md experiments.")
    parser.add_argument("--symbols", nargs="+", default=["000783", "300750"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end")
    parser.add_argument("--config", help="Use one explicit config for all symbols instead of symbol-specific research configs")
    parser.add_argument("--duckdb-path")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--oos-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--output-dir", default="data/output/experiments/plan_05_18")
    args = parser.parse_args()

    out_dir = project_root() / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [str(s).zfill(6) for s in args.symbols]

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(
        _run_group(
            symbols=symbols,
            group="P1_exit",
            variants=_exit_variants(),
            out_dir=out_dir,
            start=args.start,
            end=args.end,
            config=args.config,
            duckdb_path=args.duckdb_path,
            train_days=args.train_days,
            oos_days=args.oos_days,
            step_days=args.step_days,
        )
    )
    all_rows.extend(
        _run_group(
            symbols=symbols,
            group="P1_dk_fade",
            variants=[{"id": f"fade_{n}", "bt": {"dk_fade_exit_n": n}} for n in [0, 2, 3, 5]],
            out_dir=out_dir,
            start=args.start,
            end=args.end,
            config=args.config,
            duckdb_path=args.duckdb_path,
            train_days=args.train_days,
            oos_days=args.oos_days,
            step_days=args.step_days,
        )
    )
    all_rows.extend(
        _run_group(
            symbols=symbols,
            group="P2_entry",
            variants=[{"id": "breakout_off", "bt": {"require_price_breakout": False}}]
            + [
                {"id": f"breakout_{n}", "bt": {"require_price_breakout": True, "breakout_lookback": n}}
                for n in [15, 20, 25]
            ],
            out_dir=out_dir,
            start=args.start,
            end=args.end,
            config=args.config,
            duckdb_path=args.duckdb_path,
            train_days=args.train_days,
            oos_days=args.oos_days,
            step_days=args.step_days,
        )
    )
    all_rows.extend(
        _run_group(
            symbols=symbols,
            group="P4_position",
            variants=[
                {
                    "id": f"vol_{str(vol).replace('.', '_')}_dd_{int(throttle)}",
                    "bt": {"volatility_target_ann": vol, "drawdown_throttle_enabled": throttle},
                }
                for vol in [0.0, 0.15, 0.20, 0.25]
                for throttle in [False, True]
            ],
            out_dir=out_dir,
            start=args.start,
            end=args.end,
            config=args.config,
            duckdb_path=args.duckdb_path,
            train_days=args.train_days,
            oos_days=args.oos_days,
            step_days=args.step_days,
        )
    )

    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "plan_05_18_summary.csv"
    summary.sort_values(["calmar_ratio", "annualized_return"], ascending=[False, False], na_position="last").to_csv(summary_path, index=False)
    print(f"Plan summary: {summary_path}")
    for group in ["P1_exit", "P1_dk_fade", "P2_entry", "P4_position"]:
        print(f"{group}: {out_dir / group / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
