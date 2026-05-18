#!/usr/bin/env python
"""Export current best configs and trade details for selected stocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.config import build_bt_kwargs
from src.backtest.single_stock import run_single_stock_backtest
from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams
from src.settings import load_config, project_root


def _read_watchlist(path: Path) -> list[str]:
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            symbols.append(value.zfill(6))
    if not symbols:
        raise SystemExit(f"empty watchlist: {path}")
    return symbols


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_native(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _load_wfo(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing WFO result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_platform_params(wfo: dict[str, Any]) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    folds = wfo.get("platform_by_fold") or wfo.get("best_params_by_fold") or []
    if not folds:
        raise ValueError(f"WFO result has no selected fold params for {wfo.get('symbol')}")
    latest = max(folds, key=lambda row: int(row.get("fold", -1)))
    return int(latest.get("fold", -1)), dict(latest.get("params") or {}), folds


def _trend_config(base_cfg: dict[str, Any], selected_params: dict[str, Any]) -> dict[str, Any]:
    trend_cfg = dict(base_cfg.get("trend_signal", {}) or {})
    trend_cfg.update(selected_params)
    trend_cfg["mode"] = str(trend_cfg.get("mode", "macd_cross"))
    return trend_cfg


def _summary_row(symbol: str, stock_name: str, res: Any, wfo: dict[str, Any], selected_fold: int, params: dict[str, Any]) -> dict[str, Any]:
    agg = wfo.get("aggregated", {}) or {}
    return {
        "symbol": symbol,
        "stock_name": stock_name,
        "selection_source": f"E_SINGLE_stable/{symbol}_wfo_20260515.json",
        "selection_method": "latest_platform_fold",
        "selected_fold": selected_fold,
        "mode": wfo.get("mode", "macd_cross"),
        "macd_fast": params.get("macd_fast"),
        "macd_slow": params.get("macd_slow", 26),
        "macd_signal": params.get("macd_signal"),
        "min_run_len": params.get("min_run_len", 1),
        "wfo_oos_total_return": agg.get("total_return_combined"),
        "wfo_oos_annualized_return": agg.get("annualized_return_combined"),
        "wfo_oos_sharpe": agg.get("sharpe_ratio_combined"),
        "wfo_oos_max_drawdown": agg.get("max_drawdown_combined"),
        "wfo_oos_calmar": agg.get("calmar_ratio_combined"),
        "backtest_period": res.period,
        "total_return": res.total_return,
        "annualized_return": res.annualized_return,
        "buy_hold_return": res.buy_hold_return,
        "sharpe_ratio": res.sharpe_ratio,
        "max_drawdown": res.max_drawdown,
        "calmar_ratio": res.calmar_ratio,
        "n_trades": res.n_trades,
        "win_rate": res.win_rate,
        "avg_hold_days": res.avg_hold_days,
        "profit_lock_exits": res.profit_lock_exits,
        "time_stop_exits": res.time_stop_exits,
        "stop_loss_exits": res.stop_loss_exits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export selected single-stock best configs and trades.")
    parser.add_argument("--config", default="configs/prod-v1.yaml")
    parser.add_argument("--watchlist", default="configs/watchlist_wfo_passing.txt")
    parser.add_argument("--wfo-dir", default="data/output/experiments/E_SINGLE_stable")
    parser.add_argument("--output-dir", default="data/output/selected_single_stock_best")
    parser.add_argument("--config-output", default="configs/research/selected_single_stock_best.yaml")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = _read_watchlist(project_root() / args.watchlist)
    output_dir = project_root() / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    name_cache_path = resolve_stock_name_cache_path(cfg)
    names = resolve_stock_names(symbols, name_cache_path)
    benchmark_symbol = str((cfg.get("risk", {}) or {}).get("benchmark_symbol", "510300")).zfill(6)
    read_symbols = list(symbols)
    if bool((cfg.get("risk", {}) or {}).get("enable_index_filter", False)) and benchmark_symbol not in read_symbols:
        read_symbols.append(benchmark_symbol)

    with DuckDBManager(config_path=args.config) as db:
        all_df = db.read_daily_frame(symbols=read_symbols, start=args.start, end=args.end)

    rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    configs: dict[str, Any] = {
        "generated_from": "scripts/export_selected_single_stock_best.py",
        "base_config": args.config,
        "watchlist": args.watchlist,
        "selection_method": "latest_platform_fold from WFO platform_by_fold",
        "shared_backtest": cfg.get("backtest", {}),
        "shared_signal_filter": cfg.get("signal_filter", {}),
        "symbols": {},
    }

    for symbol in symbols:
        wfo_path = project_root() / args.wfo_dir / f"{symbol}_wfo_20260515.json"
        wfo = _load_wfo(wfo_path)
        selected_fold, selected_params, fold_rows = _latest_platform_params(wfo)
        trend_cfg = _trend_config(cfg, selected_params)
        params = DKTrendParams.from_mapping(trend_cfg)
        stock_df = all_df[all_df["symbol"].astype(str).str.zfill(6) == symbol].copy()
        if stock_df.empty:
            raise SystemExit(f"no daily data found for {symbol}")
        stock_name = names.get(symbol, symbol)
        bt_kwargs = build_bt_kwargs(cfg)
        bt_kwargs["stock_name"] = stock_name
        res = run_single_stock_backtest(symbol, stock_df, params, **bt_kwargs)
        rows.append(_summary_row(symbol, stock_name, res, wfo, selected_fold, trend_cfg))

        trades = res.trade_log.copy()
        trades.insert(0, "trade_no", range(1, len(trades) + 1))
        trades.insert(0, "stock_name", stock_name)
        trades.insert(0, "symbol", symbol)
        trade_frames.append(trades)
        trades.to_csv(output_dir / f"{symbol}_trades.csv", index=False)

        configs["symbols"][symbol] = {
            "stock_name": stock_name,
            "wfo_source": str(wfo_path.relative_to(project_root())),
            "selected_fold": selected_fold,
            "fold_selected_params": [
                {"fold": int(row.get("fold", -1)), "params": dict(row.get("params") or {})}
                for row in fold_rows
            ],
            "trend_signal": trend_cfg,
            "backtest_result": {
                "period": res.period,
                "total_return": res.total_return,
                "annualized_return": res.annualized_return,
                "sharpe_ratio": res.sharpe_ratio,
                "max_drawdown": res.max_drawdown,
                "calmar_ratio": res.calmar_ratio,
                "n_trades": res.n_trades,
                "win_rate": res.win_rate,
            },
        }

    summary = pd.DataFrame(rows)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary.to_csv(output_dir / "best_configs_summary.csv", index=False)
    all_trades.to_csv(output_dir / "trade_details.csv", index=False)
    config_output = project_root() / args.config_output
    config_output.parent.mkdir(parents=True, exist_ok=True)
    rendered_yaml = yaml.safe_dump(_native(configs), allow_unicode=True, sort_keys=False)
    config_output.write_text(rendered_yaml, encoding="utf-8")
    (output_dir / "best_configs.yaml").write_text(
        rendered_yaml,
        encoding="utf-8",
    )

    print(f"wrote {config_output}")
    print(f"wrote {output_dir / 'best_configs.yaml'}")
    print(f"wrote {output_dir / 'best_configs_summary.csv'}")
    print(f"wrote {output_dir / 'trade_details.csv'}")
    print(f"wrote {len(symbols)} per-symbol trade files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
