#!/usr/bin/env python
"""Generate the 000783 return-improvement plan outputs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_eastmoney_dkbar import return_quality_score, score_backtest_candidate
from src.backtest.config import build_bt_kwargs
from src.backtest.performance_panel import compute_performance_panel
from src.backtest.single_stock import run_single_stock_backtest
from src.backtest.wfo import _VALID_BT_KWARGS, _oos_trend_with_warmup, trade_contribution_metrics
from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, TrendMode
from src.settings import load_config, project_root

LABEL_RE = re.compile(
    r"^(?P<lst_method>ema|sma)(?P<lst_period>\d+)_"
    r"(?P<bar_method>wma)(?P<bar_period>\d+)_"
    r"(?P<color>price_change)_c(?P<confirm>\d+)_h(?P<hysteresis>[0-9.]+)$"
)


def _params_from_config(cfg: dict[str, Any]) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    raw["mode"] = TrendMode.EASTMONEY_DKBAR.value
    return DKTrendParams.from_mapping(raw)


def params_from_label(label: str, base: DKTrendParams) -> DKTrendParams:
    match = LABEL_RE.match(str(label))
    if not match:
        raise ValueError(f"unsupported candidate label: {label}")
    data = asdict(base)
    data.update(
        {
            "mode": TrendMode.EASTMONEY_DKBAR.value,
            "lst_method": match.group("lst_method"),
            "lst_period": int(match.group("lst_period")),
            "bar_method": match.group("bar_method"),
            "bar_period": int(match.group("bar_period")),
            "bar_color_method": match.group("color"),
            "state_confirm_days": int(match.group("confirm")),
            "hysteresis_pct": float(match.group("hysteresis")),
        }
    )
    return DKTrendParams.from_mapping(data)


def _read_ohlcv(symbol: str, start: str | None, end: str | None, config: str, duckdb_path: str | None) -> pd.DataFrame:
    with DuckDBManager(config_path=config, duckdb_path=duckdb_path) as db:
        df = db.read_daily_frame(symbols=[symbol], start=start, end=end)
    if df.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    return df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def enrich_candidates(
    candidates: pd.DataFrame,
    *,
    symbol: str,
    ohlcv: pd.DataFrame,
    base_params: DKTrendParams,
    bt_kwargs: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        label = str(row["label"])
        params = params_from_label(label, base_params)
        metrics = score_backtest_candidate(symbol, ohlcv, params, bt_kwargs)
        out = row.to_dict()
        out.update(metrics)
        out["return_quality"] = return_quality_score(out)
        rows.append(out)
    return pd.DataFrame(rows).sort_values(
        ["return_quality", "total_return", "calmar_ratio", "n_trades"],
        ascending=[False, False, False, False],
        na_position="last",
    )


def fixed_param_wfo(
    *,
    symbol: str,
    ohlcv: pd.DataFrame,
    params: DKTrendParams,
    bt_kwargs: dict[str, Any],
    label: str,
    experiment_id: str,
    train_days: int,
    oos_days: int,
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
        train_df = ohlcv.iloc[start:start + train_days].copy()
        oos_df = ohlcv.iloc[start + train_days:start + train_days + oos_days].copy()
        oos_trend = _oos_trend_with_warmup(train_df, oos_df, params, run_kwargs)
        res = run_single_stock_backtest(
            symbol,
            oos_df,
            params,
            cost_bps=cost_bps,
            initial_capital=initial_capital,
            trend_override=oos_trend,
            **run_kwargs,
        )
        returns.append(res.daily_returns)
        if not res.trade_log.empty:
            t = res.trade_log.copy()
            t.insert(0, "fold", fold)
            t.insert(0, "experiment_id", experiment_id)
            t.insert(1, "label", label)
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
                **trade_contribution_metrics(res.trade_log, res.total_return),
            }
        )
        fold += 1
        start += oos_days

    stitched = pd.concat(returns).sort_index() if returns else pd.Series(dtype=float)
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    panel = compute_performance_panel(stitched.to_numpy(dtype=np.float64)) if not stitched.empty else None
    contrib = trade_contribution_metrics(all_trades, panel.total_return if panel is not None else float("nan"))
    summary = {
        "experiment_id": experiment_id,
        "label": label,
        "n_folds": len(fold_rows),
        "total_return_oos": float(panel.total_return) if panel is not None else float("nan"),
        "annualized_return_oos": float(panel.annualized_return) if panel is not None else float("nan"),
        "max_drawdown_oos": float(panel.max_drawdown) if panel is not None else float("nan"),
        "calmar_oos": float(panel.calmar_ratio) if panel is not None else float("nan"),
        "sharpe_oos": float(panel.sharpe_ratio) if panel is not None else float("nan"),
        "n_trades": int(len(all_trades)),
        "win_rate": float((pd.to_numeric(all_trades["return"], errors="coerce") > 0).mean()) if not all_trades.empty else float("nan"),
        **contrib,
        "folds": fold_rows,
    }
    summary["return_quality_oos"] = return_quality_score(
        {
            "total_return": summary["total_return_oos"],
            "annualized_return": summary["annualized_return_oos"],
            "max_drawdown": summary["max_drawdown_oos"],
            "calmar_ratio": summary["calmar_oos"],
            "sharpe_ratio": summary["sharpe_oos"],
            "n_trades": summary["n_trades"],
            "largest_trade_contribution": summary["largest_trade_contribution"],
        }
    )
    return summary, all_trades


def _candidate_overrides(row: pd.Series) -> dict[str, Any]:
    params = params_from_label(str(row["label"]), DKTrendParams(mode=TrendMode.EASTMONEY_DKBAR))
    return {
        "lst_method": params.lst_method,
        "lst_period": params.lst_period,
        "bar_method": params.bar_method,
        "bar_period": params.bar_period,
        "bar_color_method": params.bar_color_method,
        "state_confirm_days": params.state_confirm_days,
        "hysteresis_pct": params.hysteresis_pct,
    }


def _experiment_variants(top_candidates: pd.DataFrame) -> list[dict[str, Any]]:
    attack = top_candidates[top_candidates["label"].astype(str).str.startswith("sma60_")].head(1)
    defense = top_candidates[top_candidates["label"].astype(str).str.startswith("sma205_")].head(1)
    seeds = pd.concat([attack, defense], ignore_index=True)
    variants: list[dict[str, Any]] = []
    for _, row in seeds.iterrows():
        label = str(row["label"])
        params_override = _candidate_overrides(row)
        variants.append({"suffix": "baseline", "label": label, "params": params_override, "bt": {}})
        for limit in [0.04, 0.05, 0.06, 0.07, 0.08]:
            variants.append({"suffix": f"intrapos_dd_{limit:.2f}", "label": label, "params": params_override, "bt": {"intrapos_dd_limit": limit}})
        for fade in [3, 4, 5]:
            variants.append({"suffix": f"dk_fade_{fade}", "label": label, "params": params_override, "bt": {"dk_fade_exit_n": fade}})
        for trigger, trailing in [(0.08, 0.04), (0.10, 0.05), (0.12, 0.06)]:
            variants.append({
                "suffix": f"profit_lock_{trigger:.2f}_{trailing:.2f}",
                "label": label,
                "params": params_override,
                "bt": {"profit_lock_trigger": trigger, "profit_lock_trailing": trailing},
            })
        for confirm in [1, 2]:
            p = dict(params_override)
            p["state_confirm_days"] = confirm
            variants.append({"suffix": f"confirm_{confirm}", "label": label, "params": p, "bt": {}})
    return variants


def _select_best(rows: pd.DataFrame) -> pd.Series:
    work = rows.copy()
    eligible = work[
        (work["total_return_oos"] > 0)
        & (work["max_drawdown_oos"] <= 0.15)
        & (work["n_trades"] >= 3)
        & ((work["largest_trade_contribution"].isna()) | (work["largest_trade_contribution"] <= 0.60))
    ]
    if eligible.empty:
        eligible = work[(work["total_return_oos"] > 0) & (work["max_drawdown_oos"] <= 0.20)]
    if eligible.empty:
        eligible = work[work["total_return_oos"] > 0]
    if eligible.empty:
        eligible = work
    return eligible.sort_values(
        ["return_quality_oos", "total_return_oos", "calmar_oos"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_native(v) for v in value]
    if isinstance(value, tuple):
        return [_native(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_best_config(path: Path, cfg: dict[str, Any], best: pd.Series) -> None:
    trend = dict(cfg.get("trend_signal", {}) or {})
    trend.update({k: _native(best[k]) for k in ["lst_method", "lst_period", "bar_method", "bar_period", "bar_color_method", "state_confirm_days", "hysteresis_pct"] if k in best})
    trend["mode"] = TrendMode.EASTMONEY_DKBAR.value
    backtest = dict(cfg.get("backtest", {}) or {})
    for key in ["intrapos_dd_limit", "dk_fade_exit_n", "profit_lock_trigger", "profit_lock_trailing"]:
        if key in best and pd.notna(best[key]):
            value = best[key]
            backtest[key] = int(value) if key == "dk_fade_exit_n" else float(value)

    payload = {
        "paths": cfg.get("paths", {}),
        "trend_signal": trend,
        "backtest": backtest,
        "signal_filter": cfg.get("signal_filter", {}),
        "research_result": {
            "source": "scripts/run_000783_return_plan.py",
            "experiment_id": str(best["experiment_id"]),
            "total_return_oos": float(best["total_return_oos"]),
            "annualized_return_oos": float(best["annualized_return_oos"]),
            "max_drawdown_oos": float(best["max_drawdown_oos"]),
            "calmar_oos": float(best["calmar_oos"]),
            "n_trades": int(best["n_trades"]),
            "largest_trade_contribution": None
            if pd.isna(best["largest_trade_contribution"])
            else float(best["largest_trade_contribution"]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_native(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Complete the 000783 single-stock return plan outputs.")
    parser.add_argument("--symbol", default="000783")
    parser.add_argument("--config", default="configs/eastmoney_dkbar_test.yaml")
    parser.add_argument("--grid-csv", default="data/output/000783_dkbar_return_grid_full.csv")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end")
    parser.add_argument("--duckdb-path")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--oos-days", type=int, default=252)
    args = parser.parse_args()

    symbol = str(args.symbol).zfill(6)
    cfg = load_config(args.config)
    base_params = _params_from_config(cfg)
    bt_kwargs = build_bt_kwargs(cfg)
    bt_kwargs["stock_name"] = symbol
    ohlcv = _read_ohlcv(symbol, args.start, args.end, args.config, args.duckdb_path)

    grid_path = project_root() / args.grid_csv
    raw_candidates = pd.read_csv(grid_path)
    candidates = enrich_candidates(
        raw_candidates,
        symbol=symbol,
        ohlcv=ohlcv,
        base_params=base_params,
        bt_kwargs=bt_kwargs,
    )
    out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / f"{symbol}_return_candidates.csv"
    candidates.to_csv(candidate_path, index=False)

    top = candidates.head(max(int(args.top_n), 1)).copy()
    wfo_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        label = str(getattr(row, "label"))
        params = params_from_label(label, base_params)
        summary, trades = fixed_param_wfo(
            symbol=symbol,
            ohlcv=ohlcv,
            params=params,
            bt_kwargs=bt_kwargs,
            label=label,
            experiment_id=f"R02_top{rank:02d}",
            train_days=int(args.train_days),
            oos_days=int(args.oos_days),
        )
        summary.update(_candidate_overrides(pd.Series({"label": label})))
        wfo_rows.append(summary)
        if not trades.empty:
            trade_frames.append(trades)

    wfo_df = pd.DataFrame([{k: v for k, v in row.items() if k != "folds"} for row in wfo_rows])
    wfo_path = out_dir / f"{symbol}_top10_wfo_contribution.csv"
    wfo_df.sort_values(["return_quality_oos", "total_return_oos"], ascending=[False, False]).to_csv(wfo_path, index=False)
    trades_path = out_dir / f"{symbol}_top10_wfo_trades.csv"
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(trades_path, index=False)
    else:
        pd.DataFrame().to_csv(trades_path, index=False)

    exp_rows: list[dict[str, Any]] = []
    for idx, variant in enumerate(_experiment_variants(candidates), start=1):
        params_data = asdict(base_params)
        params_data.update(variant["params"])
        params_data["mode"] = TrendMode.EASTMONEY_DKBAR.value
        bt = {**bt_kwargs, **variant["bt"]}
        summary, _ = fixed_param_wfo(
            symbol=symbol,
            ohlcv=ohlcv,
            params=DKTrendParams.from_mapping(params_data),
            bt_kwargs=bt,
            label=str(variant["label"]),
            experiment_id=f"R03_R04_{idx:02d}_{variant['suffix']}",
            train_days=int(args.train_days),
            oos_days=int(args.oos_days),
        )
        summary.update(variant["params"])
        summary.update(variant["bt"])
        exp_rows.append(summary)

    exp_df = pd.DataFrame([{k: v for k, v in row.items() if k != "folds"} for row in exp_rows])
    exp_path = out_dir / f"{symbol}_exit_entry_experiments.csv"
    exp_df.sort_values(["return_quality_oos", "total_return_oos"], ascending=[False, False]).to_csv(exp_path, index=False)

    all_results = pd.concat([wfo_df, exp_df], ignore_index=True, sort=False)
    best = _select_best(all_results)
    best_config_path = project_root() / f"configs/research/{symbol}_best_return.yaml"
    _write_best_config(best_config_path, cfg, best)

    print(f"R01 candidates: {candidate_path}")
    print(f"R02 WFO summary: {wfo_path}")
    print(f"R02 trade contribution: {trades_path}")
    print(f"R03/R04 experiments: {exp_path}")
    print(f"R05 best config: {best_config_path}")
    print(
        "Best: "
        f"{best['experiment_id']} total={best['total_return_oos']:.4f} "
        f"ann={best['annualized_return_oos']:.4f} mdd={best['max_drawdown_oos']:.4f} "
        f"calmar={best['calmar_oos']:.2f} trades={int(best['n_trades'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
