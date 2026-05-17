#!/usr/bin/env python
"""Score DKBar-style trading candidates with legacy anchor diagnostics."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.config import build_bt_kwargs
from src.backtest.single_stock import run_single_stock_backtest
from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.settings import load_config, project_root

REQUIRED_ANCHOR_COLUMNS = {"symbol", "date", "lst", "bar_high", "bar_low", "bar_color"}


def load_anchors(path: str | Path, *, symbol: str | None = None) -> pd.DataFrame:
    """Load and validate legacy DKBar anchor rows."""
    anchor_path = Path(path).expanduser()
    if not anchor_path.is_absolute():
        anchor_path = project_root() / anchor_path
    anchors = pd.read_csv(anchor_path, dtype={"symbol": str})
    missing = REQUIRED_ANCHOR_COLUMNS - set(anchors.columns)
    if missing:
        raise ValueError(f"anchor file missing required columns: {sorted(missing)}")

    anchors = anchors.copy()
    anchors["symbol"] = anchors["symbol"].astype(str).str.zfill(6)
    anchors["date"] = pd.to_datetime(anchors["date"]).dt.normalize()
    anchors["bar_color"] = anchors["bar_color"].astype(str).str.lower().str.strip()
    bad_colors = sorted(set(anchors["bar_color"]) - {"red", "green"})
    if bad_colors:
        raise ValueError(f"anchor file contains unsupported bar_color values: {bad_colors}")

    for col in ["lst", "bar_high", "bar_low"]:
        anchors[col] = pd.to_numeric(anchors[col], errors="raise")

    if symbol is not None:
        anchors = anchors[anchors["symbol"] == str(symbol).zfill(6)].copy()
    return anchors.sort_values(["symbol", "date"]).reset_index(drop=True)


def _trend_switches_per_year(trend: pd.DataFrame) -> float:
    valid = trend[trend["trend_state"].isin(["red", "green"])].copy()
    if valid.empty:
        return 0.0
    switch_count = int((valid["trend_state"] != valid["trend_state"].shift(1)).sum()) - 1
    switch_count = max(switch_count, 0)
    dates = pd.to_datetime(valid["trade_date"])
    years = max((dates.max() - dates.min()).days / 365.0, 0.25)
    return switch_count / years


def score_candidate(
    ohlcv: pd.DataFrame,
    anchors: pd.DataFrame,
    params: DKTrendParams,
    *,
    label: str = "candidate",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return aggregate metrics and per-anchor diagnostics for one parameter candidate."""
    if anchors.empty:
        raise ValueError("anchors is empty")

    trend = compute_dktrend(ohlcv, params).reset_index(drop=True)
    trend["trade_date"] = pd.to_datetime(trend["trade_date"]).dt.normalize()
    merged = anchors.merge(
        trend[
            [
                "trade_date",
                "lst",
                "bar_high",
                "bar_low",
                "bar_color",
                "trend_state",
                "dk_signal",
            ]
        ],
        left_on="date",
        right_on="trade_date",
        how="left",
        suffixes=("_target", "_actual"),
    )

    for col in ["lst", "bar_high", "bar_low"]:
        merged[f"{col}_error"] = (
            pd.to_numeric(merged[f"{col}_actual"], errors="coerce")
            - pd.to_numeric(merged[f"{col}_target"], errors="coerce")
        ).abs()
    merged["bar_color_match"] = merged["bar_color_actual"] == merged["bar_color_target"]
    merged["reason"] = merged.apply(_anchor_reason, axis=1)

    matched = int(merged["trade_date"].notna().sum())
    color_count = len(merged)
    bar_errors = pd.concat([merged["bar_high_error"], merged["bar_low_error"]], ignore_index=True)
    metrics = {
        "label": label,
        "matched_anchors": matched,
        "missing_anchors": int(len(merged) - matched),
        "lst_mae": _safe_mean(merged["lst_error"]),
        "bar_mae": _safe_mean(bar_errors),
        "bar_color_accuracy": float(merged["bar_color_match"].fillna(False).mean()),
        "bar_color_hits": int(merged["bar_color_match"].sum()),
        "bar_color_total": color_count,
        "trend_switches_per_year": _trend_switches_per_year(trend),
    }
    return metrics, merged


def score_backtest_candidate(
    symbol: str,
    ohlcv: pd.DataFrame,
    params: DKTrendParams,
    bt_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return single-stock backtest metrics for one parameter candidate."""
    res = run_single_stock_backtest(symbol, ohlcv, params, **bt_kwargs)
    return {
        "total_return": float(res.total_return),
        "annualized_return": float(res.annualized_return),
        "excess_annualized_return": float(res.excess_annualized_return),
        "sharpe_ratio": float(res.sharpe_ratio),
        "max_drawdown": float(res.max_drawdown),
        "calmar_ratio": float(res.calmar_ratio),
        "n_trades": int(res.n_trades),
        "win_rate": float(res.win_rate),
        "avg_hold_days": float(res.avg_hold_days),
    }


def objective_score(metrics: dict[str, Any], *, sort_by: str) -> float:
    """Score candidates for ranking; anchors are diagnostics unless blended is requested."""
    ret = _finite(metrics.get("total_return"))
    ann = _finite(metrics.get("annualized_return"))
    excess = _finite(metrics.get("excess_annualized_return"))
    sharpe = _finite(metrics.get("sharpe_ratio"))
    max_dd = max(_finite(metrics.get("max_drawdown")), 0.0)
    color_acc = _finite(metrics.get("bar_color_accuracy"))
    lst_mae = max(_finite(metrics.get("lst_mae")), 0.0)
    bar_mae = max(_finite(metrics.get("bar_mae")), 0.0)
    switches = _finite(metrics.get("trend_switches_per_year"))
    trades = int(metrics.get("n_trades") or 0)

    low_trade_penalty = 0.30 if trades == 0 else (0.15 if trades == 1 else 0.0)
    return_quality = sharpe + 1.50 * ann + 0.50 * excess + 0.25 * ret - 1.25 * max_dd - low_trade_penalty
    anchor_fit = color_acc - 0.60 * lst_mae - 0.80 * bar_mae
    switch_penalty = max(0.0, abs(switches - 5.0) - 3.0) * 0.03

    if sort_by == "anchor_fit":
        return anchor_fit - switch_penalty
    if sort_by == "blended":
        return return_quality + 0.35 * anchor_fit - switch_penalty
    return return_quality


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if pd.notna(x) else default


def _anchor_reason(row: pd.Series) -> str:
    if pd.isna(row.get("trade_date")):
        return "missing date in OHLCV"
    if row.get("bar_color_match"):
        return "bar_color matched"
    return (
        f"target {row.get('bar_color_target')}, actual {row.get('bar_color_actual')}; "
        f"trend_state {row.get('trend_state')}"
    )


def parse_candidate_spec(spec: str, base: DKTrendParams) -> tuple[str, DKTrendParams]:
    """Parse ``label:key=value,key=value`` into a DKTrendParams candidate."""
    if ":" in spec:
        label, raw_pairs = spec.split(":", 1)
    else:
        label, raw_pairs = spec, spec
    data = asdict(base)
    data["mode"] = TrendMode.EASTMONEY_DKBAR.value
    for pair in [p.strip() for p in raw_pairs.split(",") if p.strip()]:
        if "=" not in pair:
            raise ValueError(f"invalid candidate override {pair!r}; expected key=value")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if key not in data:
            raise ValueError(f"unknown DKTrendParams field in candidate: {key}")
        data[key] = _coerce_value(raw_value.strip(), data[key])
    return label.strip() or "candidate", DKTrendParams.from_mapping(data)


def default_candidates(base: DKTrendParams) -> list[tuple[str, DKTrendParams]]:
    """Candidate set matching the current plan's documented baselines."""
    specs = [
        "config:",
        "persistent_config:bar_color_method=persistent_price_change",
        "ema250_wma5:lst_method=ema,lst_period=250,bar_method=wma,bar_period=5",
        "persistent_ema250_wma5:lst_method=ema,lst_period=250,bar_method=wma,bar_period=5,bar_color_method=persistent_price_change",
        "sma205_wma20:lst_method=sma,lst_period=205,bar_method=wma,bar_period=20",
        "persistent_sma205_wma20:lst_method=sma,lst_period=205,bar_method=wma,bar_period=20,bar_color_method=persistent_price_change",
        "sma60_wma20:lst_method=sma,lst_period=60,bar_method=wma,bar_period=20",
        "persistent_sma60_wma20:lst_method=sma,lst_period=60,bar_method=wma,bar_period=20,bar_color_method=persistent_price_change",
    ]
    return [parse_candidate_spec(spec, base) for spec in specs]


def grid_candidates(base: DKTrendParams) -> list[tuple[str, DKTrendParams]]:
    """Focused return-search grid; broad enough to find trading candidates quickly."""
    specs: list[str] = []
    for lst_method, lst_period, bar_method, bar_period, color_method, confirm_days, hyst in itertools.product(
        ["ema", "sma"],
        [40, 60, 80, 100, 150, 205, 250],
        ["wma"],
        [5, 10, 20],
        ["price_change"],
        [1, 2, 3],
        [0.0, 0.003],
    ):
        label = f"{lst_method}{lst_period}_{bar_method}{bar_period}_{color_method}_c{confirm_days}_h{hyst:g}"
        specs.append(
            f"{label}:"
            f"lst_method={lst_method},lst_period={lst_period},"
            f"bar_method={bar_method},bar_period={bar_period},"
            f"bar_color_method={color_method},state_confirm_days={confirm_days},hysteresis_pct={hyst}"
        )
    return [parse_candidate_spec(spec, base) for spec in specs]


def _coerce_value(raw: str, old_value: Any) -> Any:
    if isinstance(old_value, bool):
        return raw.lower() in {"1", "true", "yes", "y"}
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(raw)
    if isinstance(old_value, float):
        return float(raw)
    return raw


def _params_from_config(cfg: dict[str, Any]) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    raw["mode"] = TrendMode.EASTMONEY_DKBAR.value
    return DKTrendParams.from_mapping(raw)


def _read_ohlcv(symbol: str, start: str | None, end: str | None, config: str | None, duckdb_path: str | None) -> pd.DataFrame:
    with DuckDBManager(config_path=config, duckdb_path=duckdb_path) as db:
        df = db.read_daily_frame(symbols=[symbol], start=start, end=end)
    if df.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Score DKBar-style trading candidates by returns with anchor diagnostics.")
    parser.add_argument("--symbol", default="000783")
    parser.add_argument("--anchors", default="data/anchors/eastmoney_dkbar_000783.csv")
    parser.add_argument("--config", default="configs/eastmoney_dkbar_test.yaml")
    parser.add_argument("--duckdb-path")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end")
    parser.add_argument("--candidate", action="append", help="label:key=value,key=value; repeatable")
    parser.add_argument("--grid-search", action="store_true", help="Search a focused DK bar parameter grid")
    parser.add_argument("--max-candidates", type=int, default=0, help="Limit candidates after generation; 0 keeps all")
    parser.add_argument("--top-n", type=int, default=30, help="Rows to print after sorting; CSV output keeps all rows")
    parser.add_argument("--no-backtest", action="store_true", help="Only score anchors; skip return metrics")
    parser.add_argument(
        "--sort-by",
        choices=["return_quality", "blended", "anchor_fit"],
        default="return_quality",
        help="Ranking objective. Default prioritizes backtest return quality.",
    )
    parser.add_argument("--output-csv", help="Optional aggregate metrics CSV path")
    parser.add_argument("--details", action="store_true", help="Print per-anchor diagnostics for each candidate")
    args = parser.parse_args()

    symbol = str(args.symbol).zfill(6)
    cfg = load_config(args.config)
    base = _params_from_config(cfg)
    anchors = load_anchors(args.anchors, symbol=symbol)
    df = _read_ohlcv(symbol, args.start, args.end, args.config, args.duckdb_path)

    if args.candidate:
        candidates = [parse_candidate_spec(spec, base) for spec in args.candidate]
    elif args.grid_search:
        candidates = grid_candidates(base)
    else:
        candidates = default_candidates(base)
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[: int(args.max_candidates)]

    bt_kwargs = build_bt_kwargs(cfg) if not args.no_backtest else {}
    bt_kwargs["stock_name"] = symbol

    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for label, params in candidates:
        metrics, detail = score_candidate(df, anchors, params, label=label)
        if not args.no_backtest:
            metrics.update(score_backtest_candidate(symbol, df, params, bt_kwargs))
        metrics["objective_score"] = objective_score(metrics, sort_by=args.sort_by)
        rows.append(metrics)
        detail.insert(0, "label", label)
        detail_frames.append(detail)

    sort_cols = ["objective_score", "bar_color_accuracy"] if args.no_backtest else [
        "objective_score",
        "total_return",
        "sharpe_ratio",
        "bar_color_accuracy",
    ]
    metrics_df = pd.DataFrame(rows).sort_values(
        sort_cols,
        ascending=[False] * len(sort_cols),
        na_position="last",
    )
    print(metrics_df.head(max(int(args.top_n), 1)).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.details:
        detail_df = pd.concat(detail_frames, ignore_index=True)
        columns = [
            "label",
            "date",
            "lst_target",
            "lst_actual",
            "bar_high_target",
            "bar_high_actual",
            "bar_low_target",
            "bar_low_actual",
            "bar_color_target",
            "bar_color_actual",
            "trend_state",
            "reason",
        ]
        print()
        print(detail_df[columns].to_string(index=False))

    if args.output_csv:
        output_path = Path(args.output_csv).expanduser()
        if not output_path.is_absolute():
            output_path = project_root() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
