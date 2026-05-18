#!/usr/bin/env python
"""Validate signal-quality rules against forward returns.

Example:
python scripts/validate_signal_quality_rules.py \
  --symbols 000783 300750 600519 \
  --start 2018-01-01 --end 2026-05-01 \
  --output data/output/signal_quality_validation/
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

from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, compute_dktrend
from src.settings import load_config


def _mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _cliffs_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return float((gt - lt) / (len(a) * len(b)))


def _rule_frame(df: pd.DataFrame, trend: pd.DataFrame, market_returns: pd.Series | None) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce").reset_index(drop=True)
    volume = pd.to_numeric(df.get("volume", pd.Series(np.nan, index=df.index)), errors="coerce").reset_index(drop=True)
    high = pd.to_numeric(df["high"], errors="coerce").reset_index(drop=True)
    low = pd.to_numeric(df["low"], errors="coerce").reset_index(drop=True)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    atr_median = atr.rolling(60, min_periods=60).median()
    avg_vol = volume.rolling(20, min_periods=20).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    stock_ret_60 = close.pct_change(60)

    out = pd.DataFrame(index=trend.index)
    out["run_len_ge_2"] = pd.to_numeric(trend.get("dk_run_len", 0), errors="coerce") >= 2
    out["volume_ge_1_3x"] = volume >= avg_vol * 1.3
    out["consensus_red_ge_2"] = (
        pd.to_numeric(trend["consensus_red_count"], errors="coerce") >= 2
        if "consensus_red_count" in trend.columns
        else False
    )
    out["atr_below_median"] = atr < atr_median
    out["atr_above_median"] = atr > atr_median
    out["close_above_ma20"] = close > ma20
    out["ma20_slope_gt_0"] = (ma20 - ma20.shift(5)) > 0
    if market_returns is not None and not market_returns.empty:
        aligned = market_returns.reset_index(drop=True).reindex(out.index).fillna(0.0)
        market_cum_10 = (1.0 + aligned).rolling(10, min_periods=10).apply(
            lambda x: (x + 1.0).prod() - 1.0,
            raw=True,
        )
        idx_cum_60 = (1.0 + aligned).rolling(60, min_periods=60).apply(
            lambda x: (x + 1.0).prod() - 1.0,
            raw=True,
        )
        out["market_10d_gt_0"] = market_cum_10 > 0
        out["rs_60_gt_0"] = stock_ret_60 > idx_cum_60
    else:
        out["market_10d_gt_0"] = False
        out["rs_60_gt_0"] = stock_ret_60 > 0
    return out.fillna(False)


def _summarize_rule(symbol: str, rule: str, mask: pd.Series, forward_ret: pd.Series) -> dict:
    valid = forward_ret.notna()
    a = forward_ret[valid & mask].to_numpy(dtype=np.float64)
    b = forward_ret[valid & ~mask].to_numpy(dtype=np.float64)
    return {
        "symbol": symbol,
        "rule": rule,
        "n_true": int(len(a)),
        "n_false": int(len(b)),
        "mean_true": float(np.mean(a)) if len(a) else float("nan"),
        "mean_false": float(np.mean(b)) if len(b) else float("nan"),
        "median_true": float(np.median(a)) if len(a) else float("nan"),
        "median_false": float(np.median(b)) if len(b) else float("nan"),
        "win_rate_true": float(np.mean(a > 0)) if len(a) else float("nan"),
        "win_rate_false": float(np.mean(b > 0)) if len(b) else float("nan"),
        "mann_whitney_p": _mann_whitney_p(a, b) if len(a) and len(b) else float("nan"),
        "cliffs_d": _cliffs_d(a, b),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate signal quality rules with forward returns")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end")
    parser.add_argument("--forward-days", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--duckdb-path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = [str(s).strip().zfill(6) for s in args.symbols]
    risk_cfg = cfg.get("risk", {}) or {}
    benchmark_symbol = str(risk_cfg.get("benchmark_symbol", "510300")).strip().zfill(6)
    read_symbols = sorted(set(symbols + [benchmark_symbol]))
    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        data = db.read_daily_frame(symbols=read_symbols, start=args.start, end=args.end)

    params = DKTrendParams.from_mapping(dict(cfg.get("trend_signal", {}) or {}))
    rows: list[dict] = []
    for symbol in symbols:
        df = data[data["symbol"].astype(str).str.zfill(6) == symbol].copy().reset_index(drop=True)
        if df.empty:
            continue
        idx_df = data[data["symbol"].astype(str).str.zfill(6) == benchmark_symbol].copy().reset_index(drop=True)
        market_returns = pd.to_numeric(idx_df["close"], errors="coerce").pct_change() if not idx_df.empty else None
        trend = compute_dktrend(df, params).reset_index(drop=True)
        buy_mask = trend["dk_signal"].astype(str).eq("buy")
        close = pd.to_numeric(df["close"], errors="coerce")
        forward_ret = close.shift(-int(args.forward_days)) / close - 1.0
        rule_df = _rule_frame(df, trend, market_returns)
        for rule in rule_df.columns:
            rows.append(_summarize_rule(symbol, rule, rule_df[rule] & buy_mask, forward_ret.where(buy_mask)))

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    out_path = out_dir / "signal_quality_rule_validation.csv"
    result.to_csv(out_path, index=False)
    print(f"written {out_path}")
    if not result.empty:
        ranked = result.assign(edge=result["median_true"] - result["median_false"]).sort_values("edge", ascending=False)
        print(ranked[["symbol", "rule", "n_true", "n_false", "edge", "cliffs_d", "mann_whitney_p"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
