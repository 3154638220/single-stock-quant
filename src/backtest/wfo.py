"""Walk-forward optimization for DK trend parameters."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from statistics import mode
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.performance_panel import aggregate_panels, compute_performance_panel
from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams, TrendMode

DEFAULT_PARAM_GRID: dict[str, list] = {
    "macd_fast": [8, 10, 12, 14],
    "macd_slow": [22, 26, 30],
    "macd_signal": [7, 9, 11],
    "min_run_len": [1, 2, 3],
    "stop_loss_pct": [0.05, 0.08, 0.10],
}

_BT_PARAM_KEYS = {
    "stop_loss_pct", "trailing_stop_pct", "atr_stop_multiplier", "atr_stop_period",
    "volume_confirm", "volume_lookback", "volume_ratio_min",
    "risk_per_trade_pct", "position_size_cap",
    "stop_reentry_enabled", "stop_reentry_cooldown", "stop_reentry_min_run",
    "min_quality_score",
}

# All kwargs that run_single_stock_backtest accepts (beyond cost_bps, cost_params, initial_capital).
_VALID_BT_KWARGS = {
    "stock_name", "volume_confirm", "volume_lookback", "volume_ratio_min",
    "consensus_n_agree", "enable_index_filter", "index_ohlcv",
    "benchmark_symbol", "extreme_lookback_days", "extreme_drop_threshold",
    "risk_off_factor", "stop_loss_pct", "trailing_stop_pct",
    "atr_stop_multiplier", "atr_stop_period", "risk_per_trade_pct",
    "position_size_cap", "stop_reentry_enabled", "stop_reentry_cooldown",
    "stop_reentry_min_run", "cost_params",
    "min_quality_score", "quality_score_mode",
}


def _param_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def normalize_param_grid(raw: dict[str, Any] | None) -> dict[str, list[Any]]:
    """Return a clean parameter grid, falling back to defaults when no grid is configured."""
    if not raw:
        return {k: list(v) for k, v in DEFAULT_PARAM_GRID.items()}
    grid: dict[str, list[Any]] = {}
    for key, value in raw.items():
        if isinstance(value, (list, tuple)):
            vals = list(value)
        else:
            vals = [value]
        if vals:
            grid[str(key)] = vals
    return grid


def _params_with(base: DKTrendParams, overrides: dict[str, Any], mode_value: str) -> tuple[DKTrendParams, dict[str, Any]]:
    data = asdict(base)
    bt_kwargs: dict[str, Any] = {}
    trend_overrides = dict(overrides)
    for key in _BT_PARAM_KEYS:
        if key in trend_overrides:
            bt_kwargs[key] = trend_overrides.pop(key)
    data.update(trend_overrides)
    data["mode"] = mode_value
    return DKTrendParams.from_mapping(data), bt_kwargs


def _stability(best_by_fold: list[dict[str, Any]]) -> dict[str, Any]:
    params = [x["params"] for x in best_by_fold]
    out: dict[str, Any] = {}
    if not params:
        return out
    for key in params[0]:
        vals = [p[key] for p in params]
        numeric = np.array(vals, dtype=np.float64)
        try:
            param_mode = mode(vals)
        except Exception:
            param_mode = vals[0]
        out[key] = {
            "mode": param_mode,
            "variance": float(np.var(numeric)) if numeric.size else float("nan"),
        }
    is_sharpes = np.array([x["is_sharpe"] for x in best_by_fold], dtype=np.float64)
    oos_sharpes = np.array([x["oos_sharpe"] for x in best_by_fold], dtype=np.float64)
    valid = np.isfinite(is_sharpes) & np.isfinite(oos_sharpes)
    out["is_oos_sharpe_corr"] = (
        float(np.corrcoef(is_sharpes[valid], oos_sharpes[valid])[0, 1]) if int(valid.sum()) >= 2 else float("nan")
    )
    return out


def run_walk_forward_optimization(
    symbol: str,
    ohlcv: pd.DataFrame,
    *,
    base_params: DKTrendParams | None = None,
    param_grid: dict[str, list[Any]] | None = None,
    train_days: int = 504,
    oos_days: int = 126,
    mode: TrendMode | str = TrendMode.MACD_CROSS,
    window: str = "rolling",
    cost_bps: float = 15.0,
    initial_capital: float = 100_000.0,
    bt_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run rolling or expanding WFO and return a JSON-serializable report."""
    code = str(symbol).strip().zfill(6)
    df = ohlcv.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize() if "trade_date" in df else pd.to_datetime(df.index).normalize()
    df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    if len(df) < train_days + oos_days:
        raise ValueError("not enough rows for requested train/oos windows")

    mode_value = mode.value if isinstance(mode, TrendMode) else str(mode)
    base = base_params or DKTrendParams(mode=mode_value)
    grid = normalize_param_grid(param_grid)
    combos = _param_combinations(grid)
    if not combos:
        raise ValueError("param_grid is empty")

    base_bt = dict(bt_kwargs or {})
    panels = []
    oos_returns = []
    oos_panel_dicts = []
    best_by_fold = []
    start = 0
    fold = 0
    while start + train_days + oos_days <= len(df):
        train_start = 0 if str(window).lower() == "expanding" else start
        train_end = start + train_days
        oos_start = train_end
        oos_end = train_end + oos_days
        train_df = df.iloc[train_start:train_end].copy()
        oos_df = df.iloc[oos_start:oos_end].copy()

        best_params = combos[0]
        best_sharpe = -np.inf
        best_bt: dict[str, Any] = {}
        for combo in combos:
            params, combo_bt = _params_with(base, combo, mode_value)
            bt = {**base_bt, **combo_bt}
            res = run_single_stock_backtest(
                code,
                train_df,
                params,
                cost_bps=cost_bps,
                initial_capital=initial_capital,
                **{k: v for k, v in bt.items() if k in _VALID_BT_KWARGS},
            )
            score = res.sharpe_ratio if np.isfinite(res.sharpe_ratio) else -np.inf
            if score > best_sharpe:
                best_sharpe = float(score)
                best_params = combo
                best_bt = combo_bt

        selected, sel_bt = _params_with(base, best_params, mode_value)
        bt_final = {**base_bt, **best_bt}
        oos_res = run_single_stock_backtest(
            code,
            oos_df,
            selected,
            cost_bps=cost_bps,
            initial_capital=initial_capital,
            **{k: v for k, v in bt_final.items() if k in _VALID_BT_KWARGS},
        )
        panel = compute_performance_panel(
            oos_res.daily_returns.to_numpy(dtype=np.float64),
            n_concurrent_strategies=len(combos),
        )
        panels.append(panel)
        oos_returns.append(oos_res.daily_returns)
        oos_panel = panel.to_dict()
        oos_panel.update(
            {
                "fold": fold,
                "start": pd.Timestamp(oos_df["trade_date"].iloc[0]).date().isoformat(),
                "end": pd.Timestamp(oos_df["trade_date"].iloc[-1]).date().isoformat(),
            }
        )
        oos_panel_dicts.append(oos_panel)
        best_by_fold.append(
            {
                "fold": fold,
                "params": dict(best_params),
                "is_sharpe": best_sharpe if np.isfinite(best_sharpe) else float("nan"),
                "oos_sharpe": panel.sharpe_ratio,
            }
        )
        fold += 1
        start += oos_days

    stitched = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    aggregated = aggregate_panels(panels)
    if not stitched.empty:
        combined = compute_performance_panel(
            stitched.to_numpy(dtype=np.float64),
            n_concurrent_strategies=len(combos),
        ).to_dict()
        aggregated.update({f"{k}_combined": v for k, v in combined.items()})

    return {
        "symbol": code,
        "mode": mode_value,
        "param_grid": grid,
        "n_folds": len(panels),
        "train_days": int(train_days),
        "oos_days": int(oos_days),
        "window": str(window),
        "oos_panels": oos_panel_dicts,
        "aggregated": aggregated,
        "best_params_by_fold": best_by_fold,
        "parameter_stability": _stability(best_by_fold),
    }
