"""Walk-forward optimization for DK trend parameters."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import product
from statistics import mode
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.performance_panel import aggregate_panels, compute_performance_panel
from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.indicators.donchian import compute_donchian_trend
from src.models.meta_label import FEATURE_COLUMNS, LogisticMetaModel, build_training_samples
from src.signals.consensus import compute_consensus_trend
from src.signals.generator import apply_volume_confirmation

DEFAULT_PARAM_GRID: dict[str, list] = {
    "macd_fast": [8, 10, 12, 14],
    "macd_slow": [22, 26, 30],
    "macd_signal": [7, 9, 11],
    "min_run_len": [1, 2, 3],
    "stop_loss_pct": [0.05, 0.08, 0.10],
}

# 2026-05-18 return-improvement grid (4×3×4×2×3×3 = 864 combos).
# It keeps the search in the MACD + exit layer where historical experiments
# found the most reliable return leverage.
# 2026-05-19 cleanup: dk_fade_exit_n removed after diagnostics showed no edge.
EXTENDED_PARAM_GRID: dict[str, list] = {
    "macd_fast": [8, 10, 12, 14],
    "macd_slow": [22, 26, 30],
    "macd_signal": [7, 8, 9, 10],
    "min_run_len": [1, 2],
    "profit_lock_trigger": [0.10, 0.12, 0.15],
    "profit_lock_trailing": [0.04, 0.05, 0.06],
}

FOCUSED_PARAM_GRID: dict[str, list] = EXTENDED_PARAM_GRID

_BT_PARAM_KEYS = {
    "stop_loss_pct", "trailing_stop_pct", "atr_stop_multiplier", "atr_stop_period",
    "atr_trailing_mult", "atr_trailing_min_gain",
    "volume_confirm", "volume_lookback", "volume_ratio_min",
    "risk_per_trade_pct", "position_size_cap",
    "stop_reentry_enabled", "stop_reentry_cooldown", "stop_reentry_min_run",
    "min_quality_score", "quality_score_floor",
    "time_stop_days", "time_stop_min_return",
    "profit_lock_trigger", "profit_lock_trailing",
    "profit_lock_trigger_hq", "profit_lock_trailing_hq", "quality_hq_threshold",
    "market_exit_mode",
    "sector_drop_threshold", "sector_ma_period",
    "volatility_target_ann", "volatility_lookback",
    "drawdown_throttle_enabled",
    "meta_label_threshold", "meta_label_mode",
    "require_above_ma120", "require_positive_rs60", "require_index_trend_bullish",
    "require_weekly_bullish", "weekly_ma_fast", "weekly_ma_slow",
    "volatility_high_vol_multiple", "volatility_high_vol_scale",
    "dk_fade_exit_n", "intrapos_dd_limit",
    "require_price_breakout", "breakout_lookback",
    "require_adx_min", "adx_period",
    "require_pullback_entry", "pullback_wait_days",
    "enable_index_ma20_filter",
}

# All kwargs that run_single_stock_backtest accepts (beyond cost_bps, cost_params, initial_capital).
_VALID_BT_KWARGS = {
    "stock_name", "volume_confirm", "volume_lookback", "volume_ratio_min",
    "consensus_n_agree", "enable_index_filter", "index_ohlcv",
    "benchmark_symbol", "extreme_lookback_days", "extreme_drop_threshold",
    "risk_off_factor", "stop_loss_pct", "trailing_stop_pct",
    "atr_stop_multiplier", "atr_stop_period", "atr_trailing_mult", "atr_trailing_min_gain",
    "risk_per_trade_pct",
    "position_size_cap", "stop_reentry_enabled", "stop_reentry_cooldown",
    "stop_reentry_min_run", "cost_params",
    "min_quality_score", "quality_score_mode", "quality_score_floor",
    "time_stop_days", "time_stop_min_return",
    "profit_lock_trigger", "profit_lock_trailing",
    "profit_lock_trigger_hq", "profit_lock_trailing_hq", "quality_hq_threshold",
    "market_exit_mode",
    "sector_index_ohlcv", "sector_drop_threshold", "sector_ma_period",
    "volatility_target_ann", "volatility_lookback",
    "drawdown_throttle_enabled",
    "meta_model", "meta_label_threshold", "meta_label_mode",
    "require_above_ma120", "require_positive_rs60", "require_index_trend_bullish",
    "require_weekly_bullish", "weekly_ma_fast", "weekly_ma_slow",
    "volatility_high_vol_multiple", "volatility_high_vol_scale",
    "dk_fade_exit_n", "intrapos_dd_limit",
    "require_price_breakout", "breakout_lookback",
    "require_adx_min", "adx_period",
    "require_pullback_entry", "pullback_wait_days",
    "enable_index_ma20_filter",
}


def _param_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def json_safe(value: Any) -> Any:
    """Return a strict-JSON-safe copy with NaN/inf converted to null."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def trade_contribution_metrics(trade_log: pd.DataFrame, total_return: float) -> dict[str, float]:
    """Largest winning trade and its share of total strategy return."""
    if trade_log.empty or "return" not in trade_log:
        return {"largest_trade_return": float("nan"), "largest_trade_contribution": float("nan")}
    returns = pd.to_numeric(trade_log["return"], errors="coerce").dropna()
    winners = returns[returns > 0]
    if winners.empty:
        return {"largest_trade_return": 0.0, "largest_trade_contribution": float("nan")}
    largest = float(winners.max())
    total = float(total_return) if np.isfinite(total_return) else float("nan")
    return {
        "largest_trade_return": largest,
        "largest_trade_contribution": largest / total if total > 0 else float("nan"),
    }


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


def _composite_score(
    res,
    *,
    train_days: int = 504,
    min_trades_per_year: float = 2.0,
    max_trades_per_year: float = 30.0,
    max_drawdown_limit: float = 0.35,
    w_calmar: float = 0.30,
    w_sharpe: float = 0.25,
    w_dd_score: float = 0.25,
    w_total_return: float = 0.10,
    w_trade_freq: float = 0.10,
    reliability_mode: str = "standard",
) -> float:
    """Return-quality composite objective score.

    Returns ``-inf`` when any hard constraint is violated.
    The 2026-05-18 plan relaxes sparse-trade penalties but tightens drawdown,
    and adds total return as a direct secondary objective.
    """
    years = max(train_days / 252.0, 0.25)
    n_per_year = res.n_trades / years

    if n_per_year < min_trades_per_year:
        return -np.inf
    if n_per_year > max_trades_per_year:
        return -np.inf
    if res.max_drawdown > max_drawdown_limit:
        return -np.inf

    sharpe = np.clip(res.sharpe_ratio, -2.0, 3.0) if np.isfinite(res.sharpe_ratio) else -2.0
    calmar = np.clip(res.calmar_ratio, -1.0, 3.0) if np.isfinite(res.calmar_ratio) else -1.0
    total_ret = np.clip(res.total_return, -0.5, 1.0) if np.isfinite(res.total_return) else -0.5
    # Nonlinear MDD penalty: gentle below 20%, steep above 20%
    if np.isfinite(res.max_drawdown):
        if res.max_drawdown > 0.20:
            dd_score = max(0.0, (0.30 - res.max_drawdown) / 0.10)  # 20%→1.0, 30%→0.0
        else:
            dd_score = max(0.0, 1.0 - res.max_drawdown * 3.5)      # steeper gradient: 15%→0.475
    else:
        dd_score = 0.0
    mode = str(reliability_mode).lower()
    effective_w_trade_freq = 0.0 if mode == "quality_first" else float(w_trade_freq)
    trade_freq_score = min(float(n_per_year) / 10.0, 1.0)

    raw_score = (
        w_calmar * calmar
        + w_sharpe * sharpe
        + w_dd_score * dd_score
        + w_total_return * total_ret
        + effective_w_trade_freq * trade_freq_score
    )
    if mode == "quality_first":
        reliability = 1.0
    else:
        reliability = min(1.0, max(float(res.n_trades), 0.0) / 10.0)
        reliability = max(reliability, 0.30)
    return raw_score * reliability if raw_score >= 0 else raw_score / reliability


def bootstrap_sharpe_ci(
    returns: np.ndarray | pd.Series,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for annualised Sharpe ratio.

    Parameters
    ----------
    returns:
        Daily strategy returns (can include NaN for days out of market).
    n_boot:
        Number of bootstrap resamples.
    ci_level:
        Confidence level (default 0.95 for 95% CI).

    Returns
    -------
    dict with keys ``sharpe`` (point estimate), ``ci_lower``, ``ci_upper``,
    ``positive_fraction`` (fraction of bootstrap Sharpes > 0).
    """
    rng = np.random.default_rng(random_state)
    if isinstance(returns, pd.Series):
        vals = returns.dropna().to_numpy(dtype=np.float64)
    else:
        vals = returns[np.isfinite(returns)]
    n = len(vals)
    if n < 10:
        return {"sharpe": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "positive_fraction": float("nan")}

    point_sharpe = float(np.mean(vals) / np.std(vals) * np.sqrt(252)) if np.std(vals) > 0 else 0.0

    boot_sharpes = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        s = np.std(sample)
        boot_sharpes[b] = float(np.mean(sample) / s * np.sqrt(252)) if s > 0 else 0.0

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_sharpes, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_sharpes, 100.0 * (1.0 - alpha)))
    pos_frac = float(np.mean(boot_sharpes > 0))

    return {
        "sharpe": point_sharpe,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "positive_fraction": pos_frac,
    }


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


def _eval_wfo_combo(task: tuple[Any, ...]) -> float:
    (
        code,
        train_df,
        base,
        combo,
        mode_value,
        base_bt,
        cost_bps,
        initial_capital,
        train_days,
        score_min_trades_per_year,
        score_max_trades_per_year,
        score_max_drawdown_limit,
        score_reliability_mode,
    ) = task
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
    return float(
        _composite_score(
            res,
            train_days=train_days,
            min_trades_per_year=float(score_min_trades_per_year),
            max_trades_per_year=float(score_max_trades_per_year),
            max_drawdown_limit=float(score_max_drawdown_limit),
            reliability_mode=str(score_reliability_mode),
        )
    )


def _stability(best_by_fold: list[dict[str, Any]]) -> dict[str, Any]:
    params = [x["params"] for x in best_by_fold]
    out: dict[str, Any] = {}
    if not params:
        return out
    for key in params[0]:
        vals = [p[key] for p in params]
        try:
            param_mode = mode(vals)
        except Exception:
            param_mode = vals[0]
        numeric = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=np.float64)
        out[key] = {
            "mode": param_mode,
            "variance": float(np.var(numeric)) if numeric.size else float("nan"),
        }
    is_scores = np.array([x["is_score"] for x in best_by_fold], dtype=np.float64)
    oos_sharpes = np.array([x["oos_sharpe"] for x in best_by_fold], dtype=np.float64)
    valid = np.isfinite(is_scores) & np.isfinite(oos_sharpes)
    corr_ready = int(valid.sum()) >= 2 and float(np.std(is_scores[valid])) > 0 and float(np.std(oos_sharpes[valid])) > 0
    out["is_oos_score_corr"] = (
        float(np.corrcoef(is_scores[valid], oos_sharpes[valid])[0, 1]) if corr_ready else float("nan")
    )
    return out


def _numeric_params(combo: dict[str, Any]) -> dict[str, float]:
    """Extract numeric parameters from a combo dict for distance computation."""
    out: dict[str, float] = {}
    for k, v in combo.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _normalize_param_ranges(combos: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Compute (min, max) ranges for each numeric parameter across all combos."""
    numeric_keys: set[str] = set()
    for c in combos:
        for k in c:
            try:
                float(c[k])
                numeric_keys.add(k)
            except (TypeError, ValueError):
                pass
    ranges: dict[str, tuple[float, float]] = {}
    for k in sorted(numeric_keys):
        vals = [float(c[k]) for c in combos]
        lo, hi = min(vals), max(vals)
        ranges[k] = (lo, hi if hi > lo else lo + 1.0)
    return ranges


def _param_distance(
    a: dict[str, float],
    b: dict[str, float],
    ranges: dict[str, tuple[float, float]],
) -> float:
    """Normalised Euclidean distance between two parameter sets."""
    keys = set(a) & set(b) & set(ranges)
    if not keys:
        return 0.0
    ssq = 0.0
    for k in keys:
        lo, hi = ranges[k]
        span = hi - lo
        ssq += ((a[k] - b[k]) / span) ** 2
    return float(np.sqrt(ssq / len(keys)))


def _select_platform(
    combos: list[dict[str, Any]],
    scores: list[float],
    *,
    top_fraction: float = 0.20,
    isolation_penalty: float = 0.10,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Select a parameter *platform* instead of a single isolated peak.

    Returns (best_platform_combo, platform_score, platform_info).
    platform_info includes ``is_isolated``, ``neighbourhood_density``,
    ``original_best``, and ``n_top_region``.
    """
    n = len(combos)
    if n <= 1:
        return combos[0], float(scores[0] if scores else -np.inf), {}

    n_top = max(2, int(np.ceil(n * top_fraction)))
    sorted_idx = np.argsort([float(s) if np.isfinite(s) else -np.inf for s in scores])[::-1]
    top_idx = sorted_idx[:n_top]
    top_scores = [float(scores[i]) for i in top_idx]

    ranges = _normalize_param_ranges(combos)
    top_numeric = [_numeric_params(combos[i]) for i in top_idx]

    best_idx_in_top = 0
    best_platform_score = -np.inf
    best_density = 0.0

    for j, (idx, num) in enumerate(zip(top_idx, top_numeric)):
        distances = []
        for other_num in top_numeric:
            distances.append(_param_distance(num, other_num, ranges))
        distances.sort()
        # Density: fraction of top region within "close" distance (median of pairwise distances)
        median_dist = float(np.median(distances)) if len(distances) > 1 else 0.0
        threshold = max(median_dist, 0.01)
        density = sum(1.0 for d in distances if d <= threshold) / len(distances)
        platform_score = top_scores[j] * (1.0 - isolation_penalty * (1.0 - density))
        if platform_score > best_platform_score:
            best_platform_score = platform_score
            best_idx_in_top = j
            best_density = density

    selected_idx = top_idx[best_idx_in_top]
    original_best_idx = top_idx[0]
    is_isolated = best_idx_in_top != 0

    return (
        combos[selected_idx],
        best_platform_score,
        {
            "is_isolated": is_isolated,
            "original_best": combos[original_best_idx] if is_isolated else combos[selected_idx],
            "neighbourhood_density": best_density,
            "n_top_region": n_top,
            "score_original_best": top_scores[0],
            "score_platform": best_platform_score,
        },
    )


def _parameter_drift(best_by_fold: list[dict[str, Any]]) -> dict[str, Any]:
    """Quantify how much the selected parameters shift from fold to fold."""
    if len(best_by_fold) < 2:
        return {"n_folds": len(best_by_fold), "mean_drift": float("nan"), "max_drift": float("nan")}

    params = [x["params"] for x in best_by_fold]
    ranges = _normalize_param_ranges(params)
    drifts = []
    for i in range(1, len(params)):
        d = _param_distance(
            _numeric_params(params[i]),
            _numeric_params(params[i - 1]),
            ranges,
        )
        drifts.append(d)
    return {
        "n_folds": len(best_by_fold),
        "mean_drift": float(np.mean(drifts)),
        "median_drift": float(np.median(drifts)),
        "max_drift": float(np.max(drifts)),
        "drift_by_fold": drifts,
    }


def _select_stable_params(
    fold_results: list[dict[str, Any]],
    *,
    min_folds: int = 5,
    top_n: int = 5,
) -> dict[str, Any]:
    """Select parameters by cross-fold IS stability instead of peak score."""
    valid_rows: list[tuple[dict[str, Any], float]] = []
    for row in fold_results:
        params = row.get("params")
        if not isinstance(params, dict):
            continue
        raw_score = row.get("is_score", row.get("platform_score", float("nan")))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            valid_rows.append((dict(params), score))

    if len(valid_rows) < min_folds:
        return {
            "params": {},
            "n_folds": len(valid_rows),
            "used": False,
            "reason": f"need at least {min_folds} folds",
        }

    grouped: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    for params, score in valid_rows:
        key = tuple(sorted(params.items()))
        entry = grouped.setdefault(key, {"params": params, "scores": []})
        entry["scores"].append(score)

    ranked: list[dict[str, Any]] = []
    for entry in grouped.values():
        scores = np.array(entry["scores"], dtype=np.float64)
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        ranked.append(
            {
                "params": dict(entry["params"]),
                "n": int(len(scores)),
                "is_score_mean": mean,
                "is_score_std": std,
                "stability_score": float(mean / (std + 0.1)),
            }
        )

    ranked.sort(key=lambda x: (x["stability_score"], x["is_score_mean"], x["n"]), reverse=True)
    top = ranked[: max(1, int(top_n))]
    keys = sorted({k for row in top for k in row["params"]})
    centre: dict[str, Any] = {}
    for key in keys:
        vals = [row["params"][key] for row in top if key in row["params"]]
        if not vals:
            continue
        if all(isinstance(v, bool) for v in vals):
            centre[key] = max(set(vals), key=vals.count)
            continue
        try:
            numeric_vals = np.array(vals, dtype=np.float64)
            value = float(np.median(numeric_vals))
            if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                value = int(round(value))
            centre[key] = value
        except (TypeError, ValueError):
            centre[key] = max(set(vals), key=vals.count)

    return {
        "params": centre,
        "n_folds": len(valid_rows),
        "used": True,
        "top_region": top,
    }


def _build_heatmap_data(
    combos: list[dict[str, Any]],
    scores: list[float],
    oos_sharpes: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build 2D heatmap data for every pair of numeric parameters.

    Returns a list of dicts, each with keys ``x_param``, ``y_param``,
    ``x_vals``, ``y_vals``, ``z_matrix`` (IS scores), and optionally
    ``z_oos_matrix`` (OOS Sharpe values).
    """
    numeric_keys = sorted(_normalize_param_ranges(combos))
    if len(numeric_keys) < 2:
        return []

    heatmaps = []
    for xi, xk in enumerate(numeric_keys):
        for yk in numeric_keys[xi + 1:]:
            # Find unique sorted values for each param
            x_vals = sorted(set(float(c[xk]) for c in combos))
            y_vals = sorted(set(float(c[yk]) for c in combos))
            nx, ny = len(x_vals), len(y_vals)
            z = np.full((ny, nx), np.nan)
            z_oos = np.full((ny, nx), np.nan) if oos_sharpes else None
            # Build lookup: (x_val, y_val) -> average score (marginalising over other params)
            accum: dict[tuple[float, float], list[tuple[float, float | None]]] = {}
            for c, s, oos in zip(combos, scores, oos_sharpes or [None] * len(combos)):
                key = (float(c[xk]), float(c[yk]))
                accum.setdefault(key, []).append((float(s), float(oos) if oos is not None and np.isfinite(oos) else None))
            for (xv, yv), entries in accum.items():
                xi_idx = x_vals.index(xv)
                yi_idx = y_vals.index(yv)
                z[yi_idx, xi_idx] = float(np.mean([e[0] for e in entries if np.isfinite(e[0])]))
                if z_oos is not None:
                    oos_vals = [e[1] for e in entries if e[1] is not None]
                    z_oos[yi_idx, xi_idx] = float(np.mean(oos_vals)) if oos_vals else float("nan")
            hm: dict[str, Any] = {
                "x_param": xk,
                "y_param": yk,
                "x_vals": x_vals,
                "y_vals": y_vals,
                "z_is_score": z.tolist(),
            }
            if z_oos is not None:
                hm["z_oos_sharpe"] = z_oos.tolist()
            heatmaps.append(hm)
    return heatmaps


def _buy_signal_dates(df: pd.DataFrame, params: DKTrendParams, bt_kwargs: dict[str, Any]) -> pd.DatetimeIndex:
    """Compute BUY signal dates for one train window using the selected signal stack."""
    volume_confirm = bool(bt_kwargs.get("volume_confirm", False))
    volume_lookback = int(bt_kwargs.get("volume_lookback", 20))
    volume_ratio_min = float(bt_kwargs.get("volume_ratio_min", 1.0))
    consensus_n = bt_kwargs.get("consensus_n_agree")
    if consensus_n is not None and int(consensus_n) > 1:
        trend = compute_consensus_trend(
            df,
            base_params=params,
            n_agree=int(consensus_n),
            volume_confirm=volume_confirm,
            volume_lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    elif params.mode == TrendMode.DONCHIAN_BREAKOUT:
        trend = compute_donchian_trend(
            df,
            entry_window=params.donchian_entry_window,
            exit_window=params.donchian_exit_window,
            min_run_len=params.min_run_len,
        ).reset_index(drop=True)
        trend = apply_volume_confirmation(
            trend,
            enabled=volume_confirm,
            lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    else:
        trend = compute_dktrend(df, params).reset_index(drop=True)
        trend = apply_volume_confirmation(
            trend,
            enabled=volume_confirm,
            lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    dates = pd.to_datetime(df["trade_date"]).dt.normalize().reset_index(drop=True)
    mask = trend["dk_signal"].astype(str).eq("buy")
    return pd.DatetimeIndex(dates[mask])


def _fit_meta_model_for_fold(
    train_df: pd.DataFrame,
    params: DKTrendParams,
    bt_kwargs: dict[str, Any],
    *,
    min_samples: int = 10,
    l2_penalty: float = 0.1,
    label_type: str = "profit_aware",
    max_drawdown_threshold: float = 0.08,
    model_type: str = "logistic",
    use_daily_samples: bool = False,
):
    """Train a fold-local meta-label model, returning None when samples are weak.

    Parameters
    ----------
    model_type:
        ``"logistic"`` for LogisticMetaModel or ``"gbm"`` for GBMMetaModel.
    use_daily_samples:
        If True, use all DK red days as training samples (daily-level).
        If False, use only BUY signal days (signal-level, default).
    """
    # Compute DK trend state for signal-context features
    dk_trend_state = _compute_dk_state(train_df, params, bt_kwargs)

    if use_daily_samples:
        from src.models.meta_label import build_daily_labels
        trend_full = compute_dktrend(train_df, params).reset_index(drop=True)
        X, y, _ = build_daily_labels(
            train_df,
            trend_full,
            index_ohlcv=bt_kwargs.get("index_ohlcv"),
            forward_days=10,
            label_type=label_type,
            dk_trend_state=dk_trend_state,
        )
    else:
        signal_dates = _buy_signal_dates(train_df, params, bt_kwargs)
        if len(signal_dates) < min_samples:
            return None
        X, y, _ = build_training_samples(
            train_df,
            signal_dates,
            index_ohlcv=bt_kwargs.get("index_ohlcv"),
            label_type=label_type,
            max_drawdown_threshold=max_drawdown_threshold,
            dk_trend_state=dk_trend_state,
        )

    effective_min_samples = max(int(min_samples), 15) if model_type == "gbm" else int(min_samples)
    if len(X) < effective_min_samples or len(np.unique(y)) < 2:
        return None

    if model_type == "gbm":
        from src.models.meta_label_gbm import GBMMetaModel
        model = GBMMetaModel(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=max(5, len(X) // 10),
        )
    else:
        model = LogisticMetaModel(l2_penalty=l2_penalty)

    model.fit(X, y, feature_names=FEATURE_COLUMNS)
    return model


def _compute_dk_state(
    df: pd.DataFrame,
    params: DKTrendParams,
    bt_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """Compute DK trend state with run_len for signal-context features."""
    volume_confirm = bool(bt_kwargs.get("volume_confirm", False))
    volume_lookback = int(bt_kwargs.get("volume_lookback", 20))
    volume_ratio_min = float(bt_kwargs.get("volume_ratio_min", 1.0))
    if params.mode == TrendMode.DONCHIAN_BREAKOUT:
        trend = compute_donchian_trend(
            df,
            entry_window=params.donchian_entry_window,
            exit_window=params.donchian_exit_window,
            min_run_len=params.min_run_len,
        ).reset_index(drop=True)
        trend = apply_volume_confirmation(
            trend, enabled=volume_confirm, lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    else:
        trend = compute_dktrend(df, params).reset_index(drop=True)
        trend = apply_volume_confirmation(
            trend, enabled=volume_confirm, lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    keep_cols = ["trade_date", "dk_color", "dk_run_len"]
    if "dk_value" in trend.columns:
        keep_cols.append("dk_value")
    return trend[keep_cols].copy()


def _oos_trend_with_warmup(
    train_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    params: DKTrendParams,
    bt_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """Compute OOS trend using train rows as indicator/state warmup."""
    context = pd.concat([train_df, oos_df], ignore_index=True)
    volume_confirm = bool(bt_kwargs.get("volume_confirm", False))
    volume_lookback = int(bt_kwargs.get("volume_lookback", 20))
    volume_ratio_min = float(bt_kwargs.get("volume_ratio_min", 1.0))
    consensus_n_agree = bt_kwargs.get("consensus_n_agree")

    if consensus_n_agree is not None and int(consensus_n_agree) > 1:
        trend = compute_consensus_trend(
            context,
            base_params=params,
            n_agree=int(consensus_n_agree),
            volume_confirm=volume_confirm,
            volume_lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    elif params.mode == TrendMode.DONCHIAN_BREAKOUT:
        trend = compute_donchian_trend(
            context,
            entry_window=params.donchian_entry_window,
            exit_window=params.donchian_exit_window,
            min_run_len=params.min_run_len,
        ).reset_index(drop=True)
        trend = apply_volume_confirmation(
            trend,
            enabled=volume_confirm,
            lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)
    else:
        trend = compute_dktrend(context, params).reset_index(drop=True)
        trend = apply_volume_confirmation(
            trend,
            enabled=volume_confirm,
            lookback=volume_lookback,
            volume_ratio_min=volume_ratio_min,
        ).reset_index(drop=True)

    return trend.iloc[-len(oos_df):].reset_index(drop=True)


def run_walk_forward_optimization(
    symbol: str,
    ohlcv: pd.DataFrame,
    *,
    base_params: DKTrendParams | None = None,
    param_grid: dict[str, list[Any]] | None = None,
    train_days: int = 504,
    oos_days: int = 126,
    step_days: int | None = None,
    mode: TrendMode | str = TrendMode.MACD_CROSS,
    window: str = "rolling",
    cost_bps: float = 15.0,
    initial_capital: float = 100_000.0,
    bt_kwargs: dict[str, Any] | None = None,
    enable_meta_label: bool = False,
    meta_label_threshold: float = 0.50,
    meta_label_mode: str = "hard",
    meta_label_min_samples: int = 10,
    meta_label_type: str = "profit_aware",
    meta_model_type: str = "logistic",
    meta_use_daily_samples: bool = False,
    stability_weighting: bool = False,
    score_min_trades_per_year: float = 2.0,
    score_max_trades_per_year: float = 30.0,
    score_max_drawdown_limit: float = 0.35,
    score_reliability_mode: str = "standard",
    n_jobs: int = 1,
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
    step = int(step_days if step_days is not None else oos_days)
    if step <= 0:
        raise ValueError("step_days must be positive")

    base_bt = dict(bt_kwargs or {})
    panels = []
    oos_returns = []
    oos_panel_dicts = []
    best_by_fold = []
    platform_by_fold: list[dict[str, Any]] = []
    all_is_scores: list[float] = []
    all_oos_sharpes: list[float] = []
    start = 0
    fold = 0
    while start + train_days + oos_days <= len(df):
        train_start = 0 if str(window).lower() == "expanding" else start
        train_end = start + train_days
        oos_start = train_end
        oos_end = train_end + oos_days
        train_df = df.iloc[train_start:train_end].copy()
        oos_df = df.iloc[oos_start:oos_end].copy()

        if int(n_jobs) > 1 and len(combos) > 1:
            tasks = [
                (
                    code,
                    train_df,
                    base,
                    combo,
                    mode_value,
                    base_bt,
                    cost_bps,
                    initial_capital,
                    train_days,
                    score_min_trades_per_year,
                    score_max_trades_per_year,
                    score_max_drawdown_limit,
                    score_reliability_mode,
                )
                for combo in combos
            ]
            with ProcessPoolExecutor(max_workers=int(n_jobs)) as executor:
                is_scores = list(executor.map(_eval_wfo_combo, tasks))
        else:
            is_scores = [
                _eval_wfo_combo(
                    (
                        code,
                        train_df,
                        base,
                        combo,
                        mode_value,
                        base_bt,
                        cost_bps,
                        initial_capital,
                        train_days,
                        score_min_trades_per_year,
                        score_max_trades_per_year,
                        score_max_drawdown_limit,
                        score_reliability_mode,
                    )
                )
                for combo in combos
            ]

        # Platform selection (top 20% region)
        best_platform, platform_score, platform_info = _select_platform(
            combos, is_scores, top_fraction=0.20, isolation_penalty=0.10,
        )

        # Also track the single best for backward compatibility
        best_idx = int(np.argmax([float(s) if np.isfinite(s) else -np.inf for s in is_scores]))
        best_params = combos[best_idx]
        best_score = float(is_scores[best_idx])
        best_bt = _params_with(base, best_params, mode_value)[1]

        selected, sel_bt = _params_with(base, best_platform, mode_value)
        bt_final = {**base_bt, **best_bt}
        # Merge platform's bt overrides too
        _, platform_bt = _params_with(base, best_platform, mode_value)
        bt_final.update(platform_bt)

        meta_model = (
            _fit_meta_model_for_fold(
                train_df,
                selected,
                bt_final,
                min_samples=int(meta_label_min_samples),
                label_type=str(meta_label_type),
                model_type=str(meta_model_type),
                use_daily_samples=bool(meta_use_daily_samples),
            )
            if enable_meta_label
            else None
        )
        if enable_meta_label:
            bt_final["meta_model"] = meta_model
            bt_final["meta_label_threshold"] = float(meta_label_threshold)
            bt_final["meta_label_mode"] = str(meta_label_mode)

        oos_trend = _oos_trend_with_warmup(train_df, oos_df, selected, bt_final)
        oos_res = run_single_stock_backtest(
            code,
            oos_df,
            selected,
            cost_bps=cost_bps,
            initial_capital=initial_capital,
            trend_override=oos_trend,
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
                "total_return": float(oos_res.total_return),
                "annualized_return": float(oos_res.annualized_return),
                "max_drawdown": float(oos_res.max_drawdown),
                "calmar_ratio": float(oos_res.calmar_ratio),
                "n_trades": int(oos_res.n_trades),
                **trade_contribution_metrics(oos_res.trade_log, oos_res.total_return),
            }
        )
        oos_panel_dicts.append(oos_panel)
        best_by_fold.append(
            {
                "fold": fold,
                "params": dict(best_params),
                "is_score": best_score if np.isfinite(best_score) else float("nan"),
                "oos_sharpe": panel.sharpe_ratio,
            }
        )
        platform_by_fold.append({
            "fold": fold,
            "params": dict(best_platform),
            "platform_score": platform_score,
            "oos_sharpe": panel.sharpe_ratio,
            "meta_label_trained": meta_model is not None,
            "platform_info": platform_info,
        })
        all_is_scores.extend(is_scores)
        all_oos_sharpes.append(panel.sharpe_ratio)
        fold += 1
        start += step

    stitched = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    aggregated = aggregate_panels(panels)
    if not stitched.empty:
        combined = compute_performance_panel(
            stitched.to_numpy(dtype=np.float64),
            n_concurrent_strategies=len(combos),
        ).to_dict()
        aggregated.update({f"{k}_combined": v for k, v in combined.items()})
        # S6.3 — Bootstrap CI on stitched OOS Sharpe
        boot_ci = bootstrap_sharpe_ci(stitched.to_numpy(dtype=np.float64))
        aggregated["sharpe_bootstrap_ci"] = boot_ci

    heatmaps = _build_heatmap_data(combos, all_is_scores, all_oos_sharpes if all_oos_sharpes else None)

    stable_parameter_selection = (
        _select_stable_params(platform_by_fold, min_folds=5)
        if stability_weighting
        else {"params": {}, "n_folds": len(platform_by_fold), "used": False, "reason": "disabled"}
    )

    return {
        "symbol": code,
        "mode": mode_value,
        "param_grid": grid,
        "n_folds": len(panels),
        "train_days": int(train_days),
        "oos_days": int(oos_days),
        "step_days": int(step),
        "window": str(window),
        "enable_meta_label": bool(enable_meta_label),
        "meta_label_mode": str(meta_label_mode) if enable_meta_label else "off",
        "meta_label_threshold": float(meta_label_threshold),
        "stability_weighting": bool(stability_weighting),
        "score_reliability_mode": str(score_reliability_mode),
        "n_jobs": int(n_jobs),
        "oos_panels": oos_panel_dicts,
        "aggregated": aggregated,
        "best_params_by_fold": best_by_fold,
        "platform_by_fold": platform_by_fold,
        "stable_parameter_selection": stable_parameter_selection,
        "parameter_stability": _stability(best_by_fold),
        "parameter_drift": _parameter_drift([x for x in platform_by_fold]),
        "heatmaps": heatmaps,
    }


def run_nested_walk_forward_optimization(
    symbol: str,
    ohlcv: pd.DataFrame,
    *,
    base_params: DKTrendParams | None = None,
    param_grid: dict[str, list[Any]] | None = None,
    outer_train_days: int = 756,
    outer_oos_days: int = 252,
    outer_step_days: int | None = None,
    inner_train_days: int = 504,
    inner_oos_days: int = 126,
    inner_step_days: int | None = None,
    mode: TrendMode | str = TrendMode.MACD_CROSS,
    window: str = "rolling",
    cost_bps: float = 15.0,
    initial_capital: float = 100_000.0,
    bt_kwargs: dict[str, Any] | None = None,
    enable_meta_label: bool = False,
    meta_label_threshold: float = 0.50,
    meta_label_mode: str = "hard",
    meta_label_min_samples: int = 10,
    meta_label_type: str = "profit_aware",
    meta_model_type: str = "logistic",
    meta_use_daily_samples: bool = False,
    stability_weighting: bool = True,
    score_min_trades_per_year: float = 2.0,
    score_max_trades_per_year: float = 30.0,
    score_max_drawdown_limit: float = 0.35,
    score_reliability_mode: str = "standard",
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Nested walk-forward optimisation.

    Outer splits provide truly unseen OOS evaluation; inner WFO runs on each
    outer-train block to select parameters without peeking at outer-OOS data.

    Returns a dict with ``outer_folds`` (list of per-fold results), ``aggregated``
    (stitched OOS metrics), and ``inner_wfo_summaries`` (parameter stability
    across outer folds).
    """
    code = str(symbol).strip().zfill(6)
    df = ohlcv.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize() if "trade_date" in df else pd.to_datetime(df.index).normalize()
    df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)

    min_required = outer_train_days + outer_oos_days
    if len(df) < min_required:
        raise ValueError(f"not enough rows: need at least {min_required}, got {len(df)}")

    mode_value = mode.value if isinstance(mode, TrendMode) else str(mode)
    base = base_params or DKTrendParams(mode=mode_value)
    grid = normalize_param_grid(param_grid)
    combos = _param_combinations(grid)
    if not combos:
        raise ValueError("param_grid is empty")
    outer_step = int(outer_step_days if outer_step_days is not None else outer_oos_days)
    if outer_step <= 0:
        raise ValueError("outer_step_days must be positive")

    base_bt = dict(bt_kwargs or {})

    oos_returns: list[pd.Series] = []
    outer_fold_results: list[dict[str, Any]] = []
    inner_wfo_summaries: list[dict[str, Any]] = []
    all_selected_params: list[dict[str, Any]] = []

    start = 0
    fold = 0
    while start + outer_train_days + outer_oos_days <= len(df):
        outer_train_start = start
        outer_train_end = start + outer_train_days
        outer_oos_start = outer_train_end
        outer_oos_end = outer_train_end + outer_oos_days

        outer_train_df = df.iloc[outer_train_start:outer_train_end].copy()
        outer_oos_df = df.iloc[outer_oos_start:outer_oos_end].copy()

        # --- Inner WFO on outer_train to select best params ---
        inner_result = run_walk_forward_optimization(
            code,
            outer_train_df,
            base_params=base,
            param_grid=grid,
            train_days=inner_train_days,
            oos_days=inner_oos_days,
            step_days=inner_step_days,
            mode=mode_value,
            window=window,
            cost_bps=cost_bps,
            initial_capital=initial_capital,
            bt_kwargs=base_bt,
            enable_meta_label=enable_meta_label,
            meta_label_threshold=meta_label_threshold,
            meta_label_mode=meta_label_mode,
            meta_label_min_samples=meta_label_min_samples,
            meta_label_type=str(meta_label_type),
            meta_model_type=str(meta_model_type),
            meta_use_daily_samples=bool(meta_use_daily_samples),
            stability_weighting=stability_weighting,
            score_min_trades_per_year=float(score_min_trades_per_year),
            score_max_trades_per_year=float(score_max_trades_per_year),
            score_max_drawdown_limit=float(score_max_drawdown_limit),
            score_reliability_mode=str(score_reliability_mode),
            n_jobs=int(n_jobs),
        )

        # Prefer a cross-fold stable parameter centre; fall back to voting when
        # the inner WFO has too few folds for a stability estimate.
        platform_by_fold = inner_result.get("platform_by_fold", [])
        best_by_fold = inner_result.get("best_params_by_fold", [])
        stable_selection = (
            _select_stable_params(platform_by_fold, min_folds=5)
            if stability_weighting
            else {"params": {}, "n_folds": len(platform_by_fold), "used": False, "reason": "disabled"}
        )

        if stable_selection.get("used"):
            selected_params = stable_selection["params"]
        elif platform_by_fold:
            # Use the most frequent platform-selected params across inner folds
            param_votes: dict[str, list[dict[str, Any]]] = {}
            for pf in platform_by_fold:
                key = str(sorted(pf["params"].items()))
                param_votes.setdefault(key, []).append(pf["params"])
            best_key = max(param_votes, key=lambda k: len(param_votes[k]))
            selected_params = param_votes[best_key][0]
        elif best_by_fold:
            stable_selection = _select_stable_params(best_by_fold, min_folds=5)
            selected_params = stable_selection["params"] if stable_selection.get("used") else best_by_fold[-1]["params"]
        else:
            stable_selection = {"params": {}, "n_folds": 0, "used": False, "reason": "no inner folds"}
            selected_params = combos[0]

        # --- Evaluate selected params on outer OOS ---
        params, sel_bt = _params_with(base, selected_params, mode_value)
        bt_final = {**base_bt, **sel_bt}
        meta_model = (
            _fit_meta_model_for_fold(
                outer_train_df,
                params,
                bt_final,
                min_samples=int(meta_label_min_samples),
                label_type=str(meta_label_type),
                model_type=str(meta_model_type),
                use_daily_samples=bool(meta_use_daily_samples),
            )
            if enable_meta_label
            else None
        )
        if enable_meta_label:
            bt_final["meta_model"] = meta_model
            bt_final["meta_label_threshold"] = float(meta_label_threshold)
            bt_final["meta_label_mode"] = str(meta_label_mode)
        oos_trend = _oos_trend_with_warmup(outer_train_df, outer_oos_df, params, bt_final)
        oos_res = run_single_stock_backtest(
            code,
            outer_oos_df,
            params,
            cost_bps=cost_bps,
            initial_capital=initial_capital,
            trend_override=oos_trend,
            **{k: v for k, v in bt_final.items() if k in _VALID_BT_KWARGS},
        )

        oos_returns.append(oos_res.daily_returns)

        outer_fold_results.append({
            "fold": fold,
            "outer_train_start": pd.Timestamp(outer_train_df["trade_date"].iloc[0]).date().isoformat(),
            "outer_train_end": pd.Timestamp(outer_train_df["trade_date"].iloc[-1]).date().isoformat(),
            "outer_oos_start": pd.Timestamp(outer_oos_df["trade_date"].iloc[0]).date().isoformat(),
            "outer_oos_end": pd.Timestamp(outer_oos_df["trade_date"].iloc[-1]).date().isoformat(),
            "selected_params": dict(selected_params),
            "meta_label_trained": meta_model is not None,
            "oos_total_return": oos_res.total_return,
            "oos_annualized_return": oos_res.annualized_return,
            "oos_sharpe": oos_res.sharpe_ratio,
            "oos_calmar": oos_res.calmar_ratio,
            "oos_max_drawdown": oos_res.max_drawdown,
            "oos_n_trades": oos_res.n_trades,
            **{f"oos_{k}": v for k, v in trade_contribution_metrics(oos_res.trade_log, oos_res.total_return).items()},
        })

        inner_wfo_summaries.append({
            "fold": fold,
            "inner_n_folds": inner_result["n_folds"],
            "inner_aggregated": inner_result["aggregated"],
            "inner_stability": inner_result.get("parameter_stability", {}),
            "stable_selection": stable_selection,
            "selected_params": dict(selected_params),
        })

        all_selected_params.append(dict(selected_params))
        fold += 1
        start += outer_step

    stitched = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    aggregated = {}
    if not stitched.empty:
        combined = compute_performance_panel(
            stitched.to_numpy(dtype=np.float64),
            n_concurrent_strategies=len(combos),
        ).to_dict()
        aggregated.update(combined)
        # S6.3 — Bootstrap CI on stitched OOS Sharpe
        boot_ci = bootstrap_sharpe_ci(stitched.to_numpy(dtype=np.float64))
        aggregated["sharpe_bootstrap_ci"] = boot_ci

    return {
        "symbol": code,
        "mode": mode_value,
        "param_grid": grid,
        "n_outer_folds": len(outer_fold_results),
        "outer_train_days": int(outer_train_days),
        "outer_oos_days": int(outer_oos_days),
        "outer_step_days": int(outer_step),
        "inner_train_days": int(inner_train_days),
        "inner_oos_days": int(inner_oos_days),
        "inner_step_days": int(inner_step_days if inner_step_days is not None else inner_oos_days),
        "window": str(window),
        "enable_meta_label": bool(enable_meta_label),
        "meta_label_mode": str(meta_label_mode) if enable_meta_label else "off",
        "meta_label_threshold": float(meta_label_threshold),
        "stability_weighting": bool(stability_weighting),
        "score_reliability_mode": str(score_reliability_mode),
        "n_jobs": int(n_jobs),
        "outer_folds": outer_fold_results,
        "aggregated": aggregated,
        "inner_wfo_summaries": inner_wfo_summaries,
        "stable_params_by_outer_fold": [s.get("stable_selection", {}) for s in inner_wfo_summaries],
        "parameter_stability": _stability([{"params": p, "is_score": float("nan"), "oos_sharpe": float("nan")} for p in all_selected_params]),
        "parameter_drift": _parameter_drift([{"params": p} for p in all_selected_params]),
    }
