"""Unified backtest parameter construction shared across all entry points."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.backtest.transaction_costs import TransactionCostParams, transaction_cost_params_from_mapping


def build_bt_kwargs(cfg: dict[str, Any], *, index_ohlcv: pd.DataFrame | None = None) -> dict[str, Any]:
    """Build the complete kwargs dict for ``run_single_stock_backtest()``.

    All three entry points (single, batch, WFO) must use this function so that the
    same config produces identical backtest behaviour.
    """
    bt_cfg = cfg.get("backtest", {}) or {}
    filt_cfg = cfg.get("signal_filter", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}
    trend_cfg = cfg.get("trend_signal", {}) or {}
    consensus_n = trend_cfg.get("consensus_n_agree")
    tc_cfg = bt_cfg.get("transaction_cost", {}) or {}

    return {
        "cost_bps": float(bt_cfg.get("cost_bps", 15.0)),
        "cost_params": transaction_cost_params_from_mapping(tc_cfg) if tc_cfg else None,
        "initial_capital": float(bt_cfg.get("initial_capital", 100000)),
        "volume_confirm": bool(filt_cfg.get("volume_confirm", False)),
        "volume_lookback": int(filt_cfg.get("volume_lookback", 20)),
        "volume_ratio_min": float(filt_cfg.get("volume_ratio_min", 1.0)),
        "consensus_n_agree": int(consensus_n) if trend_cfg.get("mode") == "consensus" and consensus_n is not None else None,
        "enable_index_filter": bool(risk_cfg.get("enable_index_filter", False)),
        "index_ohlcv": index_ohlcv,
        "benchmark_symbol": str(risk_cfg.get("benchmark_symbol", "510300")),
        "extreme_lookback_days": int(risk_cfg.get("extreme_lookback_days", 10)),
        "extreme_drop_threshold": float(risk_cfg.get("extreme_drop_threshold", 0.05)),
        "risk_off_factor": float(risk_cfg.get("risk_off_factor", 0.0)),
        "stop_loss_pct": float(bt_cfg.get("stop_loss_pct", 0.0)),
        "trailing_stop_pct": float(bt_cfg.get("trailing_stop_pct", 0.0)),
        "atr_stop_multiplier": float(bt_cfg.get("atr_stop_multiplier", 0.0)),
        "atr_stop_period": int(bt_cfg.get("atr_stop_period", 14)),
        "risk_per_trade_pct": float(bt_cfg.get("risk_per_trade_pct", 0.0)),
        "position_size_cap": float(bt_cfg.get("position_size_cap", 1.0)),
        "stop_reentry_enabled": bool(bt_cfg.get("stop_reentry_enabled", False)),
        "stop_reentry_cooldown": int(bt_cfg.get("stop_reentry_cooldown", 3)),
        "stop_reentry_min_run": int(bt_cfg.get("stop_reentry_min_run", 2)),
        "min_quality_score": float(filt_cfg.get("min_quality_score", 0.0)),
        "quality_score_mode": str(filt_cfg.get("quality_score_mode", "hard")),
    }
