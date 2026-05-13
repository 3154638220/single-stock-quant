"""项目根路径与全局配置加载。"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml

_LOG = logging.getLogger(__name__)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _append_unique(paths: list[Path], path: Path) -> None:
    if path not in paths:
        paths.append(path)


def config_path_candidates(config_path: Union[str, Path]) -> list[Path]:
    """Return config lookup paths rooted at the project directory."""
    root = project_root()
    raw = Path(config_path).expanduser()
    candidates: list[Path] = []

    if raw.is_absolute():
        _append_unique(candidates, raw)
        return candidates

    _append_unique(candidates, root / raw)
    return candidates


def resolve_config_path(config_path: Union[str, Path]) -> Path:
    candidates = config_path_candidates(config_path)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is not None:
        path = resolve_config_path(config_path)
        with open(path, "r", encoding="utf-8") as f:
            return _deep_merge(DEFAULT_CONFIG, yaml.safe_load(f) or {})

    root = project_root()
    candidates: list[Path] = []
    env_path = os.environ.get("QUANT_CONFIG", "").strip()
    if env_path:
        candidates.extend(config_path_candidates(env_path))
    candidates.extend(
        [
            root / "config.yaml",
            root / "config.yaml.example",
        ]
    )
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return _deep_merge(DEFAULT_CONFIG, yaml.safe_load(f) or {})
    _LOG.warning("未找到配置文件（config.yaml / config.yaml.example），使用代码内默认参数。")
    return copy.deepcopy(DEFAULT_CONFIG)


def resolve_asof_trade_end(paths: Optional[Dict[str, Any]] = None) -> pd.Timestamp:
    """
    全市场日线截面使用的统一交易日上界。

    ``paths["asof_trade_date"]`` 为非空字符串时解析（如 ``2026-03-27``），
    否则为运行当日 ``normalize()``。
    """
    paths = paths or {}
    raw = paths.get("asof_trade_date")
    if raw is None:
        return pd.Timestamp.now().normalize()
    s = str(raw).strip()
    if not s:
        return pd.Timestamp.now().normalize()
    return pd.Timestamp(s).normalize()


def has_explicit_asof_trade_date(paths: Optional[Dict[str, Any]] = None) -> bool:
    """是否配置了非空的 ``asof_trade_date``（用于输出标签与 CLI 覆盖判断）。"""
    paths = paths or {}
    raw = paths.get("asof_trade_date")
    return bool(raw is not None and str(raw).strip())


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归深合并配置，用户配置只覆盖显式给出的键。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# 默认配置骨架。
DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "duckdb_path": "data/market.duckdb",
        "output_dir": "data/output",
        "logs_dir": "data/logs",
        "stock_name_cache": "data/stock_names.csv",
        "asof_trade_date": "",
    },
    "logging": {
        "format": "text",
    },
    "akshare": {
        "adjust": "qfq",
        "sleep_between_symbols_sec": 0.5,
        "max_fetch_retries": 3,
        "retry_delay_sec": 2.0,
        "request_timeout_sec": 10.0,
        "fetch_workers": 2,
        "daily_source_preference": "sina",
        "daily_allow_fallback": True,
    },
    "database": {
        "table_daily": "a_share_daily",
        "table_audit": "data_fetch_audit",
        "auto_backfill_derived_on_init": True,
        "apply_legacy_migrations": False,
    },
    "quality": {
        "max_calendar_gap_days": 20,
        "null_ratio_max": 0.05,
        "fail_on_ohlc_invalid": True,
        "fail_on_large_gaps": True,
    },
    "trend_signal": {
        "mode": "macd_cross",
        "consensus_n_agree": 2,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "ma_fast": 5,
        "ma_slow": 20,
        "ma_smooth": 3,
        "boll_window": 20,
        "min_run_len": 1,
    },
    "signal_filter": {
        "volume_confirm": False,
        "volume_lookback": 20,
        "volume_ratio_min": 1.0,
    },
    "backtest": {
        "cost_bps": 15.0,
        "initial_capital": 100000,
        "execution": "tplus1_open",
        "stop_loss_pct": 0.0,
        "trailing_stop_pct": 0.0,
        "atr_stop_multiplier": 0.0,
        "atr_stop_period": 14,
        "risk_per_trade_pct": 0.0,
        "position_size_cap": 1.0,
        "stop_reentry_enabled": False,
        "stop_reentry_cooldown": 3,
        "stop_reentry_min_run": 2,
        "transaction_cost": {
            "commission_buy_bps": 2.5,
            "commission_sell_bps": 2.5,
            "slippage_bps_per_side": 2.0,
            "stamp_duty_sell_bps": 5.0,
        },
    },
    "wfo": {
        "param_grid": {
            "macd_fast": [8, 10, 12, 14],
            "macd_slow": [22, 26, 30],
            "macd_signal": [7, 9, 11],
            "min_run_len": [1, 2, 3],
            "stop_loss_pct": [0.05, 0.08, 0.10],
        },
    },
    "risk": {
        "enable_index_filter": False,
        "benchmark_symbol": "510300",
        "extreme_lookback_days": 10,
        "extreme_drop_threshold": 0.05,
        "risk_off_factor": 0.0,
    },
    "notify": {
        "wecom_webhook_url": "",
        "mention_all": False,
    },
}
