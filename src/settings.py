"""项目根路径与全局配置加载。"""

from __future__ import annotations

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
            return yaml.safe_load(f) or {}

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
                return yaml.safe_load(f) or {}
    _LOG.warning("未找到配置文件（config.yaml / config.yaml.example），使用代码内默认参数。")
    return {}


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
    """递归深合并：override 中的值完全覆盖 base 中对应键。

    对于嵌套字典，整表替换（不逐键合并），确保 composite_extended 等配置节的
    覆盖语义为"全量替换"而非"增量补丁"。
    """
    result = {**base}
    for key, value in override.items():
        result[key] = value
    return result


# 默认配置骨架。
DEFAULT_CONFIG: Dict[str, Any] = {
    "trend_signal": {
        "mode": "macd_cross",
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "ma_fast": 5,
        "ma_slow": 20,
        "ma_smooth": 3,
        "boll_window": 20,
    },
    "backtest": {
        "cost_bps": 15.0,
        "initial_capital": 100000,
        "execution": "tplus1_open",
    },
}
