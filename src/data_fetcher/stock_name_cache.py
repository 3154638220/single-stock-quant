"""Stock name cache helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.settings import project_root


def _name_column(df: pd.DataFrame) -> pd.Series:
    for col in ["name", "stock_name", "股票名称", "名称"]:
        if col in df.columns:
            names = df[col].fillna("").astype(str).str.strip()
            return names.mask(names.eq("") | names.str.lower().eq("nan"), "UNKNOWN")
    return pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype=object)


def _is_st_name(names: pd.Series) -> pd.Series:
    clean = names.fillna("").astype(str).str.strip().str.upper()
    return clean.str.contains("ST", regex=False)


def _display_symbol(symbol: Any) -> str:
    raw = str(symbol)
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits.zfill(6) if digits else raw


def _normalize_name_cache(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", "name"])
    symbol_col = next((c for c in ["symbol", "code", "代码", "证券代码"] if c in df.columns), "")
    name_col = next((c for c in ["name", "stock_name", "股票名称", "名称", "证券简称"] if c in df.columns), "")
    if not symbol_col or not name_col:
        return pd.DataFrame(columns=["symbol", "name"])
    out = df[[symbol_col, name_col]].rename(columns={symbol_col: "symbol", name_col: "name"}).copy()
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out = out[(out["symbol"].str.len() == 6) & out["name"].ne("") & out["name"].str.lower().ne("nan")]
    return out.drop_duplicates("symbol", keep="last").reset_index(drop=True)


def load_stock_name_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "name"])
    return _normalize_name_cache(pd.read_csv(path, dtype=str))


def resolve_stock_name_cache_path(cfg: dict[str, Any]) -> Path:
    """Return the configured local stock-name cache path."""
    raw = str((cfg.get("paths", {}) or {}).get("stock_name_cache", "data/stock_names.csv")).strip()
    path = Path(raw or "data/stock_names.csv").expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def load_stock_name_map(path: Path) -> dict[str, str]:
    """Load ``symbol -> name`` from a local CSV cache."""
    names = load_stock_name_cache(path)
    if names.empty:
        return {}
    return dict(zip(names["symbol"].astype(str), names["name"].astype(str)))


def resolve_stock_names(symbols: Iterable[Any], cache_path: Path) -> dict[str, str]:
    """Resolve display names for symbols, falling back to the normalized symbol."""
    cache = load_stock_name_map(cache_path)
    out: dict[str, str] = {}
    for item in symbols:
        code = _display_symbol(item)
        name = str(cache.get(code, "")).strip()
        out[code] = name if name else code
    return out


def attach_stock_names(dataset: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    if names.empty:
        if "name" not in out.columns:
            out["name"] = ""
        return out
    names_norm = _normalize_name_cache(names)
    if names_norm.empty:
        if "name" not in out.columns:
            out["name"] = ""
        return out
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    old_name = _name_column(out) if any(c in out.columns for c in ["name", "stock_name", "股票名称", "名称"]) else None
    out = out.drop(columns=["name"], errors="ignore").merge(names_norm, on="symbol", how="left")
    if old_name is not None:
        out["name"] = out["name"].fillna(old_name).replace({"UNKNOWN": ""})
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    return out
