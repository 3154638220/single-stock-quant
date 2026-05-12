"""Stock name cache helpers."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd


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


def ensure_stock_name_cache(
    cache_path: Path,
    *,
    force: bool = False,
    max_age_days: int = 30,
    config_path: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> None:
    """自动维护股票名称缓存，超期或缺失时拉取 fetch_stock_names.py。"""
    need_fetch = force
    if not need_fetch and cache_path.exists():
        age_sec = time.time() - cache_path.stat().st_mtime
        if age_sec > max_age_days * 86400:
            print(f"[m7] 股票名称缓存已过期 ({age_sec / 86400:.0f}d)，自动刷新...")
            need_fetch = True
    if not need_fetch and not cache_path.exists():
        print("[m7] 股票名称缓存不存在，自动拉取...")
        need_fetch = True

    if not need_fetch:
        print(f"[m7] 股票名称缓存有效: {cache_path} (年龄={max(0, (time.time() - cache_path.stat().st_mtime)) / 86400:.0f}d)")
        return

    root = project_root or cache_path.parent.parent
    fetch_script = root / "scripts" / "fetch_stock_names.py"
    cmd = [
        sys.executable, str(fetch_script),
        "--out", str(cache_path),
        "--timeout-sec", "90",
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    print(f"[m7] 执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[m7] 警告: 股票名称拉取失败 (exit={result.returncode})")
        if result.stderr.strip():
            print(f"[m7] stderr: {result.stderr.strip()[-500:]}")
    else:
        print(f"[m7] 股票名称缓存已更新: {cache_path}")


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
