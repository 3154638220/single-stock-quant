"""Index benchmark fetch specs and daily-bar standardization.

The single-stock trend system keeps this small helper so backtests can compare
strategy performance against broad-market indexes when needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class IndexFetchSpec:
    """指数拉取规格：名称、输出 symbol、AkShare symbol。"""
    name: str
    output_symbol: str
    akshare_symbol: str


DEFAULT_INDEX_SPECS: tuple[IndexFetchSpec, ...] = (
    IndexFetchSpec("csi1000", "000852", "sh000852"),
    IndexFetchSpec("csi2000", "932000", "csi932000"),
)


def parse_index_specs(items: list[str]) -> tuple[IndexFetchSpec, ...]:
    """解析 CLI --index 参数为 IndexFetchSpec 元组。"""
    if not items:
        return DEFAULT_INDEX_SPECS
    specs: list[IndexFetchSpec] = []
    for item in items:
        parts = [x.strip() for x in str(item).split(":")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"--index 需要 name:output_symbol:akshare_symbol，收到: {item!r}")
        name, output_symbol, akshare_symbol = parts
        digits = "".join(ch for ch in output_symbol if ch.isdigit())
        if len(digits) != 6:
            raise ValueError(f"output_symbol 需要 6 位代码，收到: {output_symbol!r}")
        specs.append(IndexFetchSpec(name=name, output_symbol=digits, akshare_symbol=akshare_symbol))
    return tuple(specs)


def standardize_index_daily(raw: pd.DataFrame, spec: IndexFetchSpec) -> pd.DataFrame:
    """将 AkShare/Eastmoney 指数日线标准化为统一 schema。"""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["trade_date", "open", "symbol", "name", "source_symbol"])
    col_map = {
        "日期": "trade_date",
        "date": "trade_date",
        "datetime": "trade_date",
        "开盘": "open",
        "open": "open",
        "收盘": "close",
        "close": "close",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "成交量": "volume",
        "volume": "volume",
        "成交额": "amount",
        "amount": "amount",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns}).copy()
    if "trade_date" not in df.columns or "open" not in df.columns:
        raise ValueError(f"{spec.name} 指数日线缺少 trade_date/open 列: {list(raw.columns)}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    for col in ["close", "high", "low", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = spec.output_symbol.zfill(6)
    df["name"] = spec.name
    df["source_symbol"] = spec.akshare_symbol
    keep = ["trade_date", "open", "symbol", "name", "source_symbol"]
    keep.extend([c for c in ["close", "high", "low", "volume", "amount"] if c in df.columns])
    out = df[keep].dropna(subset=["trade_date", "open"])
    return out.drop_duplicates(["symbol", "trade_date"], keep="last").sort_values(["symbol", "trade_date"])
