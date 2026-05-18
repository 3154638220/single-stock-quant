"""AkShare A 股日线拉取（仅 A 股，与 README 技术栈一致）。"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import pandas as pd

try:  # 离线研究脚本只读 DuckDB 时，不应被在线抓取依赖阻断导入。
    import akshare as ak
except ModuleNotFoundError:  # pragma: no cover - 取决于运行环境是否安装 akshare
    class _MissingAkShare:
        def _missing(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "缺少 akshare 依赖；在线抓取日线/股票池前请先安装 akshare，"
                "或使用本地 DuckDB / universe cache。"
            )

        stock_zh_a_spot_em = _missing
        stock_info_a_code_name = _missing
        stock_zh_a_daily = _missing
        stock_zh_a_hist = _missing
        fund_etf_hist_em = _missing

    ak = _MissingAkShare()

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - 取决于运行环境是否安装 duckdb
    duckdb = None  # type: ignore[assignment]

from src.settings import load_config, project_root

from .akshare_resilience import call_with_timeout, install_akshare_requests_resilience

_LOG = logging.getLogger(__name__)


def _require_akshare():
    return ak


def _standardize_daily_df(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """将 AkShare 返回的中文列名规范为英文，便于落库与张量流水线。"""
    col_map = {
        "日期": "trade_date",
        "date": "trade_date",
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
        "振幅": "amplitude_pct",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
        "turnover": "turnover",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    if "trade_date" not in df.columns:
        raise ValueError("日线数据缺少日期列，请检查 AkShare 版本与接口返回")
    df["symbol"] = symbol
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    # 数值列统一为 float，成交量可为 int，落库用 float 兼容 DuckDB
    num_cols = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude_pct",
        "pct_chg",
        "change",
        "turnover",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA
    cols_order = ["symbol", "trade_date"] + num_cols
    return df[cols_order].sort_values("trade_date").reset_index(drop=True)


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude_pct",
            "pct_chg",
            "change",
            "turnover",
        ]
    )


def _fill_derived_daily_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("trade_date").reset_index(drop=True).copy()
    prev_close = pd.to_numeric(out["close"], errors="coerce").shift(1)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    if "change" in out.columns:
        out["change"] = pd.to_numeric(out["change"], errors="coerce")
    else:
        out["change"] = pd.NA
    if "pct_chg" in out.columns:
        out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce")
    else:
        out["pct_chg"] = pd.NA
    if "amplitude_pct" in out.columns:
        out["amplitude_pct"] = pd.to_numeric(out["amplitude_pct"], errors="coerce")
    else:
        out["amplitude_pct"] = pd.NA

    out["change"] = out["change"].where(out["change"].notna(), close - prev_close)
    out["pct_chg"] = out["pct_chg"].where(
        out["pct_chg"].notna(),
        ((close - prev_close) / prev_close) * 100.0,
    )
    out["amplitude_pct"] = out["amplitude_pct"].where(
        out["amplitude_pct"].notna(),
        ((high - low) / prev_close) * 100.0,
    )
    return out


def fill_derived_daily_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    对单标的日线 DataFrame 补全涨跌额、涨跌幅、振幅（与 AkShare 口径一致：相对前收）。

    当源接口未返回或列为空时，用前收 ``close`` 与当日 OHLC 计算。
    """
    return _fill_derived_daily_fields(df)


def _symbol_with_exchange_prefix(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return f"sh{symbol}"
    if symbol.startswith(("4", "8")):
        return f"bj{symbol}"
    return f"sz{symbol}"


def _fetch_a_share_daily_via_em(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "qfq",
) -> pd.DataFrame:
    ak_mod = _require_akshare()
    raw = ak_mod.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust=adjust,
    )
    if raw is None or raw.empty:
        return _empty_daily_frame()
    df = _fill_derived_daily_fields(_standardize_daily_df(raw, symbol))
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    return df[(df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)].reset_index(drop=True)


def _fetch_a_share_daily_via_sina(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "qfq",
) -> pd.DataFrame:
    ak_mod = _require_akshare()
    raw = ak_mod.stock_zh_a_daily(
        symbol=_symbol_with_exchange_prefix(symbol),
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    if raw is None or raw.empty:
        return _empty_daily_frame()
    df = _fill_derived_daily_fields(_standardize_daily_df(raw, symbol))
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    return df[(df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)].reset_index(drop=True)


def fetch_a_share_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "qfq",
    timeout_sec: float = 10.0,
    source_preference: Optional[str] = None,
    allow_fallback: Optional[bool] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    拉取单只 A 股日线（6 位代码，无交易所前缀）。

    Parameters
    ----------
    symbol : str
        如 ``600519``、``000001``。
    start_date, end_date : str
        ``YYYYMMDD``。
    adjust : str
        ``qfq`` 前复权 | ``hfq`` 后复权 | ```` 不复权。
    """
    code = symbol.strip()
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"symbol 须为 6 位数字 A 股代码，收到: {symbol!r}")

    cfg = config or _load_config(None)
    install_akshare_requests_resilience(cfg)
    pref = (source_preference or cfg.get("akshare", {}).get("daily_source_preference", "sina")).lower()
    fallback_enabled = (
        bool(allow_fallback)
        if allow_fallback is not None
        else bool(cfg.get("akshare", {}).get("daily_allow_fallback", True))
    )

    ordered_sources: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        (
            "stock_zh_a_daily (sina)",
            lambda: _fetch_a_share_daily_via_sina(
                code,
                start_date,
                end_date,
                adjust=adjust,
            ),
        ),
        (
            "stock_zh_a_hist (em)",
            lambda: _fetch_a_share_daily_via_em(
                code,
                start_date,
                end_date,
                adjust=adjust,
            ),
        ),
    ]
    if pref in ("em", "eastmoney", "stock_zh_a_hist"):
        ordered_sources = [ordered_sources[1], ordered_sources[0]]
    if not fallback_enabled:
        ordered_sources = ordered_sources[:1]

    first_exc: Optional[BaseException] = None
    total_sources = len(ordered_sources)
    for idx, (source_name, fetcher) in enumerate(ordered_sources, start=1):
        try:
            return call_with_timeout(
                fetcher,
                timeout_sec=timeout_sec,
                label=f"{source_name}:{code}",
            )
        except Exception as exc:
            if first_exc is None:
                first_exc = exc
            if idx < total_sources:
                _LOG.warning(
                    "%s 失败 symbol=%s，尝试下一来源: %s: %s",
                    source_name,
                    code,
                    type(exc).__name__,
                    exc,
                )
            else:
                _LOG.warning(
                    "%s 失败 symbol=%s（无后续来源）: %s: %s",
                    source_name,
                    code,
                    type(exc).__name__,
                    exc,
                )
    if first_exc is not None:
        raise first_exc
    return _empty_daily_frame()


def fetch_etf_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "qfq",
    timeout_sec: float = 10.0,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Fetch an ETF daily series via AkShare Eastmoney ETF history."""
    code = symbol.strip().zfill(6)
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"ETF symbol 须为 6 位数字，收到: {symbol!r}")

    cfg = config or _load_config(None)
    install_akshare_requests_resilience(cfg)
    ak_mod = _require_akshare()

    def _fetch() -> pd.DataFrame:
        raw = ak_mod.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust=adjust,
        )
        if raw is None or raw.empty:
            return _empty_daily_frame()
        df = _fill_derived_daily_fields(_standardize_daily_df(raw, code))
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        return df[(df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)].reset_index(drop=True)

    return call_with_timeout(_fetch, timeout_sec=timeout_sec, label=f"fund_etf_hist_em:{code}")


def _project_root() -> Path:
    return project_root()


def _load_config(config_path: Optional[Union[str, Path]]) -> dict:
    return load_config(Path(config_path)) if config_path else load_config()


def _extract_symbol_codes(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None or df.empty:
        return []
    candidate_cols = ("代码", "证券代码", "股票代码", "symbol", "code")
    code_col = next((c for c in candidate_cols if c in df.columns), None)
    if code_col is None:
        code_col = df.columns[0]
    codes = (
        df[code_col]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return [c.zfill(6) for c in codes]


def _list_symbols_via_akshare(
    api_name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """与日线拉取共用 requests 超时/重试注入，config 与 ``fetch_only`` / DB 路径一致。"""
    if ak is None:
        _LOG.warning("AkShare 未安装，跳过股票池来源 %s。", api_name)
        return []
    install_akshare_requests_resilience(_load_config(config_path))
    fn = getattr(ak, api_name, None)
    if fn is None:
        return []
    return _extract_symbol_codes(fn())


def _list_symbols_from_duckdb(
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    if duckdb is None:
        _LOG.warning("DuckDB 未安装，跳过 DuckDB 股票池来源。")
        return []
    cfg = _load_config(config_path)
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    db_cfg = cfg.get("database", {}) if isinstance(cfg, dict) else {}
    duckdb_path = paths.get("duckdb_path", "data/market.duckdb")
    table_daily = db_cfg.get("table_daily", "a_share_daily")

    db_path = Path(duckdb_path)
    if not db_path.is_absolute():
        db_path = _project_root() / db_path
    if not db_path.exists():
        return []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            f"""
            SELECT DISTINCT symbol
            FROM {table_daily}
            WHERE symbol IS NOT NULL
            ORDER BY symbol
            """
        ).fetchall()
    finally:
        con.close()
    return [str(r[0]).zfill(6) for r in rows if r and r[0]]


def _universe_cache_path(
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> Path:
    cfg = _load_config(config_path)
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    p = Path(paths.get("universe_cache_path", "data/cache/universe_symbols.json"))
    if not p.is_absolute():
        p = _project_root() / p
    return p


def _load_symbols_from_local_cache(
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    path = _universe_cache_path(config_path=config_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _LOG.warning("本地股票池缓存读取失败 %s: %s", path, exc)
        return []
    if isinstance(payload, dict):
        return _extract_symbol_codes(pd.DataFrame({"code": payload.get("symbols", [])}))
    if isinstance(payload, list):
        return _extract_symbol_codes(pd.DataFrame({"code": payload}))
    return []


def _save_symbols_to_local_cache(
    symbols: List[str],
    *,
    source: str,
    config_path: Optional[Union[str, Path]] = None,
) -> None:
    path = _universe_cache_path(config_path=config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "count": len(symbols),
        "symbols": symbols,
    }
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        _LOG.warning("本地股票池缓存写入失败 %s: %s", path, exc)


def _fetch_source_with_retries(
    source: str,
    fetcher: Callable[[], List[str]],
    *,
    retries: int,
    retry_delay_sec: float,
    timeout_sec: float,
) -> List[str]:
    last_exc: Optional[BaseException] = None
    for attempt in range(max(1, retries)):
        try:
            codes = call_with_timeout(
                fetcher,
                timeout_sec=timeout_sec,
                label=f"股票池:{source}",
            )
            if codes:
                return codes
            _LOG.warning(
                "股票池来源 %s 第 %s/%s 次返回空列表。",
                source,
                attempt + 1,
                max(1, retries),
            )
        except Exception as exc:
            last_exc = exc
            _LOG.warning(
                "股票池来源 %s 第 %s/%s 次失败: %s: %s",
                source,
                attempt + 1,
                max(1, retries),
                type(exc).__name__,
                exc,
            )
        if attempt < max(1, retries) - 1:
            time.sleep(max(0.0, retry_delay_sec) * (attempt + 1))
    if last_exc is not None:
        raise RuntimeError(f"{source} exhausted retries") from last_exc
    return []


def _akshare_universe_fetchers(
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[Tuple[str, Callable[[], List[str]]], ...]:
    """
    全市场代码表来源顺序与日线增量「同源优先」：

    1. ``stock_zh_a_spot_em``：东财实时表，与 ``stock_zh_a_hist`` 同属东财链路，通常比
       ``stock_info_a_code_name``（北交所/多段请求）更稳、更少无意义超时。
    2. ``stock_info_a_code_name``：后备（含北交所等，网络差时易拖慢首轮）。
    """
    return (
        ("stock_zh_a_spot_em", lambda: _list_symbols_via_akshare("stock_zh_a_spot_em", config_path)),
        (
            "stock_info_a_code_name",
            lambda: _list_symbols_via_akshare("stock_info_a_code_name", config_path),
        ),
    )


def _fetch_akshare_universe_codes(
    *,
    config_path: Optional[Union[str, Path]] = None,
    source_timeout_sec: float,
    source_retries: int,
    retry_delay_sec: float,
) -> Optional[List[str]]:
    """
    依次尝试 AkShare 代码列表接口；成功则写入本地 universe 快照并返回代码列表。
    全部失败则返回 ``None``。
    """
    for source, fetcher in _akshare_universe_fetchers(config_path=config_path):
        try:
            codes = _fetch_source_with_retries(
                source,
                fetcher,
                retries=source_retries,
                retry_delay_sec=retry_delay_sec,
                timeout_sec=source_timeout_sec,
            )
        except Exception as exc:
            _LOG.warning(
                "股票池来源 %s 不可用，回退下一来源: %s: %s",
                source,
                type(exc).__name__,
                exc,
            )
            continue
        if codes:
            _save_symbols_to_local_cache(codes, source=source, config_path=config_path)
            _LOG.info("股票池来源: %s | 标的数: %s", source, len(codes))
            return codes
    return None


def _subsample_universe_symbols(
    symbols: List[str],
    max_symbols: Optional[int],
    *,
    seed: int = 42,
) -> List[str]:
    """
    在仅取前 N 只时，避免对「已按代码排序」的列表做 ``[:N]`` 截取。

    字典序下前几百只几乎全是深市 ``000xxx``，不含沪市 ``600/601``、创业板 ``300`` 等，
    会导致推荐结果看起来「全是 0000 开头」。此处用固定种子的随机子集，再按原列表顺序输出。
    """
    if max_symbols is None:
        return list(symbols)
    if max_symbols <= 0:
        return []
    if len(symbols) <= max_symbols:
        return list(symbols)
    rng = random.Random(seed)
    idx_all = list(range(len(symbols)))
    rng.shuffle(idx_all)
    chosen = frozenset(idx_all[:max_symbols])
    return [symbols[i] for i in range(len(symbols)) if i in chosen]


def list_default_universe_symbols(
    max_symbols: Optional[int] = None,
    *,
    config_path: Optional[Union[str, Path]] = None,
    source_timeout_sec: Optional[float] = None,
) -> List[str]:
    """
    返回当前可交易 A 股代码列表。

    默认（``universe_prefer_akshare: false``）：将 DuckDB 已有代码与本地 ``universe_symbols.json``
    取**并集**（可关 ``universe_merge_duckdb_and_cache``），若并集仍低于
    ``universe_target_min_symbols`` 则**只在此情况下**自动拉一次全市场代码表（东财优先，与日线共用
    HTTP 策略）并写入本地快照；之后并集达标，日常与增量一样不再请求列表接口。

    ``incremental_universe_duckdb_only: true``（默认）且库内标的数不低于 ``min_cached_universe_symbols``
    时：日线增量**仅使用 DuckDB 已有代码**，不合并本地全市场快照，避免每日对 5800+ 全表做增量拉取。

    ``universe_prefer_akshare: true`` 时：每次先请求 AkShare 全市场列表，失败再回退并集 / 本地。

    Parameters
    ----------
    max_symbols : int, optional
        仅取 N 只，用于测试或小规模试跑；为 ``None`` 时返回全部。
        若列表已按代码排序，不再使用字典序前 N 只（否则会几乎全是深市 ``000xxx``），
        而采用固定种子的随机子集，再保持原列表中的相对顺序。
    """
    cfg = _load_config(config_path)
    ak_cfg = cfg.get("akshare", {}) if isinstance(cfg, dict) else {}
    timeout_sec = (
        float(source_timeout_sec)
        if source_timeout_sec is not None
        else float(ak_cfg.get("universe_source_timeout_sec", 60.0))
    )
    source_retries = max(1, int(ak_cfg.get("universe_source_retries", 2)))
    retry_delay_sec = float(ak_cfg.get("universe_retry_delay_sec", 3.0))
    min_cached_symbols = max(1, int(ak_cfg.get("min_cached_universe_symbols", 1000)))
    prefer_akshare = bool(ak_cfg.get("universe_prefer_akshare", False))
    merge_sources = bool(ak_cfg.get("universe_merge_duckdb_and_cache", True))
    target_floor = int(ak_cfg.get("universe_target_min_symbols", 4500))
    incremental_duckdb_only = bool(ak_cfg.get("incremental_universe_duckdb_only", True))

    required_min = min_cached_symbols if max_symbols is None else max(1, max_symbols)
    duckdb_codes = _list_symbols_from_duckdb(config_path=config_path)
    local_cache_codes = _load_symbols_from_local_cache(config_path=config_path)

    if (
        incremental_duckdb_only
        and not prefer_akshare
        and duckdb_codes
        and len(duckdb_codes) >= min_cached_symbols
    ):
        _LOG.info(
            "股票池来源: duckdb_incremental_only（不合并本地全市场快照）| 标的数: %s",
            len(duckdb_codes),
        )
        return _subsample_universe_symbols(list(duckdb_codes), max_symbols)

    def _merged_base() -> List[str]:
        if merge_sources:
            return sorted(frozenset(duckdb_codes) | frozenset(local_cache_codes))
        return list(duckdb_codes) if duckdb_codes else list(local_cache_codes)

    if prefer_akshare:
        codes = _fetch_akshare_universe_codes(
            config_path=config_path,
            source_timeout_sec=timeout_sec,
            source_retries=source_retries,
            retry_delay_sec=retry_delay_sec,
        )
        if codes:
            return _subsample_universe_symbols(codes, max_symbols)
        _LOG.warning(
            "已启用 universe_prefer_akshare，但 AkShare 代码列表未成功，回退 DuckDB / 本地快照。"
        )

    base = _merged_base()
    if not prefer_akshare and target_floor > 0 and len(base) < target_floor:
        _LOG.info(
            "股票池并集 %s 低于 universe_target_min_symbols=%s，尝试补拉全市场代码表一次。",
            len(base),
            target_floor,
        )
        codes = _fetch_akshare_universe_codes(
            config_path=config_path,
            source_timeout_sec=timeout_sec,
            source_retries=source_retries,
            retry_delay_sec=retry_delay_sec,
        )
        if codes:
            base = sorted(frozenset(duckdb_codes) | frozenset(codes))
            _LOG.info(
                "股票池来源: merged_duckdb_and_akshare_universe | 标的数: %s",
                len(base),
            )
            if len(base) >= required_min:
                return _subsample_universe_symbols(base, max_symbols)
        else:
            _LOG.warning("全市场代码表补拉失败，继续使用 DuckDB / 本地并集或后续回退。")

    if merge_sources and base and len(base) >= required_min:
        _LOG.info("股票池来源: merged_duckdb_and_cache | 标的数: %s", len(base))
        return _subsample_universe_symbols(base, max_symbols)

    if not merge_sources and duckdb_codes:
        if len(duckdb_codes) >= required_min:
            _LOG.info("股票池来源: duckdb_cached_symbols | 标的数: %s", len(duckdb_codes))
            return _subsample_universe_symbols(duckdb_codes, max_symbols)
        _LOG.warning(
            "DuckDB 缓存标的过少（%s < %s），尝试本地快照与实时股票池来源。",
            len(duckdb_codes),
            required_min,
        )

    if not merge_sources and local_cache_codes:
        if len(local_cache_codes) >= required_min:
            _LOG.info("股票池来源: local_universe_cache | 标的数: %s", len(local_cache_codes))
            return _subsample_universe_symbols(local_cache_codes, max_symbols)
        _LOG.warning(
            "本地股票池快照过少（%s < %s），继续尝试实时股票池来源。",
            len(local_cache_codes),
            required_min,
        )

    if not prefer_akshare:
        codes = _fetch_akshare_universe_codes(
            config_path=config_path,
            source_timeout_sec=timeout_sec,
            source_retries=source_retries,
            retry_delay_sec=retry_delay_sec,
        )
        if codes:
            return _subsample_universe_symbols(codes, max_symbols)

    fallback_codes = local_cache_codes if len(local_cache_codes) >= len(duckdb_codes) else duckdb_codes
    if fallback_codes:
        _LOG.warning(
            "实时股票池全部失败，降级使用本地缓存来源 | 标的数: %s",
            len(fallback_codes),
        )
        return _subsample_universe_symbols(fallback_codes, max_symbols)
    return []


def fetch_many_daily(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    *,
    adjust: str = "qfq",
    sleep_sec: float = 0.0,
    timeout_sec: float = 10.0,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """顺序拉取多只标的并纵向合并；可选间隔以降低限流风险。"""
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = fetch_a_share_daily(
            sym,
            start_date,
            end_date,
            adjust=adjust,
            timeout_sec=timeout_sec,
            config=config,
        )
        if not df.empty:
            frames.append(df)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_max_symbols(value: int | None) -> int | None:
    """CLI compatibility: non-positive max-symbols means full universe.

    Migrated from scripts/fetch_only.py (A3: tests should import from src/).
    """
    if value is None:
        return None
    return value if value > 0 else None
