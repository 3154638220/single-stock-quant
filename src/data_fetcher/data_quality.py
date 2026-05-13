"""数据质量检查：日线表与复权跳变。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd


@dataclass
class QualityConfig:
    """与 config.yaml 中 ``quality`` 段对齐。"""

    max_calendar_gap_days: int = 20
    null_ratio_max: float = 0.05
    ohlc_cols: tuple[str, ...] = ("open", "high", "low", "close")
    # 若为 False：仅统计，不把 OHLC/长假间隔判为「未通过」（常见于源站、停牌、长假）
    fail_on_ohlc_invalid: bool = True
    fail_on_large_gaps: bool = True

    @classmethod
    def from_mapping(cls, m: Optional[Dict[str, Any]]) -> "QualityConfig":
        if not m:
            return cls()
        return cls(
            max_calendar_gap_days=int(m.get("max_calendar_gap_days", 20)),
            null_ratio_max=float(m.get("null_ratio_max", 0.05)),
            ohlc_cols=tuple(
                m.get("ohlc_cols", ["open", "high", "low", "close"]),
            ),
            fail_on_ohlc_invalid=bool(m.get("fail_on_ohlc_invalid", True)),
            fail_on_large_gaps=bool(m.get("fail_on_large_gaps", True)),
        )


@dataclass
class QualityReport:
    ok: bool
    duplicate_pk_rows: int = 0
    ohlc_invalid_rows: int = 0
    large_gap_rows: int = 0
    null_ratio_violations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"duplicate_pk={self.duplicate_pk_rows}",
            f"ohlc_invalid={self.ohlc_invalid_rows}",
            f"large_gaps={self.large_gap_rows}",
            f"null_violations={len(self.null_ratio_violations)}",
        ]
        return "; ".join(parts)


def _duplicate_pk_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    q = f"""
    SELECT COALESCE(SUM(cnt - 1), 0)::BIGINT AS dup_extra
    FROM (
      SELECT symbol, trade_date, COUNT(*) AS cnt
      FROM {table}
      GROUP BY 1, 2
      HAVING COUNT(*) > 1
    ) t
    """
    row = conn.execute(q).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def _ohlc_invalid_count(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    cols: tuple[str, ...],
) -> int:
    """违反 high/low 与 open/close 关系的行数（四价均非空时检查）。"""
    o, h, lo, c = cols[0], cols[1], cols[2], cols[3]
    q = f"""
    SELECT COUNT(*)::BIGINT
    FROM {table}
    WHERE {o} IS NOT NULL AND {h} IS NOT NULL AND {lo} IS NOT NULL AND {c} IS NOT NULL
      AND (
        {h} < GREATEST({o}, {c})
        OR {lo} > LEAST({o}, {c})
        OR {h} < {lo}
        OR {o} <= 0 OR {h} <= 0 OR {lo} <= 0 OR {c} <= 0
      )
    """
    row = conn.execute(q).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def _large_gap_count(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    max_gap_days: int,
) -> int:
    """相邻交易日历间隔超过阈值的条数（按 symbol 分区）。"""
    q = f"""
    WITH w AS (
      SELECT
        symbol,
        trade_date,
        lag(trade_date) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_d
      FROM {table}
    )
    SELECT COUNT(*)::BIGINT
    FROM w
    WHERE prev_d IS NOT NULL
      AND date_diff('day', prev_d, trade_date) > {int(max_gap_days)}
    """
    row = conn.execute(q).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def _describe_column_names(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = conn.execute(f"DESCRIBE {table}").fetchall()
    return {str(r[0]) for r in rows}


def _null_ratio_checks(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    threshold: float,
    columns: List[str],
) -> List[str]:
    violations: List[str] = []
    total = conn.execute(f"SELECT COUNT(*)::BIGINT FROM {table}").fetchone()
    if total is None:
        return violations
    n = int(total[0] or 0)
    if n == 0:
        return violations
    existing = _describe_column_names(conn, table)
    for col in columns:
        if col not in existing:
            continue
        row = conn.execute(
            f"""
            SELECT SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END)::BIGINT
            FROM {table}
            """,
        ).fetchone()
        if row is None:
            continue
        nulls = int(row[0] or 0)
        ratio = nulls / n
        if ratio > threshold:
            violations.append(f"{col}: {ratio:.4f} > {threshold}")
    return violations


def run_quality_checks(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    cfg: Optional[QualityConfig] = None,
) -> QualityReport:
    """
    对日线表执行质量检查（只读）。

    - 主键重复：``(symbol, trade_date)`` 出现次数 > 1 的额外行数之和
    - OHLC：四价齐全时校验 high/low 与开收关系及正数
    - 连续性：同 symbol 相邻交易日历间隔 > ``max_calendar_gap_days``
    - 空值：各列空值比例超过 ``null_ratio_max`` 则记入 violations
    """
    qc = cfg or QualityConfig()
    notes: List[str] = []

    dup = _duplicate_pk_count(conn, table)
    ohlc_n = _ohlc_invalid_count(conn, table, qc.ohlc_cols)
    gaps = _large_gap_count(conn, table, qc.max_calendar_gap_days)

    price_cols = list(qc.ohlc_cols) + [
        c
        for c in ("volume", "amount", "amplitude_pct", "pct_chg", "change", "turnover")
        if c not in qc.ohlc_cols
    ]
    null_v = _null_ratio_checks(conn, table, qc.null_ratio_max, price_cols)

    ohlc_fail = qc.fail_on_ohlc_invalid and ohlc_n > 0
    gap_fail = qc.fail_on_large_gaps and gaps > 0
    ok = dup == 0 and not ohlc_fail and not gap_fail and len(null_v) == 0
    if not ok:
        if dup:
            notes.append("存在重复主键，请修复后再依赖该表。")
        if ohlc_fail:
            notes.append("存在 OHLC 逻辑非法或非正价格行。")
        if gap_fail:
            notes.append(
                f"存在相邻交易日历间隔 > {qc.max_calendar_gap_days} 天的记录（长假或缺数）。",
            )
        if null_v:
            notes.append("部分列空值比例超过阈值：" + "; ".join(null_v))

    return QualityReport(
        ok=ok,
        duplicate_pk_rows=dup,
        ohlc_invalid_rows=ohlc_n,
        large_gap_rows=gaps,
        null_ratio_violations=null_v,
        notes=notes,
    )


def validate_daily_frame(df: pd.DataFrame, *, cfg: Optional[QualityConfig] = None) -> QualityReport:
    """
    对尚未落库的 DataFrame 做与库内一致的 OHLC / 主键检查（用于写入前校验）。
    不含「大间隔」检查（需全历史排序上下文）。
    """
    qc = cfg or QualityConfig()
    notes: List[str] = []
    if df.empty:
        return QualityReport(ok=True, notes=["empty frame"])

    dup = 0
    if "symbol" in df.columns and "trade_date" in df.columns:
        dup = len(df) - len(df.drop_duplicates(subset=["symbol", "trade_date"]))

    o, h, lo, c = qc.ohlc_cols[0], qc.ohlc_cols[1], qc.ohlc_cols[2], qc.ohlc_cols[3]
    sub = df[[o, h, lo, c]].dropna(how="any")
    ohlc_n = 0
    if not sub.empty:
        bad = (
            (sub[h] < sub[[o, c]].max(axis=1))
            | (sub[lo] > sub[[o, c]].min(axis=1))
            | (sub[h] < sub[lo])
            | (sub[o] <= 0)
            | (sub[h] <= 0)
            | (sub[lo] <= 0)
            | (sub[c] <= 0)
        )
        ohlc_n = int(bad.sum())

    null_v: List[str] = []
    price_cols = list(qc.ohlc_cols) + [
        x
        for x in ("volume", "amount", "amplitude_pct", "pct_chg", "change", "turnover")
        if x in df.columns and x not in qc.ohlc_cols
    ]
    for col in price_cols:
        if col not in df.columns:
            continue
        ratio = float(df[col].isna().mean())
        if ratio > qc.null_ratio_max:
            null_v.append(f"{col}: {ratio:.4f} > {qc.null_ratio_max}")

    ohlc_fail = qc.fail_on_ohlc_invalid and ohlc_n > 0
    ok = dup == 0 and not ohlc_fail and len(null_v) == 0
    if not ok:
        if dup:
            notes.append("DataFrame 内存在重复 (symbol, trade_date)。")
        if ohlc_fail:
            notes.append("存在 OHLC 非法行。")
        if null_v:
            notes.extend(null_v)

    return QualityReport(
        ok=ok,
        duplicate_pk_rows=dup,
        ohlc_invalid_rows=ohlc_n,
        large_gap_rows=0,
        null_ratio_violations=null_v,
        notes=notes,
    )


# ── P2-7: 复权检测 ──────────────────────────────────────────────────────

def check_split_jump(
    df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    pct_chg_col: str = "pct_chg",
    close_col: str = "close",
    volume_col: str = "volume",
    jump_threshold: float = 0.50,
) -> list[str]:
    """P2-7: 检测单日涨跌幅绝对值超过阈值但非停牌的非复权数据跳变。

    除权除息日价格会出现断崖式跳变（不复权数据），表现为单日涨跌幅绝对值
    远超涨跌停限制且成交量正常（非停牌）。此函数检测这类异常并返回警告列表。

    Parameters
    ----------
    df : 日线数据，须含 symbol_col, date_col, pct_chg_col 等列
    jump_threshold : 单日涨幅绝对值阈值（默认 50%，超出涨跌停限制）

    Returns
    -------
    list[str] : 告警消息列表，每项描述一个可疑跳变
    """
    alerts: list[str] = []
    need = {symbol_col, date_col, pct_chg_col, close_col, volume_col}
    missing = need - set(df.columns)
    if missing:
        alerts.append(f"check_split_jump: 缺少列 {missing}")
        return alerts

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
    d[pct_chg_col] = pd.to_numeric(d[pct_chg_col], errors="coerce")
    d[volume_col] = pd.to_numeric(d[volume_col], errors="coerce")
    d[close_col] = pd.to_numeric(d[close_col], errors="coerce")

    # 非停牌（volume > 0）且涨跌幅绝对值 > threshold
    suspicious = d[
        (d[volume_col].notna() & (d[volume_col] > 0))
        & d[pct_chg_col].notna()
        & (d[pct_chg_col].abs() > jump_threshold)
        & d[close_col].notna()
        & (d[close_col] > 0)
    ]

    if suspicious.empty:
        return alerts

    for _, row in suspicious.iterrows():
        sym = str(row[symbol_col]).zfill(6)
        dt = row[date_col]
        pct = float(row[pct_chg_col])
        alerts.append(
            f"[复权警告] {sym} @ {dt.date()}: "
            f"单日涨跌幅 {pct:+.2%}，成交量正常，疑似未复权数据跳变"
        )

    import logging
    _log = logging.getLogger(__name__)
    for alert in alerts:
        _log.warning(alert)

    return alerts
