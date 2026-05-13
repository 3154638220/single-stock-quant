"""结构化事件日志（DuckDB run_events 表）。

记录系统关键决策事件（涨停过滤、信号触发、数据拉取失败等），
支持后续 SQL 查询分析。

用法::

    from src.logging_config import log_event

    log_event(conn, "limit_up_filter", {
        "symbol": "600001",
        "signal_date": "2026-04-28",
        "action": "buy_delayed",
    }, run_id="signal_20260428")

表结构会在首次写入或查询前自动确保存在::

    CREATE TABLE IF NOT EXISTS run_events (
        event_id BIGINT PRIMARY KEY DEFAULT nextval('seq_run_events_id'),
        run_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        event_type VARCHAR NOT NULL,
        event_payload JSON,
        run_id VARCHAR,
        symbol VARCHAR,
        signal_date DATE
    );
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import duckdb

_LOG = logging.getLogger(__name__)


def ensure_event_log_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure the lightweight event log table exists in the current DuckDB file."""
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_run_events_id START 1")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_events (
            event_id BIGINT PRIMARY KEY DEFAULT nextval('seq_run_events_id'),
            run_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            event_type VARCHAR NOT NULL,
            event_payload JSON,
            run_id VARCHAR,
            symbol VARCHAR,
            signal_date DATE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_events_type ON run_events(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_events_ts ON run_events(run_ts)")

# ── 事件类型枚举 ─────────────────────────────────────────────────────────

class EventType:
    """常用事件类型常量。"""
    LIMIT_UP_FILTER = "limit_up_filter"
    TREND_SIGNAL = "trend_signal"
    BACKTEST_RUN = "backtest_run"
    DATA_FETCH_FAILURE = "data_fetch_failure"
    DATA_FETCH_SUCCESS = "data_fetch_success"
    CONFIG_VALIDATION_ERROR = "config_validation_error"
    SCHEMA_MIGRATION = "schema_migration"


def log_event(
    conn: duckdb.DuckDBPyConnection,
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    run_id: str | None = None,
    symbol: str | None = None,
    signal_date: str | date | None = None,
) -> bool:
    """
    写入一条结构化事件到 run_events 表。

    Parameters
    ----------
    conn
        DuckDB 连接。调用方负责管理连接生命周期。
    event_type
        事件类型（建议使用 EventType 常量）。
    payload
        事件负载（JSON 对象），可为 None。
    run_id
        管线运行标识（可选）。
    symbol
        标的代码（可选，如 600001）。
    signal_date
        信号日（可选）。

    Returns
    -------
    bool
        写入成功返回 True，异常返回 False。
    """
    try:
        ensure_event_log_schema(conn)
        payload_json = json.dumps(payload, ensure_ascii=False) if payload else None
        conn.execute(
            """
            INSERT INTO run_events
                (event_type, event_payload, run_id, symbol, signal_date)
            VALUES (?, ?::JSON, ?, ?, ?::DATE)
            """,
            [
                str(event_type),
                payload_json,
                run_id,
                symbol,
                str(signal_date) if signal_date else None,
            ],
        )
        return True
    except Exception as exc:
        _LOG.warning("写入事件日志失败 (type=%s): %s", event_type, exc)
        return False


def query_events(
    conn: duckdb.DuckDBPyConnection,
    *,
    event_type: str | None = None,
    run_id: str | None = None,
    signal_date: str | date | None = None,
    symbol: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    查询 run_events 表中的事件。

    Parameters
    ----------
    conn
        DuckDB 连接。
    event_type
        按事件类型过滤（可选）。
    run_id
        按运行 ID 过滤（可选）。
    signal_date
        按信号日过滤（可选）。
    symbol
        按标的过滤（可选）。
    limit
        返回最大行数。

    Returns
    -------
    list[dict]
        事件记录列表。
    """
    where: list[str] = []
    params: list[Any] = []

    if event_type:
        where.append("event_type = ?")
        params.append(event_type)
    if run_id:
        where.append("run_id = ?")
        params.append(run_id)
    if signal_date:
        where.append("signal_date = ?::DATE")
        params.append(str(signal_date))
    if symbol:
        where.append("symbol = ?")
        params.append(symbol)

    clause = " AND ".join(where) if where else "TRUE"
    q = f"""
        SELECT event_id, run_ts, event_type, event_payload, run_id, symbol, signal_date
        FROM run_events
        WHERE {clause}
        ORDER BY run_ts DESC
        LIMIT {int(limit)}
    """

    try:
        ensure_event_log_schema(conn)
        rows = conn.execute(q, params).fetchall()
    except Exception as exc:
        _LOG.warning("查询事件日志失败: %s", exc)
        return []

    return [
        {
            "event_id": int(r[0]),
            "run_ts": str(r[1]),
            "event_type": str(r[2]),
            "event_payload": json.loads(r[3]) if r[3] else None,
            "run_id": str(r[4]) if r[4] else None,
            "symbol": str(r[5]) if r[5] else None,
            "signal_date": str(r[6]) if r[6] else None,
        }
        for r in rows
    ]
