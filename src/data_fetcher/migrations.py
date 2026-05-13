"""Optional DuckDB schema migration helpers for the trend-timing system.

New databases are initialized directly by ``DuckDBManager``. This module is
only used when ``database.apply_legacy_migrations`` is enabled, primarily to
record a small schema version marker or to bring very old local databases up to
the current daily-bar shape.
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb

_LOG = logging.getLogger(__name__)

MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "daily_core_tables",
        """
        CREATE TABLE IF NOT EXISTS a_share_daily (
            symbol VARCHAR NOT NULL,
            trade_date DATE NOT NULL,
            open DOUBLE,
            close DOUBLE,
            high DOUBLE,
            low DOUBLE,
            volume DOUBLE,
            amount DOUBLE,
            amplitude_pct DOUBLE,
            pct_chg DOUBLE,
            change DOUBLE,
            turnover DOUBLE,
            PRIMARY KEY (symbol, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_a_share_daily_trade_date_symbol
            ON a_share_daily(trade_date, symbol);
        """,
    ),
    (
        2,
        "daily_derived_columns",
        """
        ALTER TABLE a_share_daily ADD COLUMN IF NOT EXISTS change DOUBLE;
        ALTER TABLE a_share_daily ADD COLUMN IF NOT EXISTS pct_chg DOUBLE;
        ALTER TABLE a_share_daily ADD COLUMN IF NOT EXISTS amplitude_pct DOUBLE;
        """,
    ),
    (
        3,
        "data_fetch_audit",
        """
        CREATE TABLE IF NOT EXISTS data_fetch_audit (
            run_id VARCHAR NOT NULL,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP NOT NULL,
            symbol_count INTEGER NOT NULL,
            rows_written BIGINT NOT NULL,
            failures INTEGER NOT NULL,
            duration_ms BIGINT NOT NULL,
            PRIMARY KEY (run_id)
        );
        """,
    ),
]

_META_TABLE = "schema_migrations"
_META_DDL = f"""
CREATE TABLE IF NOT EXISTS {_META_TABLE} (
    version INTEGER PRIMARY KEY,
    label VARCHAR NOT NULL,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _current_version(conn: duckdb.DuckDBPyConnection) -> int:
    try:
        row = conn.execute(f"SELECT MAX(version) FROM {_META_TABLE}").fetchone()
    except Exception:
        return 0
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def apply_migrations(
    conn: duckdb.DuckDBPyConnection,
    *,
    target_version: int | None = None,
) -> list[int]:
    """Apply pending forward-only migrations and return applied versions."""
    conn.execute(_META_DDL)
    current = _current_version(conn)
    target = max((v for v, _, _ in MIGRATIONS), default=0) if target_version is None else max(0, int(target_version))

    applied: list[int] = []
    for ver, label, sql in MIGRATIONS:
        if ver <= current:
            continue
        if ver > target:
            break
        _LOG.info("Applying migration v%d: %s", ver, label)
        try:
            conn.execute("BEGIN TRANSACTION;")
            for stmt in _split_sql(sql):
                conn.execute(stmt)
            conn.execute(f"INSERT INTO {_META_TABLE} (version, label) VALUES (?, ?);", [ver, label])
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        applied.append(ver)
    return applied


def get_migration_status(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Return current, applied, and pending migration metadata."""
    current = _current_version(conn)
    applied = [{"version": ver, "label": label} for ver, label, _ in MIGRATIONS if ver <= current]
    pending = [{"version": ver, "label": label} for ver, label, _ in MIGRATIONS if ver > current]
    return {"current_version": current, "applied": applied, "pending": pending}


def _split_sql(text: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(buf))
            buf = []
    if buf:
        statements.append("\n".join(buf))
    return [s.strip() for s in statements if s.strip()]
