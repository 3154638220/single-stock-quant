"""DuckDB 本地落盘与按交易日增量更新。"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

import duckdb
import pandas as pd

from src.settings import load_config, project_root

from .akshare_client import fetch_a_share_daily, fill_derived_daily_fields
from .data_quality import QualityConfig, QualityReport, run_quality_checks, validate_daily_frame

_LOG = logging.getLogger(__name__)


class SymbolUpdateResult(NamedTuple):
    """单标的增量结果：写入行数与是否拉取失败（与「无新数据」区分）。"""

    rows_written: int
    fetch_failed: bool


class SymbolFetchPayload(NamedTuple):
    symbol: str
    start_s: str
    end_dt: date
    df: Optional[pd.DataFrame]
    fetch_failed: bool
    error: Optional[BaseException]


class SymbolFetchPlan(NamedTuple):
    symbol: str
    start_s: str
    end_dt: date


class DuckDBManager:
    """
    管理 DuckDB 连接、日线表结构，以及按 ``symbol`` 的增量拉取与写入。
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        *,
        duckdb_path: Optional[str] = None,
        table_daily: Optional[str] = None,
    ) -> None:
        root = project_root()
        cfg = load_config(Path(config_path)) if config_path else load_config()
        self._cfg = cfg

        paths = cfg.get("paths", {})
        db_cfg = cfg.get("database", {})
        ak_cfg = cfg.get("akshare", {})
        qual_cfg = cfg.get("quality", {})

        self._duckdb_path = duckdb_path or paths.get("duckdb_path", "data/market.duckdb")
        self._table = table_daily or db_cfg.get("table_daily", "a_share_daily")
        self._table_audit = db_cfg.get("table_audit", "data_fetch_audit")
        self._quality = QualityConfig.from_mapping(qual_cfg if isinstance(qual_cfg, dict) else None)
        self._adjust = ak_cfg.get("adjust", "qfq")
        self._sleep = float(ak_cfg.get("sleep_between_symbols_sec", 0.0))
        self._max_fetch_retries = max(1, int(ak_cfg.get("max_fetch_retries", 3)))
        self._retry_delay_sec = float(ak_cfg.get("retry_delay_sec", 2.0))
        self._timeout_sec = float(ak_cfg.get("request_timeout_sec", 10.0))
        self._fetch_workers = max(1, int(ak_cfg.get("fetch_workers", 4)))
        self._daily_allow_fallback = bool(ak_cfg.get("daily_allow_fallback", True))
        self._backfill_derived_after_fetch = bool(ak_cfg.get("backfill_derived_after_fetch", False))
        self._auto_backfill_derived_on_init = bool(db_cfg.get("auto_backfill_derived_on_init", True))
        self._apply_legacy_migrations = bool(db_cfg.get("apply_legacy_migrations", False))

        logs_dir = paths.get("logs_dir", "data/logs")
        ld = Path(logs_dir)
        if not ld.is_absolute():
            ld = root / ld
        self._fetch_fail_log_dir = ld

        abs_db = Path(self._duckdb_path)
        if not abs_db.is_absolute():
            abs_db = root / abs_db
        self._duckdb_path_abs = abs_db
        abs_db.parent.mkdir(parents=True, exist_ok=True)

        self._conn = duckdb.connect(str(abs_db))
        self._write_lock = threading.Lock()
        self._ensure_schema()
        self._ensure_audit_schema()
        self._auto_backfill_derived_columns_if_needed()
        if self._apply_legacy_migrations:
            self._apply_pending_migrations()
        self._last_fetch_run_id: Optional[str] = None

    def _append_fetch_failure_log(self, symbol: str, exc: BaseException) -> None:
        """失败标的写入单独日志（按日滚动文件名），便于事后补拉。"""
        self._fetch_fail_log_dir.mkdir(parents=True, exist_ok=True)
        stem = f"akshare_symbol_failed_{datetime.now().strftime('%Y%m%d')}.log"
        path = self._fetch_fail_log_dir / stem
        line = (
            f"{datetime.now().isoformat(timespec='seconds')}\t{symbol}\t"
            f"{type(exc).__name__}: {exc}\n"
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    @property
    def last_fetch_run_id(self) -> Optional[str]:
        """最近一次 ``incremental_update_many`` 使用的 ``run_id``（审计主键）。"""
        return self._last_fetch_run_id

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DuckDBManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
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
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_trade_date_symbol
            ON {self._table}(trade_date, symbol)
            """
        )

    def _apply_pending_migrations(self) -> None:
        """Apply legacy DuckDB schema migrations when explicitly enabled."""
        try:
            from .migrations import apply_migrations
            applied = apply_migrations(self._conn)
            if applied:
                _LOG.info("已应用 %d 个 schema migration: %s", len(applied), applied)
        except Exception as exc:
            _LOG.warning("Schema migration 跳过（非阻塞）: %s", exc)

    def _missing_derived_daily_columns(self) -> list[str]:
        rows = self._conn.execute(f"PRAGMA table_info('{self._table}')").fetchall()
        cols = {str(r[1]) for r in rows if len(r) > 1 and r[1] is not None}
        required = {"change", "pct_chg", "amplitude_pct"}
        return sorted(required - cols)

    def _auto_backfill_derived_columns_if_needed(self) -> None:
        if not self._auto_backfill_derived_on_init:
            return
        missing = self._missing_derived_daily_columns()
        if missing:
            _LOG.warning(
                "跳过初始化衍生列回填：表 %s 缺少列 %s。",
                self._table,
                ",".join(missing),
            )
            return
        fixed = self.backfill_derived_daily_columns()
        if fixed > 0:
            _LOG.info("初始化阶段自动回填衍生列完成，修复行数: %s", fixed)

    def _ensure_audit_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_audit} (
                run_id VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                finished_at TIMESTAMP NOT NULL,
                symbol_count INTEGER NOT NULL,
                rows_written BIGINT NOT NULL,
                failures INTEGER NOT NULL,
                duration_ms BIGINT NOT NULL,
                PRIMARY KEY (run_id)
            )
            """
        )

    def record_fetch_audit(
        self,
        run_id: str,
        started_at: datetime,
        finished_at: datetime,
        symbol_count: int,
        rows_written: int,
        failures: int,
        duration_ms: int,
    ) -> None:
        """写入一次拉取审计记录（幂等：同一 ``run_id`` 仅应插入一次）。"""
        with self._write_lock:
            self._conn.execute(
            f"""
            INSERT OR REPLACE INTO {self._table_audit}
            (run_id, started_at, finished_at, symbol_count, rows_written, failures, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                started_at,
                finished_at,
                symbol_count,
                rows_written,
                failures,
                duration_ms,
            ],
        )

    def last_trade_date(self, symbol: str) -> Optional[date]:
        row = self._conn.execute(
            f"""
            SELECT MAX(trade_date) FROM {self._table}
            WHERE symbol = ?
            """,
            [symbol],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        d = row[0]
        if isinstance(d, datetime):
            return d.date()
        return d if isinstance(d, date) else pd.Timestamp(d).date()

    def incremental_update_symbol(
        self,
        symbol: str,
        *,
        default_start: str = "20150101",
        end_date: Optional[str] = None,
    ) -> SymbolUpdateResult:
        """
        对单标的增量更新：从库中最后交易日的下一自然日拉到 ``end_date``（默认今天）。

        Returns
        -------
        SymbolUpdateResult
            ``rows_written`` 与 ``fetch_failed``（拉取经重试仍失败时为 True）。
        """
        plan = self._plan_symbol_window(
            symbol,
            default_start=default_start,
            end_date=end_date,
        )
        if plan is None:
            return SymbolUpdateResult(0, False)
        payload = self._fetch_symbol_payload(plan)
        return self._write_symbol_payload(payload)

    def _plan_symbol_window(
        self,
        symbol: str,
        *,
        default_start: str,
        end_date: Optional[str],
    ) -> SymbolFetchPlan | None:
        today = datetime.now().date()
        end_s = end_date or today.strftime("%Y%m%d")
        end_dt = pd.to_datetime(end_s).date()

        last = self.last_trade_date(symbol)
        if last is None:
            start_s = default_start
        else:
            start_dt = last + timedelta(days=1)
            if start_dt > end_dt:
                return None
            start_s = start_dt.strftime("%Y%m%d")
        return SymbolFetchPlan(symbol=symbol, start_s=start_s, end_dt=end_dt)

    def _build_fetch_plans(
        self,
        symbols: List[str],
        *,
        default_start: str,
        end_date: Optional[str],
    ) -> tuple[list[SymbolFetchPlan], Dict[str, SymbolUpdateResult]]:
        plans: list[SymbolFetchPlan] = []
        done: Dict[str, SymbolUpdateResult] = {}
        for sym in symbols:
            plan = self._plan_symbol_window(
                sym,
                default_start=default_start,
                end_date=end_date,
            )
            if plan is None:
                done[sym] = SymbolUpdateResult(0, False)
                continue
            plans.append(plan)
        return plans, done

    def _fetch_symbol_payload(
        self,
        plan: SymbolFetchPlan,
    ) -> SymbolFetchPayload:
        end_s = plan.end_dt.strftime("%Y%m%d")
        df: Optional[pd.DataFrame] = None
        last_exc: Optional[BaseException] = None
        for attempt in range(self._max_fetch_retries):
            try:
                df = fetch_a_share_daily(
                    plan.symbol,
                    plan.start_s,
                    end_s,
                    adjust=self._adjust,
                    timeout_sec=self._timeout_sec,
                    allow_fallback=self._daily_allow_fallback,
                    config=self._cfg,
                )
                return SymbolFetchPayload(
                    symbol=plan.symbol,
                    start_s=plan.start_s,
                    end_dt=plan.end_dt,
                    df=df,
                    fetch_failed=False,
                    error=None,
                )
            except Exception as e:
                last_exc = e
                _LOG.warning(
                    "AkShare 拉取失败 symbol=%s 第 %s/%s 次: %s",
                    plan.symbol,
                    attempt + 1,
                    self._max_fetch_retries,
                    e,
                )
                if attempt < self._max_fetch_retries - 1:
                    time.sleep(self._retry_delay_sec * (attempt + 1))
        if last_exc is not None:
            self._append_fetch_failure_log(plan.symbol, last_exc)
        return SymbolFetchPayload(
            symbol=plan.symbol,
            start_s=plan.start_s,
            end_dt=plan.end_dt,
            df=None,
            fetch_failed=True,
            error=last_exc,
        )

    def _write_symbol_payload(self, payload: SymbolFetchPayload) -> SymbolUpdateResult:
        df = self._prepare_payload_frame(payload)
        if payload.fetch_failed:
            return SymbolUpdateResult(0, True)
        if df is None:
            return SymbolUpdateResult(0, False)

        qrep = validate_daily_frame(df, cfg=self._quality)
        if not qrep.ok:
            _LOG.warning("写入前质量提示 symbol=%s: %s", payload.symbol, qrep.summary())
            if qrep.notes:
                _LOG.warning("%s", "; ".join(qrep.notes))

        self._conn.register("df_incr", df)
        try:
            with self._write_lock:
                self._conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {self._table}
                    SELECT * FROM df_incr
                    """
                )
        finally:
            self._conn.unregister("df_incr")
        return SymbolUpdateResult(len(df), False)

    def _prepare_payload_frame(self, payload: SymbolFetchPayload) -> Optional[pd.DataFrame]:
        if payload.fetch_failed:
            return None
        df = payload.df
        if df is None or df.empty:
            return None
        df = df[df["trade_date"] <= pd.Timestamp(payload.end_dt)]
        if df.empty:
            return None
        df = df.sort_values("trade_date").drop_duplicates(
            subset=["symbol", "trade_date"],
            keep="last",
        )
        return fill_derived_daily_fields(df)

    def _write_payload_batch(
        self,
        payloads: list[SymbolFetchPayload],
        counts: Dict[str, SymbolUpdateResult],
    ) -> None:
        frames: list[pd.DataFrame] = []
        for payload in payloads:
            if payload.fetch_failed:
                counts[payload.symbol] = SymbolUpdateResult(0, True)
                continue
            df = self._prepare_payload_frame(payload)
            if df is None:
                counts[payload.symbol] = SymbolUpdateResult(0, False)
                continue
            qrep = validate_daily_frame(df, cfg=self._quality)
            if not qrep.ok:
                _LOG.warning("写入前质量提示 symbol=%s: %s", payload.symbol, qrep.summary())
                if qrep.notes:
                    _LOG.warning("%s", "; ".join(qrep.notes))
            frames.append(df)
            counts[payload.symbol] = SymbolUpdateResult(len(df), False)

        if not frames:
            return

        batch = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "trade_date"])
            .drop_duplicates(subset=["symbol", "trade_date"], keep="last")
            .reset_index(drop=True)
        )
        self._conn.register("df_incr_batch", batch)
        try:
            with self._write_lock:
                self._conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {self._table}
                    SELECT * FROM df_incr_batch
                    """
                )
        finally:
            self._conn.unregister("df_incr_batch")

    def backfill_derived_daily_columns(self) -> int:
        """
        全表检查 ``amplitude_pct`` / ``pct_chg`` / ``change``：对仍为 NULL 的行用前收与 OHLC 补算并 UPDATE。

        每 symbol 的首个交易日无前收，无法推导，保持 NULL。返回本轮消除的「含 NULL 行」数量（前后计数差）。
        """
        t = self._table
        row = self._conn.execute(
            f"""
            SELECT COUNT(*)::BIGINT FROM {t}
            WHERE change IS NULL OR pct_chg IS NULL OR amplitude_pct IS NULL
            """
        ).fetchone()
        assert row is not None, "COUNT(*) 查询不应返回 None"
        n_before = int(row[0] or 0)
        if n_before == 0:
            return 0
        with self._write_lock:
            self._conn.execute(
            f"""
            WITH w AS (
              SELECT
                symbol,
                trade_date,
                close,
                high,
                low,
                change,
                pct_chg,
                amplitude_pct,
                LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_close
              FROM {t}
            )
            UPDATE {t} AS d
            SET
              change = COALESCE(d.change, w.close - w.prev_close),
              pct_chg = COALESCE(d.pct_chg,
                CASE WHEN w.prev_close IS NOT NULL AND ABS(w.prev_close) > 1e-12
                  THEN (w.close - w.prev_close) / w.prev_close * 100.0 ELSE NULL END),
              amplitude_pct = COALESCE(d.amplitude_pct,
                CASE WHEN w.prev_close IS NOT NULL AND ABS(w.prev_close) > 1e-12
                  THEN (w.high - w.low) / w.prev_close * 100.0 ELSE NULL END)
            FROM w
            WHERE d.symbol = w.symbol AND d.trade_date = w.trade_date
              AND (d.change IS NULL OR d.pct_chg IS NULL OR d.amplitude_pct IS NULL)
            """
        )
        row2 = self._conn.execute(
            f"""
            SELECT COUNT(*)::BIGINT FROM {t}
            WHERE change IS NULL OR pct_chg IS NULL OR amplitude_pct IS NULL
            """
        ).fetchone()
        assert row2 is not None, "COUNT(*) 查询不应返回 None"
        n_after = int(row2[0] or 0)
        fixed = n_before - n_after
        _LOG.info(
            "全表衍生列回填: 含 NULL 行 %s -> %s（本轮减少 %s 行）",
            n_before,
            n_after,
            fixed,
        )
        return fixed

    def incremental_update_many(
        self,
        symbols: List[str],
        *,
        default_start: str = "20150101",
        end_date: Optional[str] = None,
        record_audit: bool = True,
        run_id: Optional[str] = None,
        backfill_derived_after: Optional[bool] = None,
    ) -> Dict[str, SymbolUpdateResult]:
        """批量增量更新；可选写入审计表（``run_id`` 默认新建 UUID）。

        结束后默认对全表回填 ``amplitude_pct`` / ``pct_chg`` / ``change`` 中的缺失值（见 ``backfill_derived_daily_columns``）。
        """
        rid = run_id or str(uuid.uuid4())
        do_backfill_derived = (
            self._backfill_derived_after_fetch
            if backfill_derived_after is None
            else bool(backfill_derived_after)
        )
        self._last_fetch_run_id = rid
        started_at = datetime.now()
        t0 = time.perf_counter()
        plans, counts = self._build_fetch_plans(
            symbols,
            default_start=default_start,
            end_date=end_date,
        )
        self._conn.execute("BEGIN TRANSACTION")
        try:
            payloads: list[SymbolFetchPayload] = []
            if self._fetch_workers <= 1 or len(plans) <= 1:
                for plan in plans:
                    payloads.append(self._fetch_symbol_payload(plan))
                    if self._sleep > 0:
                        time.sleep(self._sleep)
                self._write_payload_batch(payloads, counts)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self._fetch_workers) as executor:
                    future_map = {
                        executor.submit(self._fetch_symbol_payload, plan): plan.symbol for plan in plans
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        payloads.append(future.result())
                self._write_payload_batch(payloads, counts)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        rows_written = sum(r.rows_written for r in counts.values())
        failures = sum(1 for r in counts.values() if r.fetch_failed)
        if do_backfill_derived:
            self.backfill_derived_daily_columns()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if record_audit:
            finished_at = datetime.now()
            self.record_fetch_audit(
                rid,
                started_at,
                finished_at,
                len(symbols),
                rows_written,
                failures,
                duration_ms,
            )
        return counts

    def quality_report(self) -> QualityReport:
        """对当前日线表做只读质量检查。"""
        return run_quality_checks(self._conn, self._table, self._quality)

    def read_daily_frame(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[Union[str, date]] = None,
        end: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        """读取日线为 DataFrame，供 Polars/张量因子使用。"""
        conds: List[str] = []
        params: List[Any] = []
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conds.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        if start is not None:
            conds.append("trade_date >= ?")
            params.append(pd.Timestamp(start).date())
        if end is not None:
            conds.append("trade_date <= ?")
            params.append(pd.Timestamp(end).date())
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        q = f"SELECT * FROM {self._table}{where} ORDER BY symbol, trade_date"
        return self._conn.execute(q, params).df()
