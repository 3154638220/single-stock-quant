import json
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import requests

import src.env_check as env_check
from src.data_fetcher import akshare_resilience
from src.data_fetcher.data_quality import (
    QualityConfig,
    check_split_jump,
    run_quality_checks,
    validate_daily_frame,
)
from src.data_fetcher.index_benchmarks import (
    DEFAULT_INDEX_SPECS,
    IndexFetchSpec,
    parse_index_specs,
    standardize_index_daily,
)
from src.logging_config import JsonLineFormatter, _resolve_log_format, get_logger, setup_app_logging
from src.notify import WecomWebhookHandler, send_trend_signal
from src.signals import Signal


class _FakeResponse:
    def __init__(self, payload: dict | None = None, *, raise_http: bool = False) -> None:
        self.payload = payload if payload is not None else {"errcode": 0}
        self.raise_http = raise_http

    def raise_for_status(self) -> None:
        if self.raise_http:
            raise requests.exceptions.HTTPError("boom")

    def json(self) -> dict:
        if self.payload == {"bad": "json"}:
            raise ValueError("bad json")
        return self.payload


def test_wecom_webhook_payloads_and_failures(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_post(url: str, **kwargs):
        calls.append({"url": url, **kwargs})
        return _FakeResponse()

    monkeypatch.setattr("src.notify.requests.post", fake_post)
    handler = WecomWebhookHandler("https://example.test/hook", timeout=1.5, mention_all=True)

    assert handler.send_markdown("hello") is True
    assert calls[-1]["json"]["msgtype"] == "markdown"
    assert calls[-1]["json"]["markdown"]["mentioned_list"] == ["@all"]
    assert handler("plain alert") is True
    assert handler.send_text("text", mentioned_list=["u1"]) is True
    assert calls[-1]["json"]["text"]["mentioned_list"] == ["u1"]

    captured: list[str] = []

    class CaptureHandler(WecomWebhookHandler):
        def send_markdown(self, content: str) -> bool:
            captured.append(content)
            return True

    assert send_trend_signal(CaptureHandler("url"), "600930", "华电新能", Signal.BUY, 5.234, 3, "2026-05-13")
    assert "买入信号" in captured[0]
    assert "5.23" in captured[0]
    assert send_trend_signal(CaptureHandler("url"), "600930", "华电新能", Signal.SELL, 5.2, 1, "2026-05-14")
    assert "卖出信号" in captured[1]

    monkeypatch.setattr("src.notify.requests.post", lambda *a, **k: _FakeResponse({"errcode": 1, "errmsg": "bad"}))
    assert handler.send_markdown("bad code") is False
    assert handler.send_text("bad code") is False

    monkeypatch.setattr("src.notify.requests.post", lambda *a, **k: _FakeResponse({"bad": "json"}))
    assert handler.send_markdown("bad json") is False
    assert handler.send_text("bad json") is False

    def raise_request(*args, **kwargs):
        raise requests.exceptions.Timeout("timeout")

    monkeypatch.setattr("src.notify.requests.post", raise_request)
    assert handler.send_markdown("timeout") is False
    assert handler.send_text("timeout") is False


def test_logging_setup_json_text_and_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    record = logging.LogRecord("quant.test", logging.INFO, __file__, 10, "hello %s", ("world",), None)
    payload = json.loads(JsonLineFormatter().format(record))
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"

    try:
        raise RuntimeError("sample")
    except RuntimeError:
        exc_record = logging.LogRecord("quant.test", logging.ERROR, __file__, 11, "failed", (), sys.exc_info())
    assert "RuntimeError" in json.loads(JsonLineFormatter().format(exc_record))["exc_info"]

    monkeypatch.setenv("QUANT_LOG_FORMAT", "text")
    assert _resolve_log_format("json") == "text"
    monkeypatch.setenv("QUANT_LOG_FORMAT", "invalid")
    assert _resolve_log_format("invalid") == "json"

    logger_name = "quant.test.production_support"
    logger = logging.getLogger(logger_name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger = setup_app_logging(tmp_path, name=logger_name, level=logging.DEBUG, log_format="text")
    assert logger is setup_app_logging(tmp_path, name=logger_name, level=logging.INFO)
    assert len(logger.handlers) == 2
    logger.info("written")
    for handler in logger.handlers:
        handler.flush()
    assert list(tmp_path.glob(f"{logger_name}_*.log"))
    assert get_logger(logger_name) is logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_index_spec_parsing_and_standardization():
    assert parse_index_specs([]) == DEFAULT_INDEX_SPECS
    specs = parse_index_specs(["bench:000300:sh000300", "wide:csi932000:csi932000"])
    assert specs[0] == IndexFetchSpec("bench", "000300", "sh000300")
    assert specs[1].output_symbol == "932000"

    with pytest.raises(ValueError, match="name:output_symbol"):
        parse_index_specs(["bad"])
    with pytest.raises(ValueError, match="6 位代码"):
        parse_index_specs(["bad:12345:sh12345"])

    spec = IndexFetchSpec("bench", "000300", "sh000300")
    assert list(standardize_index_daily(pd.DataFrame(), spec).columns) == [
        "trade_date",
        "open",
        "symbol",
        "name",
        "source_symbol",
    ]

    raw = pd.DataFrame(
        {
            "日期": ["2024-01-03", "2024-01-02", "bad", "2024-01-02"],
            "开盘": ["11", "10", "12", "10.5"],
            "收盘": ["12", "10.2", "12.5", "10.8"],
            "最高": ["12.5", "10.9", "13", "11"],
            "最低": ["10.8", "9.9", "11", "10.1"],
            "成交量": ["100", "200", "300", "250"],
        }
    )
    out = standardize_index_daily(raw, spec)
    assert out["symbol"].tolist() == ["000300", "000300"]
    assert out["open"].tolist() == [10.5, 11.0]
    assert out["source_symbol"].unique().tolist() == ["sh000300"]

    with pytest.raises(ValueError, match="缺少"):
        standardize_index_daily(pd.DataFrame({"date": ["2024-01-02"], "close": [1.0]}), spec)


def test_data_quality_dataframe_database_and_split_checks():
    valid = pd.DataFrame(
        {
            "symbol": ["600000", "600000"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.2],
            "high": [10.5, 10.4],
            "low": [9.8, 10.0],
            "close": [10.3, 10.1],
            "volume": [1000, 1200],
        }
    )
    assert validate_daily_frame(valid).ok
    assert validate_daily_frame(pd.DataFrame()).notes == ["empty frame"]

    bad = pd.concat([valid, valid.iloc[[1]]], ignore_index=True)
    bad.loc[0, "high"] = 9.0
    bad.loc[1, "volume"] = None
    report = validate_daily_frame(bad, cfg=QualityConfig(null_ratio_max=0.1))
    assert not report.ok
    assert report.duplicate_pk_rows == 1
    assert report.ohlc_invalid_rows == 1
    assert report.null_ratio_violations
    assert "duplicate_pk=1" in report.summary()

    relaxed = validate_daily_frame(
        bad,
        cfg=QualityConfig(null_ratio_max=0.9, fail_on_ohlc_invalid=False),
    )
    assert not relaxed.ok
    assert relaxed.ohlc_invalid_rows == 1

    conn = duckdb.connect(":memory:")
    try:
        db_df = pd.DataFrame(
            {
                "symbol": ["600000", "600000", "600000", "600000"],
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-02-15", "2024-02-16"]),
                "open": [10.0, 10.0, 10.0, 10.0],
                "high": [10.5, 10.5, 9.5, 10.3],
                "low": [9.5, 9.5, 9.7, 9.8],
                "close": [10.2, 10.2, 10.1, None],
                "volume": [100, 100, 200, 300],
            }
        )
        conn.register("db_df", db_df)
        conn.execute("CREATE TABLE daily AS SELECT * FROM db_df")
        db_report = run_quality_checks(conn, "daily", QualityConfig(max_calendar_gap_days=20, null_ratio_max=0.1))
        assert not db_report.ok
        assert db_report.duplicate_pk_rows == 1
        assert db_report.ohlc_invalid_rows == 1
        assert db_report.large_gap_rows == 1
        assert db_report.notes

        relaxed_db_report = run_quality_checks(
            conn,
            "daily",
            QualityConfig(max_calendar_gap_days=20, null_ratio_max=0.9, fail_on_ohlc_invalid=False, fail_on_large_gaps=False),
        )
        assert not relaxed_db_report.ok
        assert relaxed_db_report.duplicate_pk_rows == 1
    finally:
        conn.close()

    missing_alerts = check_split_jump(pd.DataFrame({"symbol": ["600000"]}))
    assert len(missing_alerts) == 1
    for col in ("volume", "close", "trade_date", "pct_chg"):
        assert col in missing_alerts[0]
    alerts = check_split_jump(
        pd.DataFrame(
            {
                "symbol": ["600000", "600001"],
                "trade_date": ["2024-01-02", "2024-01-03"],
                "pct_chg": [0.6, 0.1],
                "close": [10.0, 11.0],
                "volume": [1000, 0],
            }
        )
    )
    assert len(alerts) == 1
    assert "600000" in alerts[0]


def test_akshare_resilience_config_cache_and_retries(tmp_path: Path):
    cfg = {
        "akshare": {
            "request_timeout_sec": 0.2,
            "http_connect_timeout_sec": 0.1,
            "http_read_timeout_sec": 0.2,
            "http_transport_retries": 1,
            "http_retry_backoff_sec": 0.0,
            "http_pool_connections": 2,
            "http_pool_maxsize": 3,
            "retry_delay_sec": 0.0,
            "cache_dir": str(tmp_path),
            "stale_cache_on_error": True,
            "hot_list_cache_ttl_sec": 7,
        }
    }
    conf = akshare_resilience.load_akshare_resilience_config(cfg)
    assert conf.http_pool_connections == 4
    assert conf.http_pool_maxsize == 4
    assert conf.cache_dir == tmp_path
    assert akshare_resilience.resolve_cache_ttl_seconds("hot_list", cfg) == 7
    assert akshare_resilience.resolve_cache_ttl_seconds("unknown", cfg) == 300.0
    assert akshare_resilience.call_with_timeout(lambda: "ok", timeout_sec=0) == "ok"

    attempts = {"first": 0}

    def empty_then_data() -> pd.DataFrame:
        attempts["first"] += 1
        if attempts["first"] == 1:
            return pd.DataFrame()
        return pd.DataFrame({"x": [1]})

    df = akshare_resilience.fetch_dataframe_with_cache(
        [("empty_then_data", empty_then_data)],
        cache_key="a/b c",
        cache_ttl_sec=60,
        retries=2,
        timeout_sec=1,
        retry_delay_sec=0,
        cfg=cfg,
    )
    assert df["x"].tolist() == [1]
    cache_files = list(tmp_path.glob("a_b_c.json"))
    assert len(cache_files) == 1
    cached, age, source = akshare_resilience._load_cached_dataframe(cache_files[0])
    assert cached is not None
    assert age is not None
    assert source == "empty_then_data"

    def fail() -> pd.DataFrame:
        raise RuntimeError("network down")

    fallback = akshare_resilience.fetch_dataframe_with_cache(
        [("fail", fail)],
        cache_key="a/b c",
        cache_ttl_sec=60,
        retries=1,
        timeout_sec=1,
        retry_delay_sec=0,
        cfg=cfg,
    )
    assert fallback["x"].tolist() == [1]

    no_cache_cfg = {"akshare": {**cfg["akshare"], "cache_dir": str(tmp_path / "none"), "stale_cache_on_error": False}}
    with pytest.raises(RuntimeError, match="exhausted"):
        akshare_resilience.fetch_dataframe_with_cache(
            [("fail", fail)],
            cache_key="missing",
            cache_ttl_sec=60,
            retries=1,
            timeout_sec=1,
            retry_delay_sec=0,
            cfg=no_cache_cfg,
        )

    bad_cache = tmp_path / "bad.json"
    bad_cache.write_text("{", encoding="utf-8")
    assert akshare_resilience._load_cached_dataframe(bad_cache) == (None, None, None)


def test_env_check_success_and_failure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = {
        "paths": {"duckdb_path": str(tmp_path / "market.duckdb")},
        "akshare": {"adjust": "qfq", "request_timeout_sec": 1.0},
    }
    monkeypatch.setattr(env_check, "load_config", lambda config=None: cfg)
    monkeypatch.setattr(env_check, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        env_check,
        "fetch_a_share_daily",
        lambda *args, **kwargs: pd.DataFrame({"trade_date": ["2024-01-02"], "open": [1.0]}),
    )
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "quant-system")

    assert env_check.run_checks(config=None, quiet=True) == 0

    monkeypatch.setattr(env_check, "fetch_a_share_daily", lambda *args, **kwargs: pd.DataFrame())
    assert env_check.run_checks(config=None, quiet=True) == 1

    def raise_fetch(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(env_check, "fetch_a_share_daily", raise_fetch)
    monkeypatch.setattr(env_check.socket, "getaddrinfo", lambda *args, **kwargs: [("family", "type", "proto", "canon", ("127.0.0.1", 443))])
    assert "127.0.0.1" in env_check._dns_summary(["example.test"])
    assert env_check.run_checks(config=None, quiet=True) == 1
