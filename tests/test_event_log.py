import duckdb

from src.event_log import EventType, log_event, query_events


def test_event_log_ensures_schema_before_write_and_query():
    conn = duckdb.connect(":memory:")
    try:
        ok = log_event(
            conn,
            EventType.TREND_SIGNAL,
            {"signal": "buy"},
            run_id="test_run",
            symbol="600000",
            signal_date="2024-01-02",
        )

        assert ok
        rows = query_events(conn, event_type=EventType.TREND_SIGNAL)
        assert len(rows) == 1
        assert rows[0]["event_payload"] == {"signal": "buy"}
        assert rows[0]["symbol"] == "600000"
    finally:
        conn.close()
