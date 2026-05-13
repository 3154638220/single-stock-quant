import pandas as pd

from scripts.fetch_stock import _recent_quality_summary


def test_recent_quality_summary_detects_null_invalid_and_gap():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-05"]),
            "open": [10.0, 10.2, 10.4],
            "high": [10.5, 10.1, 10.8],
            "low": [9.8, 10.0, 10.1],
            "close": [10.3, 10.4, None],
            "volume": [1000, 1200, 1300],
        }
    )

    summary = _recent_quality_summary("600000", df, window=30)

    assert summary.rows == 3
    assert summary.max_calendar_gap_days == 33
    assert summary.null_ohlcv == 1
    assert summary.invalid_ohlc == 1
    assert summary.violations(
        min_rows=5,
        max_gap_days=20,
        fail_on_nulls=True,
        fail_on_invalid_ohlc=True,
    ) == [
        "rows 3 < 5",
        "max_calendar_gap 33d > 20d",
        "null_ohlcv 1 > 0",
        "invalid_ohlc 1 > 0",
    ]


def test_recent_quality_summary_allows_opt_out_checks():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.2],
            "high": [10.5, 10.1],
            "low": [9.8, 10.0],
            "close": [10.3, None],
            "volume": [1000, 1200],
        }
    )

    summary = _recent_quality_summary("600000", df, window=30)

    assert summary.violations(
        min_rows=1,
        max_gap_days=20,
        fail_on_nulls=False,
        fail_on_invalid_ohlc=False,
    ) == []
