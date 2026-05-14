import pandas as pd

from src.features.weekly_trend import compute_weekly_trend_state


def _daily_from_weekly_closes(weekly_closes: list[float]) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01")
    for week, close in enumerate(weekly_closes):
        for day in range(5):
            date = start + pd.Timedelta(days=week * 7 + day)
            px = close + (day - 4) * 0.01
            rows.append(
                {
                    "trade_date": date,
                    "open": px,
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "volume": 100,
                }
            )
    return pd.DataFrame(rows)


def test_compute_weekly_trend_state_detects_bullish_tail():
    df = _daily_from_weekly_closes([10, 10.5, 11, 11.5, 12, 12.5])

    state = compute_weekly_trend_state(df, ma_windows=(2, 3))

    assert len(state) == len(df)
    assert state.iloc[-1] == "bullish"


def test_compute_weekly_trend_state_detects_bearish_tail():
    df = _daily_from_weekly_closes([12.5, 12, 11.5, 11, 10.5, 10])

    state = compute_weekly_trend_state(df, ma_windows=(2, 3))

    assert state.iloc[-1] == "bearish"


def test_weekly_state_does_not_use_future_week_close_before_week_ends():
    df = _daily_from_weekly_closes([12, 11, 10, 20])

    state = compute_weekly_trend_state(df, ma_windows=(2, 3))
    monday_of_jump_week = 15
    friday_of_jump_week = 19

    assert state.iloc[monday_of_jump_week] != state.iloc[friday_of_jump_week]
