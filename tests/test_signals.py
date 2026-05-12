import pandas as pd

from src.indicators import DKTrendParams, TrendMode
from src.signals import Position, Signal, generate_signals


def test_generate_signals_state_machine_does_not_repeat_buy():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 11, 12, 11, 10],
            "high": [10, 10, 10, 10, 11, 12, 11, 10],
            "low": [10, 10, 10, 10, 11, 12, 11, 10],
            "close": [10, 10, 10, 10, 11, 12, 9, 8],
        }
    )
    records = generate_signals(df, DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3))
    buy_records = [r for r in records if r.signal == Signal.BUY]
    sell_records = [r for r in records if r.signal == Signal.SELL]
    assert len(buy_records) == 1
    assert len(sell_records) == 1
    assert buy_records[0].position_after == Position.LONG
    assert sell_records[0].position_after == Position.FLAT
