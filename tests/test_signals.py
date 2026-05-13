import pandas as pd

from src.indicators import DKTrendParams, TrendMode
from src.signals import Position, Signal, compute_consensus_trend, generate_signals


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


def test_generate_signals_volume_filter_delays_buy_until_confirmed():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 11, 12, 11, 10],
            "high": [10, 10, 10, 10, 11, 12, 11, 10],
            "low": [10, 10, 10, 10, 11, 12, 11, 10],
            "close": [10, 10, 10, 10, 11, 12, 9, 8],
            "volume": [100, 100, 100, 100, 10, 300, 100, 100],
        }
    )
    records = generate_signals(
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        volume_confirm=True,
        volume_lookback=3,
        volume_ratio_min=1.0,
    )
    buys = [r for r in records if r.signal == Signal.BUY]
    assert len(buys) == 1
    assert buys[0].trade_date == pd.Timestamp("2024-01-06")


def test_consensus_trend_counts_modes_and_emits_signal():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=12),
            "open": [10, 10, 10, 10, 11, 12, 13, 13, 12, 11, 10, 9],
            "high": [10, 10, 10, 10, 11, 12, 13, 13, 12, 11, 10, 9],
            "low": [10, 10, 10, 10, 11, 12, 13, 13, 12, 11, 10, 9],
            "close": [10, 10, 10, 10, 11, 12, 13, 13, 12, 11, 10, 9],
            "volume": [100] * 12,
        }
    )
    trend = compute_consensus_trend(df, n_agree=2, base_params=DKTrendParams(boll_window=3, ma_fast=2, ma_slow=3))
    assert {"consensus_red_count", "consensus_green_count", "consensus_n_agree"} <= set(trend.columns)
    assert trend["dk_signal"].isin(["buy", "sell"]).any()
