import pandas as pd

from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams, TrendMode


def test_single_stock_backtest_closes_end_position_and_costs():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 11, 12, 11, 10],
            "high": [10, 10, 10, 10, 11, 12, 11, 10],
            "low": [10, 10, 10, 10, 11, 12, 11, 10],
            "close": [10, 10, 10, 10, 11, 12, 13, 14],
            "volume": [100] * 8,
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        cost_bps=10,
        initial_capital=10000,
    )
    assert res.n_trades == 1
    assert res.trade_log.iloc[0]["exit_reason"] == "end"
    assert res.total_return > 0


def test_single_stock_backtest_delays_limit_up_buy():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 11, 12.1, 13.2, 13],
            "high": [10, 10, 10, 10, 11, 12.1, 13.2, 13],
            "low": [10, 10, 10, 10, 11, 12.1, 13.2, 13],
            "close": [10, 10, 10, 10, 11, 12, 13, 14],
            "volume": [100] * 8,
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        cost_bps=0,
        initial_capital=10000,
    )
    assert res.trade_log.iloc[0]["buy_date"] == pd.Timestamp("2024-01-08")


def test_single_stock_backtest_cancels_delayed_buy_if_sell_signal_arrives_first():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 11, 12.1, 13.2, 8],
            "high": [10, 10, 10, 10, 11, 12.1, 13.2, 8],
            "low": [10, 10, 10, 10, 11, 12.0, 9.0, 8],
            "close": [10, 10, 10, 10, 11, 12, 9, 8],
            "volume": [100] * 8,
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        cost_bps=0,
        initial_capital=10000,
    )
    assert res.n_trades == 0


def test_single_stock_backtest_delays_sell_on_suspended_day():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=9),
            "open": [10, 10, 10, 10, 11, 12, 9, 8, 7.5],
            "high": [10, 10, 10, 10, 11, 12, 9, 8, 7.5],
            "low": [10, 10, 10, 10, 11, 12, 9, 8, 7.5],
            "close": [10, 10, 10, 10, 11, 12, 9, 8, 7],
            "volume": [100, 100, 100, 100, 100, 100, 100, 0, 100],
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        cost_bps=0,
        initial_capital=10000,
    )
    assert res.n_trades == 1
    assert res.trade_log.iloc[0]["sell_date"] == pd.Timestamp("2024-01-09")
    assert res.trade_log.iloc[0]["exit_reason"] == "signal"
    assert res.trade_log.iloc[0]["sell_price"] == 7.5
